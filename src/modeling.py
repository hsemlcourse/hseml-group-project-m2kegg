import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import FEATURE_COLS_CAT, FEATURE_COLS_NUMERIC, TARGET

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORT_DIR = Path("report")
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

BASELINE_NUMERIC = ["bestOf", "year", "month", "dayofweek", "hour"]
BASELINE_CAT = ["map_name"]


def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
    }


def make_preprocessor(num_cols, cat_cols, scale: bool = True):
    num_steps = [("scaler", StandardScaler())] if scale else []
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )




def run_baseline(train, val, test) -> dict:
    """Out-of-the-box Logistic Regression on raw minimal features (no engineered features)."""
    num_cols, cat_cols = BASELINE_NUMERIC, BASELINE_CAT
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(num_cols, cat_cols, scale=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(train[num_cols + cat_cols], train[TARGET])
    val_m = evaluate(pipe, val[num_cols + cat_cols], val[TARGET])
    test_m = evaluate(pipe, test[num_cols + cat_cols], test[TARGET])
    joblib.dump(pipe, MODELS_DIR / "baseline_logreg.joblib")
    return {"name": "baseline_logreg_raw", "val": val_m, "test": test_m}




def experiment_space():
    """(name, estimator, param_grid, use_scaling) — hyperparameter search per model."""
    return [
        (
            "logreg",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"]},
            True,
        ),
        (
            "knn",
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 15, 31, 51], "clf__weights": ["uniform", "distance"]},
            True,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 20]},
            False,
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 5],
            },
            False,
        ),
        (
            "gbdt",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
            False,
        ),
    ]


def run_experiments(train, val, test, num_cols=None, cat_cols=None):
    num_cols = num_cols or FEATURE_COLS_NUMERIC
    cat_cols = cat_cols or FEATURE_COLS_CAT

    X_train, y_train = train[num_cols + cat_cols], train[TARGET]
    X_val, y_val = val[num_cols + cat_cols], val[TARGET]
    X_test, y_test = test[num_cols + cat_cols], test[TARGET]

    cv = TimeSeriesSplit(n_splits=4)

    results = []
    best = None
    for name, estimator, grid, scale in experiment_space():
        pipe = Pipeline(
            [
                ("prep", make_preprocessor(num_cols, cat_cols, scale=scale)),
                ("clf", estimator),
            ]
        )
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        val_m = evaluate(gs.best_estimator_, X_val, y_val)
        row = {
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_roc_auc": float(gs.best_score_),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        results.append(row)
        print(f"[{name}] cv_auc={gs.best_score_:.4f}  val_auc={val_m['roc_auc']:.4f}  params={row['best_params']}")
        if best is None or val_m["roc_auc"] > best[1]:
            best = (name, val_m["roc_auc"], gs.best_estimator_)

    best_name, _, best_model = best
    test_m = evaluate(best_model, X_test, y_test)
    print(f"\n[BEST={best_name}] TEST metrics: {test_m}")
    joblib.dump(best_model, MODELS_DIR / f"best_{best_name}.joblib")
    return results, {"name": best_name, "test": test_m}


def main():
    train, val, test = load_splits()
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"target balance train={train[TARGET].mean():.3f}  val={val[TARGET].mean():.3f}  test={test[TARGET].mean():.3f}")

    baseline = run_baseline(train, val, test)
    print(f"\n[BASELINE] val={baseline['val']}  test={baseline['test']}")

    results, best = run_experiments(train, val, test)

    out = {"baseline": baseline, "experiments": results, "best_on_test": best}
    (REPORT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(REPORT_DIR / "experiments.csv", index=False)
    print(f"\nSaved {REPORT_DIR / 'results.json'} and experiments.csv")


if __name__ == "__main__":
    main()
