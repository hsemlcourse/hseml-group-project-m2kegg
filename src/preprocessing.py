"""Тут был сделан хитрый feature-engineering: добавлены статы команды за предыдущие матчи (винрейт общий, винрейт против команды соперника и т.д.). По умолчанию берутся последние 10
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/cs2_all_tiers_games.csv")
PROCESSED_DIR = Path("data/processed")

PLAYER_STAT_COLS = ["kills", "deaths", "assists", "adr", "kast", "kddiff"]
ROLL_WINDOW = 10


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df[not df["is_total"]].copy()

    df = df.dropna(subset=["datetime", "map_name", "bestOf", "team1_id", "team2_id", "team1_win"])

    df = df.drop_duplicates(subset=["match_id", "game_id"])

    df["bestOf"] = df["bestOf"].astype(int)
    df["team1_win"] = df["team1_win"].astype(int)
    df["team1_id"] = df["team1_id"].astype(int)
    df["team2_id"] = df["team2_id"].astype(int)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _team_map_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for side in ("team1", "team2"):
        other = "team2" if side == "team1" else "team1"
        cols = {
            "game_id": "game_id",
            "match_id": "match_id",
            "datetime": "datetime",
            "map_name": "map_name",
            f"{side}_id": "team_id",
            f"{other}_id": "opponent_id",
            "team1_win": "team1_win",
        }
        sub = df[list(cols.keys())].rename(columns=cols).copy()
        sub["is_team1"] = int(side == "team1")
        # team_won from team's own perspective
        sub["team_won"] = sub["team1_win"] if side == "team1" else (1 - sub["team1_win"])
        # Aggregate team's 5 player stats on this map (avg across players)
        for stat in PLAYER_STAT_COLS:
            stat_cols = [f"{side}_player{i}_{stat}" for i in range(1, 6)]
            sub[f"team_{stat}"] = df[stat_cols].mean(axis=1).values
        records.append(sub)
    return pd.concat(records, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _add_rolling_team_features(team_rows: pd.DataFrame, window: int) -> pd.DataFrame:
    team_rows = team_rows.copy()
    g = team_rows.groupby("team_id", sort=False)
    team_rows["team_games_played"] = g.cumcount()
    team_rows["team_prior_wins"] = g["team_won"].apply(lambda s: s.shift(1).cumsum()).reset_index(level=0, drop=True)
    team_rows["team_prior_winrate"] = team_rows["team_prior_wins"] / team_rows["team_games_played"]

    shifted = g["team_won"].shift(1)
    team_rows["team_winrate_last_n"] = (
        shifted.groupby(team_rows["team_id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    for stat in PLAYER_STAT_COLS:
        col = f"team_{stat}"
        shifted_stat = g[col].shift(1)
        team_rows[f"team_avg_{stat}_last_n"] = (
            shifted_stat.groupby(team_rows["team_id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    gm = team_rows.groupby(["team_id", "map_name"], sort=False)
    team_rows["team_map_games_played"] = gm.cumcount()
    team_rows["team_map_prior_wins"] = (
        gm["team_won"].apply(lambda s: s.shift(1).cumsum()).reset_index(level=[0, 1], drop=True)
    )
    team_rows["team_map_prior_winrate"] = team_rows["team_map_prior_wins"] / team_rows["team_map_games_played"]

    return team_rows


def _h2h_feature(df: pd.DataFrame) -> pd.Series:
    pair_stats = {}
    h2h = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        t1, t2 = int(row.team1_id), int(row.team2_id)
        key = (min(t1, t2), max(t1, t2))
        history = pair_stats.get(key, [])
        if history:
            # считаем либо выигрыш первой команды, либо проигрыш второй
            wins = sum(1 for h in history if h[0] == t1 and h[1] == 1) + sum(
                1 for h in history if h[0] == t2 and h[1] == 0
            )
            h2h[i] = wins / len(history)
        pair_stats.setdefault(key, []).append((t1, int(row.team1_win)))
    return pd.Series(h2h, index=df.index, name="h2h_team1_winrate")


def build_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Produce a model-ready frame of pre-match features + target."""
    team_rows = _team_map_stats(df)
    team_rows = _add_rolling_team_features(team_rows, window=window)

    keep = [
        "game_id",
        "team_id",
        "is_team1",
        "team_games_played",
        "team_prior_winrate",
        "team_winrate_last_n",
        "team_map_games_played",
        "team_map_prior_winrate",
    ] + [f"team_avg_{s}_last_n" for s in PLAYER_STAT_COLS]
    tr = team_rows[keep].rename(
        columns={
            "team_games_played": "games_played",
            "team_prior_winrate": "prior_winrate",
            "team_winrate_last_n": "winrate_last_n",
            "team_map_games_played": "map_games_played",
            "team_map_prior_winrate": "map_prior_winrate",
            **{f"team_avg_{s}_last_n": f"avg_{s}_last_n" for s in PLAYER_STAT_COLS},
        }
    )

    t1 = tr[tr["is_team1"] == 1].drop(columns=["is_team1"]).add_prefix("t1_").rename(columns={"t1_game_id": "game_id"})
    t2 = tr[tr["is_team1"] == 0].drop(columns=["is_team1"]).add_prefix("t2_").rename(columns={"t2_game_id": "game_id"})
    merged = df[
        ["game_id", "match_id", "datetime", "tournament", "map_name", "bestOf", "team1_id", "team2_id", "team1_win"]
    ].merge(t1, on="game_id", how="left").merge(t2, on="game_id", how="left")

    # Diff фичи (показали себя максимально круто)
    for feat in [
        "prior_winrate",
        "winrate_last_n",
        "map_prior_winrate",
        "games_played",
        "map_games_played",
    ] + [f"avg_{s}_last_n" for s in PLAYER_STAT_COLS]:
        merged[f"diff_{feat}"] = merged[f"t1_{feat}"] - merged[f"t2_{feat}"]

    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["h2h_team1_winrate"] = _h2h_feature(merged)

    merged["year"] = merged["datetime"].dt.year
    merged["month"] = merged["datetime"].dt.month
    merged["dayofweek"] = merged["datetime"].dt.dayofweek
    merged["hour"] = merged["datetime"].dt.hour

    rolling_cols = [c for c in merged.columns if c.startswith(("t1_", "t2_", "diff_", "h2h_"))]
    merged[rolling_cols] = merged[rolling_cols].fillna({c: 0.0 for c in rolling_cols})
    for c in rolling_cols:
        if "winrate" in c and c.startswith(("t1_", "t2_", "h2h_")):
            merged[c] = merged[c].replace(0.0, np.nan).fillna(0.5)

    return merged


def time_split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Сплитуем по хронологии: train - предыдущий период, val - период дальше, test - последние данные по datetime
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


FEATURE_COLS_NUMERIC = (
    ["bestOf", "year", "month", "dayofweek", "hour", "h2h_team1_winrate"]
    + [f"t1_{x}" for x in ["prior_winrate", "winrate_last_n", "map_prior_winrate", "games_played", "map_games_played"]]
    + [f"t2_{x}" for x in ["prior_winrate", "winrate_last_n", "map_prior_winrate", "games_played", "map_games_played"]]
    + [f"t1_avg_{s}_last_n" for s in PLAYER_STAT_COLS]
    + [f"t2_avg_{s}_last_n" for s in PLAYER_STAT_COLS]
    + [f"diff_{x}" for x in ["prior_winrate", "winrate_last_n", "map_prior_winrate", "games_played", "map_games_played"]]
    + [f"diff_avg_{s}_last_n" for s in PLAYER_STAT_COLS]
)
FEATURE_COLS_CAT = ["map_name"]
TARGET = "team1_win"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    print(f"[load_raw] {df.shape}")
    df = basic_clean(df)
    print(f"[basic_clean] {df.shape}")
    feats = build_features(df)
    print(f"[build_features] {feats.shape}")

    feats.to_csv(PROCESSED_DIR / "features.csv", index=False)
    train, val, test = time_split(feats)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[split] ranges: train[{train['datetime'].min()} .. {train['datetime'].max()}]")
    print(f"                  val[{val['datetime'].min()} .. {val['datetime'].max()}]")
    print(f"                 test[{test['datetime'].min()} .. {test['datetime'].max()}]")


if __name__ == "__main__":
    main()
