from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import basic_clean, build_features, load_raw

IMG_DIR = Path("report/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main():
    raw = load_raw()
    df = basic_clean(raw)
    feats = build_features(df)

    fig, ax = plt.subplots(figsize=(5, 4))
    df["team1_win"].value_counts().rename({0: "team1 lose", 1: "team1 win"}).plot.bar(ax=ax, color=["#d62728", "#2ca02c"])
    ax.set_title("Target distribution (per-map rows)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "01_target_balance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    df["map_name"].value_counts().plot.bar(ax=ax)
    ax.set_title("Matches per map")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "02_map_counts.png", dpi=120)
    plt.close(fig)

    wr_by_map = df.groupby("map_name")["team1_win"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    wr_by_map.plot.bar(ax=ax)
    ax.axhline(df["team1_win"].mean(), color="red", linestyle="--", label="global mean")
    ax.set_title("team1 winrate by map")
    ax.set_ylabel("winrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG_DIR / "03_winrate_by_map.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    df["datetime"].dt.to_period("M").value_counts().sort_index().plot(ax=ax)
    ax.set_title("Matches per month")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "04_matches_over_time.png", dpi=120)
    plt.close(fig)

    key_feats = ["diff_prior_winrate", "diff_winrate_last_n", "diff_avg_adr_last_n", "diff_avg_kddiff_last_n"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, key_feats):
        sns.boxplot(data=feats, x="team1_win", y=col, ax=ax)
        ax.set_title(col)
    fig.suptitle("Diff features (team1 − team2) vs target", y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "05_diff_features_vs_target.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    num_feats = feats.select_dtypes("number").drop(columns=["game_id", "match_id", "team1_id", "team2_id"], errors="ignore")
    corr = num_feats.corr(numeric_only=True)["team1_win"].abs().sort_values(ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(num_feats[corr].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap — top 15 features by |corr with target|")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "06_corr_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"Saved 6 figures to {IMG_DIR}")


if __name__ == "__main__":
    main()
