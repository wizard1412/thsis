"""Plot accuracy/MAE/weighted kappa vs number of features for fewshot_v2 experiments.

Ground truth = consensus (mean of Dr. Tan + Dr. Chien), matching multi_model_comparison.py.
Predictions = median of 10 evaluations, rounded to integer in [0, 4].
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

RESULTS_DIR = Path(__file__).parent / "results"
SCORE_COLS = [str(i) for i in range(10)]
RANGE_N = range(1, 5)
MAX_SCORE = 4

MODEL_CONFIGS = {
    "ds7b":  {"exp_prefix": "fewshot_v2_ds7b_freq",  "scored_csv": "scored_by_deepseek-r1_7b.csv"},
    "ds32b": {"exp_prefix": "fewshot_v2_ds32b_freq", "scored_csv": "scored_by_deepseek-r1_32b.csv"},
}


def load_run(n: int, cfg: dict) -> pd.DataFrame:
    csv = RESULTS_DIR / f"{cfg['exp_prefix']}{n}" / cfg["scored_csv"]
    return pd.read_csv(csv)


def predictions(df: pd.DataFrame) -> np.ndarray:
    """Median of 10 evaluations per session (to match multi_model_comparison.py)."""
    arr = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
    return np.nanmedian(arr, axis=1)


def consensus(df: pd.DataFrame) -> np.ndarray:
    return ((df["label_Dr. Tan"].astype(float) + df["label_Dr. Chien"].astype(float)) / 2).to_numpy()


def metrics(pred, gt):
    mask = ~(np.isnan(pred) | np.isnan(gt))
    yt = np.clip(np.round(gt[mask]).astype(int), 0, MAX_SCORE)
    yp = np.clip(np.round(pred[mask]).astype(int), 0, MAX_SCORE)
    exact = float(np.mean(yp == yt))
    within1 = float(np.mean(np.abs(yp - yt) <= 1))
    mae = float(np.mean(np.abs(yp - yt)))
    try:
        kappa = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    except Exception:
        kappa = np.nan
    return exact, within1, mae, kappa, int(mask.sum())


def compute_summary(cfg: dict, range_n) -> pd.DataFrame:
    rows = []
    for n in range_n:
        df = load_run(n, cfg)
        pred = predictions(df)
        gt = consensus(df)
        exact, within1, mae, kappa, nsamp = metrics(pred, gt)
        rows.append({
            "n_features": n,
            "exact":   exact,
            "within1": within1,
            "mae":     mae,
            "kappa":   kappa,
            "n_samples": nsamp,
        })
    return pd.DataFrame(rows)


def plot_single(args):
    cfg = MODEL_CONFIGS[args.model]
    summary = compute_summary(cfg, RANGE_N)
    print(summary.to_string(index=False))
    out_csv = Path(__file__).parent / f"freq_accuracy_summary_{args.model}.csv"
    summary.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        (axes[0, 0], "exact",   "Exact match accuracy",             "Accuracy"),
        (axes[0, 1], "within1", "Within-1 accuracy",                "Accuracy"),
        (axes[1, 0], "mae",     "MAE",                              "MAE"),
        (axes[1, 1], "kappa",   "Quadratic weighted kappa",         "Kappa"),
    ]
    for ax, col, title, ylabel in specs:
        ax.plot(summary.n_features, summary[col], "o-", color="black")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(RANGE_N))
        ax.grid(alpha=0.3)
        if col in ("mae", "kappa"):
            ax.set_xlabel("Number of features (top-N by frequency)")

    fig.suptitle(f"fewshot_v2_{args.model}: feature count vs metrics (vs consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / f"freq_accuracy_trend_{args.model}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_png}\nSaved: {out_csv}")


def plot_combined():
    range_common = range(1, 5)
    summaries = {m: compute_summary(MODEL_CONFIGS[m], range_common) for m in MODEL_CONFIGS}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        (axes[0, 0], "exact",   "Exact match accuracy",       "Accuracy"),
        (axes[0, 1], "within1", "Within-1 accuracy",          "Accuracy"),
        (axes[1, 0], "mae",     "MAE",                        "MAE"),
        (axes[1, 1], "kappa",   "Quadratic weighted kappa",   "Kappa"),
    ]
    markers = {"ds7b": "o-", "ds32b": "s--"}
    for ax, col, title, ylabel in specs:
        for m, s in summaries.items():
            ax.plot(s.n_features, s[col], markers[m], label=m)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range_common))
        ax.grid(alpha=0.3)
        ax.legend()
        if col in ("mae", "kappa"):
            ax.set_xlabel("Number of features (top-N by frequency)")

    fig.suptitle("fewshot_v2 ds7b vs ds32b: feature count vs metrics (vs consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / "freq_accuracy_trend_combined.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot freq metrics trend")
    parser.add_argument("--model", "-m", choices=list(MODEL_CONFIGS.keys()) + ["combined"],
                        default="ds7b", help="Model to plot (or 'combined' for overlay)")
    args = parser.parse_args()

    if args.model == "combined":
        plot_combined()
    else:
        plot_single(args)


if __name__ == "__main__":
    main()
