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
MAX_SCORE = 4

MODEL_CONFIGS = {
    "ds7b_v2":        {"exp_prefix": "fewshot_v2_ds7b_freq",       "scored_csv": "scored_by_deepseek-r1_7b.csv"},
    "ds32b_v2_amp":   {"exp_prefix": "fewshot_v2_ds32b_amp_freq",  "scored_csv": "scored_by_deepseek-r1_32b.csv"},
    "ds32b_v1":       {"exp_prefix": "fewshot_v1_ds32b_freq",      "scored_csv": "scored_by_deepseek-r1_32b.csv"},
    "ds32b_v1_amp":   {"exp_prefix": "fewshot_v1_ds32b_amp_freq",  "scored_csv": "scored_by_deepseek-r1_32b.csv"},
}


def available_range(cfg: dict, max_n: int = 10) -> list:
    """Return list of n for which the result CSV exists."""
    out = []
    for n in range(1, max_n + 1):
        csv = RESULTS_DIR / f"{cfg['exp_prefix']}{n}" / cfg["scored_csv"]
        if csv.exists():
            out.append(n)
    return out


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


def residuals(pred, gt):
    """Return (gt_int, resid) arrays after masking NaNs and rounding."""
    mask = ~(np.isnan(pred) | np.isnan(gt))
    gt_int = np.clip(np.round(gt[mask]).astype(int), 0, MAX_SCORE)
    pred_int = np.clip(np.round(pred[mask]).astype(int), 0, MAX_SCORE)
    return gt_int, pred_int - gt_int


def _add_residual_scatter(ax, gt_int, resid, label=None, color=None):
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, size=len(gt_int))
    kwargs = dict(alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
    if color:
        kwargs["color"] = color
    ax.scatter(gt_int + jitter, resid, label=label, **kwargs)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Ground Truth (consensus)")
    ax.set_ylabel("Residual (pred − GT)")
    ax.set_xticks(range(MAX_SCORE + 1))
    ax.set_yticks(range(-MAX_SCORE, MAX_SCORE + 1))
    ax.grid(alpha=0.3)


def _add_residual_hist(ax, resid, label=None, color=None):
    bins = np.arange(-MAX_SCORE - 0.5, MAX_SCORE + 1.5)
    kwargs = dict(bins=bins, edgecolor="k", alpha=0.7)
    if color:
        kwargs["color"] = color
    ax.hist(resid, label=label, **kwargs)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    mean_r = float(np.mean(resid))
    ax.axvline(mean_r, color="blue", linestyle="-", linewidth=1.5,
               label=f"mean={mean_r:+.2f}")
    ax.set_xlabel("Residual (pred − GT)")
    ax.set_ylabel("Count")
    ax.set_xticks(range(-MAX_SCORE, MAX_SCORE + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def plot_residuals_single(args):
    cfg = MODEL_CONFIGS[args.model]
    range_n = available_range(cfg, max_n=args.max_n)
    if not range_n:
        print(f"No results found for {args.model} (prefix={cfg['exp_prefix']})")
        return

    # pick specific n or all available
    ns = [args.n] if (args.n is not None) else range_n
    invalid = [n for n in ns if n not in range_n]
    if invalid:
        print(f"n={invalid} not available for {args.model}. Available: {range_n}")
        return

    fig, axes = plt.subplots(len(ns), 2, figsize=(12, 4 * len(ns)),
                             squeeze=False)
    for row, n in enumerate(ns):
        df = load_run(n, cfg)
        gt_int, resid = residuals(predictions(df), consensus(df))
        _add_residual_scatter(axes[row, 0], gt_int, resid)
        axes[row, 0].set_title(f"{args.model} (n={n}): Residual vs GT")
        _add_residual_hist(axes[row, 1], resid)
        axes[row, 1].set_title(f"{args.model} (n={n}): Residual distribution")

    fig.suptitle(f"{args.model}: residual analysis (pred − consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / f"residual_{args.model}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def plot_residuals_combined(args):
    """Overlay all models' residuals on the same scatter + histogram."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (m, cfg) in enumerate(MODEL_CONFIGS.items()):
        range_n = available_range(cfg, max_n=args.max_n)
        if not range_n:
            continue
        n = args.n if (args.n is not None and args.n in range_n) else range_n[-1]
        df = load_run(n, cfg)
        gt_int, resid = residuals(predictions(df), consensus(df))
        color = colors[idx % len(colors)]
        _add_residual_scatter(ax_scatter, gt_int, resid,
                              label=f"{m} (n={n})", color=color)
        mean_r = float(np.mean(resid))
        bins = np.arange(-MAX_SCORE - 0.5, MAX_SCORE + 1.5)
        ax_hist.hist(resid, bins=bins, alpha=0.5, edgecolor="k",
                     color=color, label=f"{m} (n={n}, mean={mean_r:+.2f})")

    ax_scatter.set_title("Residual vs GT (all models)")
    ax_scatter.legend(fontsize=8)
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Residual (pred − GT)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Residual distribution (all models)")
    ax_hist.set_xticks(range(-MAX_SCORE, MAX_SCORE + 1))
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(fontsize=8)

    fig.suptitle("Residual analysis across experiments (pred − consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / "residual_combined.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def plot_single(args):
    cfg = MODEL_CONFIGS[args.model]
    range_n = available_range(cfg, max_n=args.max_n)
    if not range_n:
        print(f"No results found for {args.model} (prefix={cfg['exp_prefix']})")
        return
    summary = compute_summary(cfg, range_n)
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
        ax.set_xticks(range_n)
        ax.grid(alpha=0.3)
        if col in ("mae", "kappa"):
            ax.set_xlabel("Number of features (top-N by frequency)")

    fig.suptitle(f"{args.model}: feature count vs metrics (vs consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / f"freq_accuracy_trend_{args.model}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_png}\nSaved: {out_csv}")


def plot_combined(max_n: int = 10):
    summaries = {}
    for m, cfg in MODEL_CONFIGS.items():
        rn = available_range(cfg, max_n=max_n)
        if rn:
            summaries[m] = compute_summary(cfg, rn)
    if not summaries:
        print("No results found for any model.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        (axes[0, 0], "exact",   "Exact match accuracy",       "Accuracy"),
        (axes[0, 1], "within1", "Within-1 accuracy",          "Accuracy"),
        (axes[1, 0], "mae",     "MAE",                        "MAE"),
        (axes[1, 1], "kappa",   "Quadratic weighted kappa",   "Kappa"),
    ]
    all_n = sorted({int(n) for s in summaries.values() for n in s.n_features})
    for ax, col, title, ylabel in specs:
        for m, s in summaries.items():
            ax.plot(s.n_features, s[col], "o-", label=m, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_n)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if col in ("mae", "kappa"):
            ax.set_xlabel("Number of features (top-N by frequency)")

    fig.suptitle("Feature count vs metrics across experiments (vs consensus)", fontsize=13)
    fig.tight_layout()
    out_png = Path(__file__).parent / "freq_accuracy_trend_combined.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot freq metrics trend")
    parser.add_argument("--model", "-m", choices=list(MODEL_CONFIGS.keys()) + ["combined"],
                        default="ds32b_v1_amp", help="Model to plot (or 'combined' for overlay)")
    parser.add_argument("--max-n", type=int, default=13,
                        help="Max feature count to include (e.g. 4 for freq1-4)")
    parser.add_argument("--residual", action="store_true",
                        help="Plot residual analysis instead of metric trend")
    parser.add_argument("--n", type=int, default=None,
                        help="Feature count to use for residual plot (default: all available)")
    args = parser.parse_args()

    if args.residual:
        if args.model == "combined":
            plot_residuals_combined(args)
        else:
            plot_residuals_single(args)
    elif args.model == "combined":
        plot_combined(max_n=args.max_n)
    else:
        plot_single(args)


if __name__ == "__main__":
    main()
