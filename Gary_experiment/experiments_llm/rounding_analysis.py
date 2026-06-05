"""Compare three GT-rounding strategies across all experiments.

Approaches
----------
A  concordant_only : only sessions where Dr. Tan == Dr. Chien (n=22)
                     GT is integer, pred rounded half-up
                     ACC, AAcc, MAE, Kappa all reported

B  mixed           : all 53 sessions
                     ACC / Kappa : half-up GT, half-up pred
                     MAE         : raw median pred, continuous GT (Dallen-style)

C  all_halfup      : all 53 sessions, half-up for both pred and GT
                     ACC, AAcc, MAE, Kappa all reported

Results saved to: rounding_analysis/
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import cohen_kappa_score

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR  = Path(__file__).parent / "rounding_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

SCORE_COLS = [str(i) for i in range(10)]
MAX_SCORE  = 4

MODEL_CONFIGS = {
    "ds32b_v2_amp": {"exp_prefix": "fewshot_v2_ds32b_amp_freq",
                     "scored_csv": "scored_by_deepseek-r1_32b.csv",
                     "max_n": 15},
    "ds32b_v1_amp": {"exp_prefix": "fewshot_v1_ds32b_amp_freq",
                     "scored_csv": "scored_by_deepseek-r1_32b.csv",
                     "max_n": 15},
    "ds32b_v1_llmfreq": {"exp_prefix": "fewshot_v1_ds32b_llmfreq",
                          "scored_csv": "scored_by_deepseek-r1_32b.csv",
                          "max_n": 10},
}

STYLE = {
    "ds32b_v2_amp":     ("o-",  "tab:blue",   "v2 + amp"),
    "ds32b_v1_amp":     ("s--", "tab:orange", "v1 + amp"),
    "ds32b_v1_llmfreq": ("^-.", "tab:green",  "v1 + llmfreq"),
}

PLOT_MODELS = {"ds32b_v1_amp"}  # set to None to plot all models

# ── helpers ────────────────────────────────────────────────────────────────────
def half_up(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x + 0.5).astype(int), 0, MAX_SCORE)

def kappa(gt_int, pred_int):
    try:
        return float(cohen_kappa_score(gt_int, pred_int, weights="quadratic"))
    except Exception:
        return float("nan")

def load(n, cfg):
    csv = RESULTS_DIR / f"{cfg['exp_prefix']}{n}" / cfg["scored_csv"]
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    pred_raw = np.nanmedian(
        df[SCORE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(), axis=1)
    tan   = df["label_Dr. Tan"].astype(float).to_numpy()
    chien = df["label_Dr. Chien"].astype(float).to_numpy()
    gt_cont = (tan + chien) / 2
    concordant = (tan == chien) & ~np.isnan(gt_cont) & ~np.isnan(pred_raw)
    all_valid  = ~np.isnan(gt_cont) & ~np.isnan(pred_raw)
    return dict(pred_raw=pred_raw, gt_cont=gt_cont,
                concordant=concordant, all_valid=all_valid)

def make_plot(df_list, metric_specs, title, out_path):
    ncols = len(metric_specs)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]
    for ax, (col, mtitle) in zip(axes, metric_specs):
        for model, (ls, color, label) in STYLE.items():
            if PLOT_MODELS and model not in PLOT_MODELS:
                continue
            for df_label, df in df_list:
                sub = df[df.model == model].sort_values("n")
                if col not in sub.columns:
                    continue
                lbl = label if len(df_list) == 1 else f"{label} ({df_label})"
                ax.plot(sub["n"], sub[col], ls, color=color, label=lbl,
                        linewidth=1.8, markersize=7)
        ax.set_title(mtitle, fontsize=10)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(col.upper() if col not in ("within1", "mae") else col)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out_path.name}")

# ── Approach A: concordant sessions only ──────────────────────────────────────
def approach_A():
    print("\n[A] Concordant sessions only (Dr. Tan == Dr. Chien)")
    rows = []
    for model, cfg in MODEL_CONFIGS.items():
        for n in range(1, cfg["max_n"] + 1):
            d = load(n, cfg)
            if d is None:
                continue
            mask     = d["concordant"]
            pred_int = half_up(d["pred_raw"][mask])
            gt_int   = d["gt_cont"][mask].astype(int)

            exact   = float(np.mean(pred_int == gt_int))
            within1 = float(np.mean(np.abs(pred_int - gt_int) <= 1))
            mae     = float(np.mean(np.abs(pred_int - gt_int)))
            k       = kappa(gt_int, pred_int)
            n_samp  = int(mask.sum())

            rows.append(dict(model=model, n=n, n_samples=n_samp,
                             exact=round(exact,4), within1=round(within1,4),
                             mae=round(mae,4), kappa=round(k,4)))
            print(f"  {model} n={n}: n={n_samp}, exact={exact:.3f}, "
                  f"within1={within1:.3f}, MAE={mae:.3f}, kappa={k:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "A_concordant_only.csv", index=False)
    make_plot([("", df)],
              [("exact","Exact Acc"), ("within1","Within-1 Acc"),
               ("mae","MAE"), ("kappa","Kappa")],
              "A: Concordant sessions only (n=22, GT integer, half-up pred)",
              OUTPUT_DIR / "A_concordant_only.png")
    return df

# ── Approach B: all sessions, mixed rounding ──────────────────────────────────
def approach_B():
    print("\n[B] All 53 sessions — ACC/Kappa: half-up GT | MAE: raw pred & continuous GT")
    rows = []
    for model, cfg in MODEL_CONFIGS.items():
        for n in range(1, cfg["max_n"] + 1):
            d = load(n, cfg)
            if d is None:
                continue
            mask     = d["all_valid"]
            pred_int = half_up(d["pred_raw"][mask])   # half-up for ACC / Kappa
            pred_c   = d["pred_raw"][mask]             # raw median for MAE
            gt_int   = half_up(d["gt_cont"][mask])    # half-up for ACC / Kappa
            gt_c     = d["gt_cont"][mask]              # continuous for MAE

            exact   = float(np.mean(pred_int == gt_int))
            within1 = float(np.mean(np.abs(pred_int - gt_int) <= 1))
            mae     = float(np.mean(np.abs(pred_c - gt_c)))
            k       = kappa(gt_int, pred_int)
            n_samp  = int(mask.sum())

            rows.append(dict(model=model, n=n, n_samples=n_samp,
                             exact=round(exact,4), within1=round(within1,4),
                             mae=round(mae,4), kappa=round(k,4)))
            print(f"  {model} n={n}: n={n_samp}, exact={exact:.3f}, "
                  f"within1={within1:.3f}, MAE={mae:.3f}, kappa={k:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "B_mixed_rounding.csv", index=False)
    make_plot([("", df)],
              [("exact","Exact Acc (half-up GT)"), ("within1","Within-1 (half-up GT)"),
               ("mae","MAE (raw pred, continuous GT)"), ("kappa","Kappa (half-up GT)")],
              "B: All 53 sessions — ACC/Kappa: half-up GT  |  MAE: raw pred & continuous GT",
              OUTPUT_DIR / "B_mixed_rounding.png")
    return df

# ── Approach C: all sessions, all half-up ─────────────────────────────────────
def approach_C():
    print("\n[C] All 53 sessions — half-up rounding for everything")
    rows = []
    for model, cfg in MODEL_CONFIGS.items():
        for n in range(1, cfg["max_n"] + 1):
            d = load(n, cfg)
            if d is None:
                continue
            mask     = d["all_valid"]
            pred_int = half_up(d["pred_raw"][mask])
            gt_int   = half_up(d["gt_cont"][mask])

            exact   = float(np.mean(pred_int == gt_int))
            within1 = float(np.mean(np.abs(pred_int - gt_int) <= 1))
            mae     = float(np.mean(np.abs(pred_int - gt_int)))
            k       = kappa(gt_int, pred_int)
            n_samp  = int(mask.sum())

            rows.append(dict(model=model, n=n, n_samples=n_samp,
                             exact=round(exact,4), within1=round(within1,4),
                             mae=round(mae,4), kappa=round(k,4)))
            print(f"  {model} n={n}: n={n_samp}, exact={exact:.3f}, "
                  f"within1={within1:.3f}, MAE={mae:.3f}, kappa={k:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "C_all_halfup.csv", index=False)
    make_plot([("", df)],
              [("exact","Exact Acc"), ("within1","Within-1 Acc"),
               ("mae","MAE"), ("kappa","Kappa")],
              "C: All 53 sessions — half-up rounding for all metrics",
              OUTPUT_DIR / "C_all_halfup.png")
    return df

# ── comparison: MAE across all approaches ─────────────────────────────────────
def comparison_mae(dA, dB, dC):
    print("\n[Summary] MAE comparison across approaches")
    plot_keys = [m for m in MODEL_CONFIGS if PLOT_MODELS is None or m in PLOT_MODELS]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(6.5 * len(plot_keys), 5))
    if len(plot_keys) == 1:
        axes = [axes]
    approach_styles = [
        (dA, "A: concordant (n=22), integer GT",   "o-",  "tab:green"),
        (dB, "B: all (n=53), raw pred + cont. GT", "s--", "tab:orange"),
        (dC, "C: all (n=53), half-up GT",          "^-.", "tab:red"),
    ]
    for ax, model in zip(axes, plot_keys):
        for df, lbl, ls, color in approach_styles:
            sub = df[df.model == model].sort_values("n")
            ax.plot(sub["n"], sub["mae"], ls, color=color, label=lbl,
                    linewidth=1.8, markersize=7)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("MAE")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("MAE comparison across rounding strategies", fontsize=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "comparison_MAE.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out.name}")

# ── comparison: Kappa across A / B / C ────────────────────────────────────────
def comparison_kappa(dA, dB, dC):
    plot_keys = [m for m in MODEL_CONFIGS if PLOT_MODELS is None or m in PLOT_MODELS]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(6.5 * len(plot_keys), 5))
    if len(plot_keys) == 1:
        axes = [axes]
    approach_styles = [
        (dA, "A: concordant (n=22)", "o-",  "tab:green"),
        (dB, "B: all (n=53), mixed", "s--", "tab:orange"),
        (dC, "C: all (n=53), half-up","^-.", "tab:red"),
    ]
    for ax, model in zip(axes, plot_keys):
        for df, lbl, ls, color in approach_styles:
            sub = df[df.model == model].sort_values("n")
            ax.plot(sub["n"], sub["kappa"], ls, color=color, label=lbl,
                    linewidth=1.8, markersize=7)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("Quadratic Weighted Kappa")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Kappa comparison across rounding strategies", fontsize=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "comparison_Kappa.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out.name}")

# ── rater comparison: banker's vs half-up on the 22 X.5 GT sessions ───────────
def rater_comparison():
    """For each experiment, on the 22 sessions where |Tan-Chien|=1 (GT=X.5):
       Compare MAE(pred, Tan), MAE(pred, Chien),
               MAE(pred, bankers_GT), MAE(pred, halfup_GT).
       The rounding closer to the midpoint of MAE(pred,Tan) and MAE(pred,Chien)
       is the more neutral choice.
    """
    print("\n[Rater comparison] 22 sessions with |Tan-Chien|=1 (GT=X.5 only)")

    def bankers(x):
        return np.clip(np.round(x).astype(int), 0, MAX_SCORE)

    rows = []
    for model, cfg in MODEL_CONFIGS.items():
        for n in range(1, cfg["max_n"] + 1):
            d = load(n, cfg)
            if d is None:
                continue
            df_csv = pd.read_csv(
                RESULTS_DIR / f"{cfg['exp_prefix']}{n}" / cfg["scored_csv"])
            tan   = df_csv["label_Dr. Tan"].astype(float).to_numpy()
            chien = df_csv["label_Dr. Chien"].astype(float).to_numpy()
            pred_raw = d["pred_raw"]

            # only sessions where |Tan - Chien| == 1  (GT = X.5)
            mask_half = (np.abs(tan - chien) == 1) & ~np.isnan(pred_raw)
            if mask_half.sum() == 0:
                continue

            pred_h = half_up(pred_raw[mask_half])
            tan_h  = tan[mask_half].astype(int)
            chien_h = chien[mask_half].astype(int)
            gt_cont = (tan_h + chien_h) / 2

            mae_tan     = float(np.mean(np.abs(pred_h - tan_h)))
            mae_chien   = float(np.mean(np.abs(pred_h - chien_h)))
            mae_bankers = float(np.mean(np.abs(pred_h - bankers(gt_cont))))
            mae_halfup  = float(np.mean(np.abs(pred_h - half_up(gt_cont))))
            mid         = (mae_tan + mae_chien) / 2  # ideal midpoint

            rows.append(dict(model=model, n=n,
                             n_half=int(mask_half.sum()),
                             mae_vs_Tan=round(mae_tan,4),
                             mae_vs_Chien=round(mae_chien,4),
                             midpoint=round(mid,4),
                             mae_bankers_GT=round(mae_bankers,4),
                             mae_halfup_GT=round(mae_halfup,4),
                             closer=("bankers" if abs(mae_bankers-mid) <= abs(mae_halfup-mid)
                                     else "half_up")))
            print(f"  {model} n={n}: Tan={mae_tan:.3f}, Chien={mae_chien:.3f}, "
                  f"mid={mid:.3f} | bankers={mae_bankers:.3f}, halfup={mae_halfup:.3f} "
                  f"→ closer: {rows[-1]['closer']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "rater_comparison.csv", index=False)

    # plot
    plot_keys = [m for m in MODEL_CONFIGS if PLOT_MODELS is None or m in PLOT_MODELS]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(6.5 * len(plot_keys), 5))
    if len(plot_keys) == 1:
        axes = [axes]
    for ax, model in zip(axes, plot_keys):
        sub = df[df.model == model].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"], sub["mae_vs_Tan"],     "o-",  color="tab:blue",
                label="MAE vs Dr. Tan", linewidth=1.8, markersize=7)
        ax.plot(sub["n"], sub["mae_vs_Chien"],   "o-",  color="tab:orange",
                label="MAE vs Dr. Chien", linewidth=1.8, markersize=7)
        ax.plot(sub["n"], sub["midpoint"],       "k--",
                label="midpoint (Tan+Chien)/2", linewidth=1.2)
        ax.plot(sub["n"], sub["mae_bankers_GT"], "s--", color="tab:green",
                label="MAE vs banker's GT", linewidth=1.8, markersize=7)
        ax.plot(sub["n"], sub["mae_halfup_GT"],  "^-.", color="tab:red",
                label="MAE vs half-up GT", linewidth=1.8, markersize=7)
        ax.set_title(f"{model}\n(22 sessions with |Tan-Chien|=1)", fontsize=10)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("MAE")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Rounding comparison vs individual rater scores\n"
                 "(midpoint = ideal neutral GT; closer line = more neutral rounding)",
                 fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "rater_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out.name}")
    return df

# ── combined summary CSV ───────────────────────────────────────────────────────
def save_summary(dA, dB, dC):
    summary = pd.concat([
        dA.assign(approach="A_concordant"),
        dB.assign(approach="B_mixed"),
        dC.assign(approach="C_all_halfup"),
    ], ignore_index=True)
    out = OUTPUT_DIR / "all_approaches_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved summary: {out.name}")

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dA = approach_A()
    dB = approach_B()
    dC = approach_C()
    comparison_mae(dA, dB, dC)
    comparison_kappa(dA, dB, dC)
    save_summary(dA, dB, dC)
    rater_comparison()
    print(f"\nAll results saved to: {OUTPUT_DIR}")
