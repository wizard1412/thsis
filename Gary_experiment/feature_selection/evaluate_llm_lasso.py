"""
Evaluate LLM-Lasso Selected Features: MLP LOOCV + LLM Scoring
==============================================================
Reads feature_selection_results.json produced by llm_lasso_select.py and
evaluates the selected feature subsets using the same MLP LOOCV and LLM
scoring pipeline as feature_selection_comparison_with_llm.py.

Results are saved to fs_results_llm/ so they can be merged with
comparison_results.csv from traditional FS methods.

Usage:
  python evaluate_llm_lasso.py
  python evaluate_llm_lasso.py --skip-mlp
  python evaluate_llm_lasso.py --skip-llm
  python evaluate_llm_lasso.py --llm-evals 3
"""

import os, json, warnings, time, re, requests, argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FEAT_CSV = "features_dataset_v2.csv"

# LLM-Lasso results (no_rag subdirectory, matching USE_RAG=False in config)
LLM_LASSO_JSON = Path(
    "experiments_llm_lasso_amplitude/results/no_rag/feature_selection_results.json"
)

OUT_DIR  = "fs_results_llm"
PNG_DIR  = os.path.join(OUT_DIR, "png")
PDF_DIR  = os.path.join(OUT_DIR, "pdf")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

CACHE_FILE = os.path.join(OUT_DIR, "llm_score_cache.json")
OUT_CSV    = os.path.join(OUT_DIR, "llm_lasso_eval_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# LLM Config  (must match feature_selection_comparison_with_llm.py)
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434/api/generate"
LLM_MODEL           = "deepseek-r1:32b"
LLM_EVAL_TIMES      = 3
REQUEST_TIMEOUT     = 300
NUM_PREDICT         = 4096
DELAY_BETWEEN_CALLS = 2
MAX_RETRIES         = 3

FEWSHOT_EXAMPLES = [
    (0, "553661824_20200629"),
    (1, "445817794_20200720"),
    (2, "879885247_20200608"),
    (3, "546503158_20200310"),
]

MAX_SCORE = 4
N_BOOT    = 2000
RNG       = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template  (identical to feature_selection_comparison_with_llm.py)
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_BASE = """The data provided below consists of extracted features derived from a subject performing the MDS-UPDRS Item 3.4 Finger Tapping test. The subject was instructed to tap the index finger on the thumb 10 times as quickly and as largely as possible.
The objective is to score the subject according to MDS-UPDRS item 3.4: FINGER TAPPING based on the selected 10 seconds of continuous finger tapping. The instruction and scoring criteria of MDS-UPDRS item 3.4: FINGER TAPPING are as follows:
"Instructions to the examiner: Each hand is tested separately. Demonstrate the task, but do not continue to perform the task while the patient is being tested. Instruct the patient to tap the index finger on the thumb 10 times as quickly AND as big as possible. Rate each side separately, evaluating speed, amplitude, hesitations, halts, and decrementing amplitude.
0: Normal: No problems.
1: Slight: Any of the following: a) the regular rhythm is broken with one or two interruptions or hesitations of the tapping movement; b) slight slowing; c) the amplitude decrements near the end of the 10 taps.
2: Mild: Any of the following: a) 3 to 5 interruptions during tapping; b) mild slowing; c) the amplitude decrements midway in the 10-tap sequence.
3: Moderate: Any of the following: a) more than 5 interruptions during tapping or at least one longer arrest (freeze) in ongoing movement; b) moderate slowing; c) the amplitude decrements starting after the 1st tap.
4: Severe: Cannot or can only barely perform the task because of slowing, interruptions, or decrements."
Below are the extracted features.
To determine the subject's score, analyze the provided features thoroughly and give a detailed motivation for the assigned score based on the MDS-UPDRS finger tapping instruction. Print the score using only one number on the last line of your answer, after the detailed motivation.

===================================
FEW-SHOT EXAMPLES
===================================
Below are examples for each score.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = {"hand", "filename", "Rating1", "Rating2", "Rating", "tap_amplitudes"}
    feat_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    X_full = df[feat_cols].values.astype(float)
    y      = df["Rating"].values.astype(float)
    print(f"  FEAT_CSV : {FEAT_CSV}")
    print(f"  Samples  : {len(df)}, Features: {len(feat_cols)}")
    return X_full, y, feat_cols, df


def load_llm_lasso_selections():
    """
    Read feature_selection_results.json and return a dict of
    {method_label: [feature_name, ...]} covering:
      - LLM-Lasso (best η)
      - Standard LASSO (η=0, from LLM-Lasso pipeline)
    Plus one entry per η in the eta search if n_selected > 0.
    """
    if not LLM_LASSO_JSON.exists():
        raise FileNotFoundError(
            f"LLM-Lasso results not found: {LLM_LASSO_JSON}\n"
            "Run llm_lasso_select.py first."
        )

    with open(LLM_LASSO_JSON, encoding="utf-8") as f:
        data = json.load(f)

    selections = {}

    # Best η selection
    best_eta   = data["best_eta"]
    best_feats = data["llm_lasso_selected_features"]
    if best_feats:
        selections[f"LLM-Lasso (η={best_eta})"] = best_feats

    # Standard LASSO (η=0) from same pipeline
    std_feats = data.get("standard_lasso_selected", [])
    if std_feats:
        selections["Std-LASSO (LLM-Lasso)"] = std_feats

    print(f"  LLM_LASSO_JSON : {LLM_LASSO_JSON}")
    print(f"  Best η={best_eta}  →  {len(best_feats)} features: {best_feats}")
    print(f"  Std LASSO (η=0) →  {len(std_feats)} features")
    return selections

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def round_score(arr):
    return np.clip(np.floor(np.asarray(arr, dtype=float) + 0.5).astype(int), 0, MAX_SCORE)


def compute_metrics(y_true, y_pred):
    yt = round_score(np.array(y_true, dtype=float))
    yp = round_score(np.array(y_pred, dtype=float))
    mask = ~(np.isnan(yt.astype(float)) | np.isnan(yp.astype(float)))
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 5:
        return dict(MAE=np.nan, Spearman=np.nan, Kappa=np.nan, AAcc=np.nan)
    mae   = mean_absolute_error(yt, yp)
    sp    = spearmanr(yt, yp).statistic
    try:
        kappa = cohen_kappa_score(yt.astype(int), yp.astype(int), weights="quadratic")
    except Exception:
        kappa = np.nan
    aacc  = (np.abs(yt - yp) <= 1).mean()
    return dict(MAE=mae, Spearman=sp, Kappa=kappa, AAcc=aacc)


def bootstrap_ci(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    records = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        records.append(compute_metrics(y_true[idx], y_pred[idx]))
    bdf = pd.DataFrame(records)
    result = {}
    for col in bdf.columns:
        result[f"{col}_CI_lo"] = bdf[col].quantile(0.025)
        result[f"{col}_CI_hi"] = bdf[col].quantile(0.975)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MLP LOOCV
# ─────────────────────────────────────────────────────────────────────────────
def mlp_loocv(X_sub, y):
    loo  = LeaveOneOut()
    mlp  = MLPRegressor(
        hidden_layer_sizes=(8,), activation="relu",
        solver="adam", alpha=1.0, max_iter=600, random_state=42,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   mlp),
    ])
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(X_sub):
        try:
            pipe.fit(X_sub[train_idx], y[train_idx])
            preds[test_idx[0]] = pipe.predict(X_sub[test_idx])[0]
        except Exception:
            pass
    return preds

# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt building
# ─────────────────────────────────────────────────────────────────────────────
def _format_feature_block(row, feat_names, indent="  "):
    lines = []
    for feat in feat_names:
        if feat in row.index:
            val = row[feat]
            if isinstance(val, float) and not np.isnan(val):
                val = round(float(val), 4)
            lines.append(f"{indent}{feat}: {val}")
    return "\n".join(lines)


def build_fewshot_text(df, feat_names):
    parts = []
    for score, prefix in FEWSHOT_EXAMPLES:
        mask = df["filename"].str.contains(prefix, na=False) if "filename" in df.columns else pd.Series([False]*len(df))
        if not mask.any():
            continue
        row = df[mask].iloc[0]
        parts.append(f"\n───────────────\nEXAMPLE FOR SCORE {score}\n───────────────")
        parts.append("STATISTICAL FEATURES:")
        parts.append(_format_feature_block(row, feat_names))
        parts.append(f"\nThis example was rated as SCORE {score}.\n")
    return "\n".join(parts)


def build_patient_prompt(row, feat_names, fewshot_text):
    subject_block  = "\n===================================\nSUBJECT TO SCORE\n==================================="
    subject_block += "\nSTATISTICAL FEATURES:\n"
    subject_block += _format_feature_block(row, feat_names)
    return PROMPT_BASE + fewshot_text + subject_block

# ─────────────────────────────────────────────────────────────────────────────
# LLM API
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  [Ollama] Cannot connect: {e}")
        return False


def call_ollama(prompt, retry=0):
    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.7, "num_predict": NUM_PREDICT},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["response"]
    except (Exception, KeyboardInterrupt) as e:
        if retry < MAX_RETRIES - 1:
            time.sleep(5)
            return call_ollama(prompt, retry + 1)
        print(f"    [LLM ERROR] {type(e).__name__}: {e}")
        return None


def parse_score(response):
    if response is None:
        return None
    for line in reversed(response.strip().split("\n")):
        m = re.search(r"\b([0-4])\b", line)
        if m:
            return int(m.group(1))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# LLM evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _aggregate(raw):
    n = len(raw)
    preds_median = np.full(n, np.nan)
    for i, scores in enumerate(raw):
        valid = [s for s in scores if s is not None]
        if valid:
            preds_median[i] = float(np.median(valid))
    return preds_median


def llm_evaluate_method(fs_name, df, feat_names, y, n_evals, cache):
    cache_key = f"{fs_name}|{','.join(feat_names)}|n{n_evals}|v2"

    if cache_key in cache:
        print(f"    [cache hit]")
        return _aggregate(cache[cache_key])

    fewshot_text = build_fewshot_text(df, feat_names)

    fewshot_mask = set()
    for _, prefix in FEWSHOT_EXAMPLES:
        matches = df[df["filename"].str.contains(prefix, na=False)].index.tolist() \
            if "filename" in df.columns else []
        fewshot_mask.update(matches)

    all_trial_scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        flag = " [few-shot]" if i in fewshot_mask else ""
        patient_scores = []
        for e in range(n_evals):
            prompt = build_patient_prompt(row, feat_names, fewshot_text)
            resp   = call_ollama(prompt)
            s      = parse_score(resp)
            patient_scores.append(s)
            time.sleep(DELAY_BETWEEN_CALLS)

        valid      = [s for s in patient_scores if s is not None]
        median_val = float(np.median(valid)) if valid else None
        all_trial_scores.append(patient_scores)
        print(f"    [{i+1:2d}/{len(df)}]{flag} scores={patient_scores} → median={median_val}")

    cache[cache_key] = all_trial_scores
    _save_cache(cache)
    return _aggregate(all_trial_scores)

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(X_full, y, feat_cols, df, selections, run_mlp, run_llm, n_evals):
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    cache    = _load_cache() if run_llm else {}
    rows     = []

    for fs_name, feats in selections.items():
        valid_feats = [f for f in feats if f in feat_idx]
        missing     = [f for f in feats if f not in feat_idx]
        if missing:
            print(f"  [warn] {fs_name}: {len(missing)} features not in dataset: {missing}")
        if not valid_feats:
            print(f"  [skip] {fs_name}: no valid features")
            continue

        idx   = [feat_idx[f] for f in valid_feats]
        X_sub = X_full[:, idx]
        print(f"\n  [{fs_name}]  n_features={len(idx)}")

        row = {
            "FS_Method":  fs_name,
            "N_Features": len(idx),
            "Category":   "llm_lasso",
        }

        # ── MLP LOOCV ──────────────────────────────────────────────────────
        if run_mlp:
            print(f"    MLP LOOCV ...", end=" ", flush=True)
            preds_mlp = mlp_loocv(X_sub, y)
            m  = compute_metrics(y, preds_mlp)
            ci = bootstrap_ci(y, preds_mlp)
            for k, v in m.items():
                row[f"MLP_{k}"] = v
            for k, v in ci.items():
                row[f"MLP_{k}"] = v
            print(f"MAE={m['MAE']:.3f}  Kappa={m['Kappa']:.3f}")

        # ── LLM scoring ───────────────────────────────────────────────────
        if run_llm:
            print(f"    LLM scoring ({n_evals}x per patient) ...")
            preds_llm = llm_evaluate_method(fs_name, df, valid_feats, y, n_evals, cache)
            m  = compute_metrics(y, preds_llm)
            ci = bootstrap_ci(y, preds_llm)
            for k, v in m.items():
                row[f"LLM_median_{k}"] = v
            for k, v in ci.items():
                row[f"LLM_median_{k}"] = v
            coverage = (~np.isnan(preds_llm)).mean()
            row["LLM_Coverage"] = coverage
            print(f"    LLM median: MAE={m['MAE']:.3f}  Kappa={m['Kappa']:.3f}  Coverage={coverage:.0%}")

        rows.append(row)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-Lasso features: MLP + LLM")
    parser.add_argument("--llm-evals", type=int, default=LLM_EVAL_TIMES,
                        help=f"LLM evaluations per patient (default: {LLM_EVAL_TIMES})")
    parser.add_argument("--skip-mlp",  action="store_true", help="Skip MLP LOOCV")
    parser.add_argument("--skip-llm",  action="store_true", help="Skip LLM scoring")
    args = parser.parse_args()

    print("=" * 60)
    print("Evaluate LLM-Lasso Selected Features: MLP + LLM")
    print("=" * 60)

    print("\n[1] Loading data ...")
    X_full, y, feat_cols, df = load_data()

    print("\n[2] Loading LLM-Lasso selections ...")
    selections = load_llm_lasso_selections()

    run_mlp = not args.skip_mlp
    run_llm = not args.skip_llm

    if run_llm and not check_ollama():
        print("  WARNING: Ollama not reachable. Skipping LLM evaluation.")
        run_llm = False

    print(f"\n[3] Running evaluation ...")
    print(f"  Evaluators: {'MLP(8) LOOCV  ' if run_mlp else ''}"
          f"{'LLM=' + LLM_MODEL + ' (' + str(args.llm_evals) + 'x)' if run_llm else ''}")

    results = run_evaluation(X_full, y, feat_cols, df, selections,
                             run_mlp=run_mlp, run_llm=run_llm, n_evals=args.llm_evals)

    results.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")

    # Summary table
    print("\n" + "=" * 90)
    has_mlp = "MLP_MAE" in results.columns
    has_llm = "LLM_median_MAE" in results.columns
    header = f"{'Method':<35} {'N':>4}"
    if has_mlp:
        header += f"  {'MLP_MAE':>8} {'MLP_Kappa':>9}"
    if has_llm:
        header += f"  {'LLM_MAE':>8} {'LLM_Kappa':>9}"
    print(header)
    print("=" * 90)
    for _, row in results.iterrows():
        line = f"  {row['FS_Method']:<33} {row['N_Features']:>4}"
        if has_mlp:
            line += f"  {row.get('MLP_MAE', np.nan):>8.3f} {row.get('MLP_Kappa', np.nan):>9.3f}"
        if has_llm:
            line += f"  {row.get('LLM_median_MAE', np.nan):>8.3f} {row.get('LLM_median_Kappa', np.nan):>9.3f}"
        print(line)
    print("=" * 90)


if __name__ == "__main__":
    main()
