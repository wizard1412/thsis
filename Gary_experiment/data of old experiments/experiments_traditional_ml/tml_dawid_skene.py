"""
Dawid-Skene EM Algorithm for multi-rater label aggregation.

Reference: Dawid, A. P. & Skene, A. M. (1979).
"Maximum likelihood estimation of observer error-rates using the EM algorithm."

This is the traditional ML analog of neural RCE (Rater Confusion Estimation):
both estimate per-rater confusion matrices and infer true labels.
The difference is Dawid-Skene uses EM instead of backpropagation.
"""

import numpy as np
from tml_config import SCORE_NUM_CLASSES, DS_MAX_ITER, DS_TOL, DS_SMOOTHING


def dawid_skene_em(ratings, num_classes=None, max_iter=None, tol=None,
                   smoothing=None, init_labels=None):
    """
    Run Dawid-Skene EM on multi-rater annotations.

    Parameters:
        ratings:      (N, R) rater annotations, -1 = missing
        num_classes:  number of classes (default: SCORE_NUM_CLASSES=4)
        max_iter:     max EM iterations (default: 100)
        tol:          convergence tolerance (default: 1e-4)
        smoothing:    Laplace smoothing (default: 1e-6)
        init_labels:  (N,) initial label estimates, or None for majority vote

    Returns dict with:
        estimated_labels:    (N,) MAP label estimates
        label_posteriors:    (N, C) posterior P(true=c | data)
        confusion_matrices:  (R, C, C) per-rater, A[r, c_obs, c_true]
        class_priors:        (C,) estimated class prevalence
        n_iterations:        number of EM iterations
        converged:           bool
    """
    if num_classes is None:
        num_classes = SCORE_NUM_CLASSES
    if max_iter is None:
        max_iter = DS_MAX_ITER
    if tol is None:
        tol = DS_TOL
    if smoothing is None:
        smoothing = DS_SMOOTHING

    N, R = ratings.shape
    C = num_classes

    # === Initialization ===
    T = np.zeros((N, C))
    if init_labels is not None:
        for i in range(N):
            T[i, int(init_labels[i])] = 1.0
    else:
        for i in range(N):
            valid = ratings[i][ratings[i] >= 0]
            if len(valid) == 0:
                T[i] = 1.0 / C
            else:
                counts = np.bincount(valid.astype(int), minlength=C)
                T[i, counts.argmax()] = 1.0

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # === M-Step ===
        # Class priors
        pi = T.sum(axis=0) / N
        pi = np.maximum(pi, smoothing)
        pi = pi / pi.sum()

        # Per-rater confusion matrices
        A = np.zeros((R, C, C))
        for r in range(R):
            valid_mask = ratings[:, r] >= 0
            for c_true in range(C):
                denom = T[valid_mask, c_true].sum() + smoothing * C
                for c_obs in range(C):
                    obs_mask = (ratings[:, r] == c_obs) & valid_mask
                    numer = T[obs_mask, c_true].sum() + smoothing
                    A[r, c_obs, c_true] = numer / denom

        # === E-Step ===
        T_new = np.zeros((N, C))
        for i in range(N):
            for c in range(C):
                log_prob = np.log(pi[c] + 1e-30)
                for r in range(R):
                    if ratings[i, r] >= 0:
                        obs = int(ratings[i, r])
                        log_prob += np.log(A[r, obs, c] + 1e-30)
                T_new[i, c] = log_prob
            # Log-sum-exp normalization
            max_log = T_new[i].max()
            T_new[i] = np.exp(T_new[i] - max_log)
            T_new[i] = T_new[i] / (T_new[i].sum() + 1e-30)

        # Check convergence
        delta = np.abs(T_new - T).max()
        T = T_new
        n_iter = iteration + 1

        if delta < tol:
            converged = True
            break

    return {
        "estimated_labels": T.argmax(axis=1),
        "label_posteriors": T,
        "confusion_matrices": A,
        "class_priors": pi,
        "n_iterations": n_iter,
        "converged": converged,
    }