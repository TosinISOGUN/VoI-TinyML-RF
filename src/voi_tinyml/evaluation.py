"""Classification metrics, bootstrap confidence intervals and exact McNemar tests (Sec. 4.7)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from statsmodels.stats.contingency_tables import mcnemar

from .config import EXP, SEED


def classification_report(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
        "per_class": [{"class": k, "precision": p[k], "recall": r[k], "f1": f[k], "support": int(s[k])}
                      for k in range(3)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_ci(y_true, y_pred, n: int = EXP.n_bootstrap, seed: int = SEED, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc, f1 = np.empty(n), np.empty(n)
    for i in range(n):
        b = rng.integers(0, len(y_true), len(y_true))
        acc[i] = accuracy_score(y_true[b], y_pred[b])
        f1[i] = f1_score(y_true[b], y_pred[b], average="macro", labels=[0, 1, 2], zero_division=0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"accuracy": tuple(np.percentile(acc, [lo, hi])), "macro_f1": tuple(np.percentile(f1, [lo, hi]))}


def exact_mcnemar(y_true, pred_a, pred_b) -> float:
    """Exact (binomial) McNemar test on paired correct/incorrect outcomes."""
    a_ok, b_ok = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    table = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
             [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
    return float(mcnemar(table, exact=True).pvalue)
