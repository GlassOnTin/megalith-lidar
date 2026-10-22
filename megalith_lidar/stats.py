from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


class PlattCalibrator:
    """
    1D Platt / Logistic scaling calibrator fitted strictly on validation logits:
      P(Y=1 | z) = sigmoid(a * z + b)
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y_true, dtype=np.int32).reshape(-1)
        clf = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
        clf.fit(z, y)
        self.a = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        exponent = np.clip(-(self.a * z + self.b), -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(exponent))).astype(np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {"a": round(self.a, 6), "b": round(self.b, 6)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PlattCalibrator":
        cal = cls()
        cal.a = float(d.get("a", 1.0))
        cal.b = float(d.get("b", 0.0))
        return cal


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` probability intervals."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if np.any(mask):
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece += (np.sum(mask) / float(n)) * abs(acc - conf)
    return float(ece)


def compute_prior_adjusted_ppv(tpr: float, fpr: float, prior: float) -> float:
    """
    Bayes' theorem Positive Predictive Value (Precision) at true spatial prevalence `prior` (pi):
      PPV(pi) = (TPR * pi) / (TPR * pi + FPR * (1 - pi))
    Avoids the base-rate fallacy of reporting raw precision from a class-balanced dataset.
    """
    num = float(tpr) * float(prior)
    den = num + float(fpr) * (1.0 - float(prior))
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def select_validation_thresholds(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    target_fprs: Sequence[float] = (0.01, 0.05, 0.10),
) -> Dict[str, Dict[str, float]]:
    """
    Select operating thresholds strictly from the validation set distribution:
      - Exact empirical quantile thresholds on validation negatives for target FPRs (e.g. 1%, 5%, 10%)
      - Maximum Youden's J statistic (TPR - FPR) which is prevalence-invariant!
    """
    p = np.asarray(val_probs, dtype=np.float64)
    y = np.asarray(val_y, dtype=np.int32)
    neg_p = p[y == 0]
    pos_p = p[y == 1]

    out: Dict[str, Dict[str, float]] = {}
    for target_fpr in target_fprs:
        # Threshold is the (1 - target_fpr) quantile of negative validation scores
        th = float(np.quantile(neg_p, max(0.0, min(1.0, 1.0 - target_fpr))))
        emp_fpr = float(np.mean(neg_p >= th))
        emp_tpr = float(np.mean(pos_p >= th))
        out[f"fpr_{target_fpr:.2f}"] = {
            "threshold": round(th, 5),
            "val_tpr": round(emp_tpr, 4),
            "val_fpr": round(emp_fpr, 4),
        }

    # Prevalence-invariant Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    best_j = -1.0
    best_th = 0.5
    for th in candidates:
        tpr = float(np.mean(pos_p >= th))
        fpr = float(np.mean(neg_p >= th))
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_th = float(th)
    out["max_youden_j"] = {
        "threshold": round(best_th, 5),
        "val_tpr": round(float(np.mean(pos_p >= best_th)), 4),
        "val_fpr": round(float(np.mean(neg_p >= best_th)), 4),
        "val_youden_j": round(best_j, 4),
    }
    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute point estimate and 95% stratified bootstrap confidence interval (2.5th, 97.5th percentiles).
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    point = float(metric_fn(y, s))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point, point, point

    boot_vals = []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        try:
            boot_vals.append(float(metric_fn(y[b_idx], s[b_idx])))
        except Exception:
            continue
    if not boot_vals:
        return point, point, point
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


def evaluate_predictions_statistically(
    y_true: np.ndarray,
    raw_logits: np.ndarray,
    calibrator: PlattCalibrator,
    val_thresholds: Dict[str, Dict[str, float]],
    n_boot: int = 1000,
) -> Dict:
    """
    Compute comprehensive, base-rate-invariant and prior-adjusted statistical evaluation metrics
    with 95% stratified bootstrap confidence intervals.
    """
    y = np.asarray(y_true, dtype=np.int32)
    probs = calibrator.predict_proba(raw_logits)
    uncal_probs = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(raw_logits, dtype=np.float64), -60.0, 60.0)))

    roc_est, roc_lo, roc_hi = bootstrap_metric_ci(y, probs, lambda yt, yp: float(roc_auc_score(yt, yp)), n_boot=n_boot)
    pr_est, pr_lo, pr_hi = bootstrap_metric_ci(y, probs, lambda yt, yp: float(average_precision_score(yt, yp)), n_boot=n_boot)

    operating_points = {}
    for name, th_info in val_thresholds.items():
        th = th_info["threshold"]

        def _tpr_fn(yt: np.ndarray, yp: np.ndarray, t: float = th) -> float:
            return float(np.mean(yp[yt == 1] >= t))

        def _fpr_fn(yt: np.ndarray, yp: np.ndarray, t: float = th) -> float:
            return float(np.mean(yp[yt == 0] >= t))

        tpr, tpr_lo, tpr_hi = bootstrap_metric_ci(y, probs, _tpr_fn, n_boot=n_boot)
        fpr, fpr_lo, fpr_hi = bootstrap_metric_ci(y, probs, _fpr_fn, n_boot=n_boot)
        lr_plus = round(tpr / max(fpr, 1e-4), 2)
        operating_points[name] = {
            "val_selected_threshold": th,
            "tpr_recall": tpr,
            "tpr_ci95": [tpr_lo, tpr_hi],
            "fpr_false_positive_rate": fpr,
            "fpr_ci95": [fpr_lo, fpr_hi],
            "youden_j": round(tpr - fpr, 4),
            "likelihood_ratio_pos": lr_plus,
            "ppv_at_prior_0.01": round(compute_prior_adjusted_ppv(tpr, fpr, 0.01), 4),
            "ppv_at_prior_0.001": round(compute_prior_adjusted_ppv(tpr, fpr, 0.001), 4),
        }

    return {
        "n_samples": int(len(y)),
        "n_positives": int(np.sum(y == 1)),
        "n_negatives": int(np.sum(y == 0)),
        "roc_auc": roc_est,
        "roc_auc_ci95": [roc_lo, roc_hi],
        "pr_auc_sample_prevalence": pr_est,
        "pr_auc_ci95": [pr_lo, pr_hi],
        "brier_score_uncalibrated": round(float(brier_score_loss(y, uncal_probs)), 4),
        "brier_score_calibrated": round(float(brier_score_loss(y, probs)), 4),
        "ece_uncalibrated": round(compute_ece(y, uncal_probs), 4),
        "ece_calibrated": round(compute_ece(y, probs), 4),
        "operating_points": operating_points,
    }
