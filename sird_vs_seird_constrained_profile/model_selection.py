"""Information criteria for comparing calibrated compartmental models.

Convention of Huaman, Plesovskaya & Leonenko, "Matching model complexity with data
detail: influenza propagation modeling as a case study", 2022 IEEE SIBIRCON,
doi:10.1109/SIBIRCON56155.2022.10017084, Sec. II.B:

* least-squares calibration with Gaussian errors, variance estimated as RSS/n (Eq. 4);
* AIC = n ln(RSS) + 2k once the terms that do not depend on the model are dropped (Eq. 6);
* AICc = AIC + (2k^2 + 2k) / (n - k - 1), recommended when k > n/40 (Eq. 7);
* k = number of free epidemiological parameters, with no +1 for the noise variance:
  SIRD k = 3 (beta, gamma, mu), SEIRD k = 4 (beta, sigma, gamma, mu).

n is the number of residuals that enter RSS, i.e. time points x compartments. Summing
several compartments assumes one error variance shared by all of them, which is a
strong assumption when they differ in scale by orders of magnitude; a single-compartment
RSS (I only) or one variance per compartment (``compute_aic_separate_variances``) are
the natural cross-checks.

When the data's S column is derived as N - I - R - D (as in covid-19_Kouprianov.csv)
and the model conserves N, the S residual equals minus the sum of the I, R and D
residuals: it adds n and RSS without adding information.

Only differences between models fitted to the same data with the same n carry meaning.
Absolute values depend on the convention: the Viruses draft writes n ln(RSS/n) + 2k,
which differs from Eq. 6 by the constant n ln n.
"""
import math

import numpy as np

DEFAULT_COMPARTMENTS = ("S", "I", "R", "D")


def _as_array(values):
    """numpy float vector from a list, array or torch tensor."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def compute_rss(y_true, y_pred, compartments=DEFAULT_COMPARTMENTS):
    """Unweighted residual sum of squares over the listed compartments.

    ``y_true`` and ``y_pred`` map compartment names to series of equal length.
    Raises ValueError on a length mismatch or on non-finite values.
    """
    total = 0.0
    for name in compartments:
        observed, predicted = _as_array(y_true[name]), _as_array(y_pred[name])
        if observed.shape != predicted.shape:
            raise ValueError(f"{name}: {observed.size} observations vs {predicted.size} predictions")
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(predicted))):
            raise ValueError(f"{name}: non-finite values")
        total += float(np.sum((predicted - observed) ** 2))
    return total


def count_residuals(y_true, compartments=DEFAULT_COMPARTMENTS):
    """n for AIC: number of observations summed in ``compute_rss``."""
    return int(sum(_as_array(y_true[name]).size for name in compartments))


def compute_aic(rss, n, k):
    """AIC = n ln(RSS) + 2k (Eq. 6 of the SIBIRCON 2022 convention, see module docstring).

    rss : residual sum of squares, > 0
    n   : number of residuals in ``rss``
    k   : number of free epidemiological parameters (SIRD 3, SEIRD 4)
    """
    if rss <= 0:
        raise ValueError("RSS must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    return n * math.log(rss) + 2 * k


def compute_aic_separate_variances(rss_by_compartment, n_by_compartment, k):
    """AIC with one error variance per compartment: sum_c n_c ln(RSS_c) + 2k.

    Same derivation as Eq. 6 with a Gaussian likelihood per compartment. The variance
    parameters are shared by every model compared on the same compartments, so, as in
    the source convention, they are not added to k.
    """
    rss, n = list(rss_by_compartment), list(n_by_compartment)
    if len(rss) != len(n) or not rss:
        raise ValueError("need one n per compartment RSS")
    if any(r <= 0 for r in rss) or any(m <= 0 for m in n):
        raise ValueError("RSS and n must be positive")
    return sum(m * math.log(r) for r, m in zip(rss, n)) + 2 * k


def compute_aicc(aic, n, k):
    """AICc = AIC + (2k^2 + 2k) / (n - k - 1) (Eq. 7); requires n > k + 1."""
    if n - k - 1 <= 0:
        raise ValueError("AICc needs n > k + 1")
    return aic + (2 * k * k + 2 * k) / (n - k - 1)


def aicc_recommended(n, k):
    """The small-sample rule of the same source: use AICc when k > n/40."""
    return k > n / 40
