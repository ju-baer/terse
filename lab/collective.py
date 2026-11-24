"""Collective / ensemble statistics (pure numpy) + early-trajectory forecasting."""
import math
import numpy as np
from collections import Counter


def icc_binary(C):
    """ANOVA intraclass correlation of correctness across samples within a problem. C: (P, N) in {0,1}.
    High ICC = outcomes are all-or-nothing per problem = samples carry little independent evidence."""
    C = np.asarray(C, float)
    P, N = C.shape
    s = C.mean(1)
    msb = N * s.var(ddof=1) if P > 1 else 0.0
    msw = float((N * s * (1 - s)).sum() / (P * (N - 1)))
    d = msb + (N - 1) * msw
    return float(np.clip((msb - msw) / d, -1, 1)) if d > 0 else 0.0


def k_eff(N, rho):
    return N / (1 + (N - 1) * rho)


def pass_at_k(C, k):
    C = np.asarray(C, int)
    P, N = C.shape
    c = C.sum(1)
    return float(np.mean([1 - math.comb(N - ci, k) / math.comb(N, k) if N - ci >= k else 1.0 for ci in c]))


def maj_curve(answers, truth, ns, R=100, seed=0):
    """answers: P lists of N parsed answers (None = unparsed). Plurality vote over random n-subsets."""
    rng = np.random.default_rng(seed)
    P = len(answers)
    out = []
    for n in ns:
        acc = 0.0
        for p in range(P):
            a = answers[p]
            hit = 0
            for _ in range(R):
                votes = Counter(a[j] for j in rng.choice(len(a), n, replace=False) if a[j] is not None)
                if votes:
                    top = max(votes.values())
                    w = [k for k, v in votes.items() if v == top]
                    hit += w[rng.integers(len(w))] == truth[p]
            acc += hit / R
        out.append(acc / P)
    return out


def wrong_mode_concentration(answers, truth):
    """Mean share of wrong answers taken by the single most common wrong answer (problems with >=2 wrong)."""
    shares = []
    for a, t in zip(answers, truth):
        w = [x for x in a if x is not None and x != t]
        if len(w) >= 2:
            shares.append(Counter(w).most_common(1)[0][1] / len(w))
    return float(np.mean(shares)) if shares else float("nan")


def fit_exp(t, y, tau_grid=None):
    """y ~ a + b*exp(-t/tau), grid over tau, closed-form (a, b). Returns (a, b, tau)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    best = None
    for tau in (tau_grid if tau_grid is not None else np.geomspace(max(t.max() / 50, 1e-3), t.max() * 5, 60)):
        X = np.stack([np.ones_like(t), np.exp(-t / tau)], 1)
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = float(((X @ coef - y) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, coef[0], coef[1], tau)
    return best[1:]


def forecast_final(t, y, frac=0.4):
    """Fit on the first `frac` of the trajectory, predict the last point. Compare with persistence baseline."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    k = max(3, int(round(frac * len(t))))
    a, b, tau = fit_exp(t[:k], y[:k])
    pred = a + b * math.exp(-t[-1] / tau)
    return dict(pred=float(pred), truth=float(y[-1]), persistence=float(y[k - 1]),
                err_fit=abs(pred - y[-1]), err_persist=abs(y[k - 1] - y[-1]))
