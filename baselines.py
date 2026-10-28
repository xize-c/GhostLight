"""
Random-pattern baselines (numpy, batched): the same set of random patterns illuminates every test sample.
  GI      : second-order correlation  <BI> - <B><I>
  DGI     : differential ghost imaging <BI> - <B>/<S> <SI>
  Rand-LS : regularized least-squares reconstruction (the SAME reconstructor DMGI uses, so any gap
            between Rand-LS and DMGI comes from the pattern design alone)
"""
import time

import numpy as np

from data_utils import minmax, ssim_psnr


def measure_np(P, X, noise, rng):
    """P: (M,N) patterns, X: (n,N) samples -> bucket readings (n,M). Relative Gaussian noise."""
    b = X @ P.T
    if noise > 0:
        b = b + noise * np.abs(b).mean(1, keepdims=True) * rng.standard_normal(b.shape).astype(b.dtype)
    return b


def gi_np(P, b):
    return (b - b.mean(1, keepdims=True)) @ (P - P.mean(0)) / P.shape[0]


def dgi_np(P, b):
    S = P.sum(1)                                     # (M,)
    BI = b @ P / len(P)                              # <BI>
    SI = S @ P / len(P)                              # <SI>
    return BI - (b.mean(1, keepdims=True) / S.mean()) * SI


def ls_np(P, b, lam=1e-3):
    """x = argmin ||Px-b||^2 + reg; dual form when M<N, primal form when M>=N."""
    P = P.astype(np.float64); b = b.astype(np.float64)
    M, N = P.shape
    if M < N:
        G = P @ P.T
        G = G + lam * np.trace(G) / M * np.eye(M)
        return np.linalg.solve(G, b.T).T @ P
    G = P.T @ P
    G = G + lam * np.trace(G) / N * np.eye(N)
    return np.linalg.solve(G, P.T @ b.T).T


def eval_recs(recs, X, mode):
    """recs/X: (n,N). GI/DGI amplitudes are arbitrary, so they are min-max normalized; LS has physical scale and is clipped to [0,1]."""
    s, p = [], []
    side = int(round(np.sqrt(X.shape[1])))
    for r, x in zip(recs, X):
        r = minmax(r) if mode == "minmax" else np.clip(r, 0, 1)
        a, b_ = ssim_psnr(r.reshape(side, side).astype(np.float64), x.reshape(side, side).astype(np.float64))
        s.append(a); p.append(b_)
    return np.array(s), np.array(p)


def default_Ms(n_pix, extra=()):
    """Pattern counts to sweep: small M values plus 1-32x the pixel count, capped by memory."""
    m_max = min(32 * n_pix, int(2.5e7 // n_pix))
    grid = [128, 256, 512, 1024, 2048] + [n_pix * 2 ** i for i in range(6)]
    return sorted(set([m for m in grid if m <= m_max] + list(extra)))


def random_baselines(X, Ms, noise=0.01, seed=123):
    """X: (n,N). Returns {method: {M: (ssim, psnr, reconstructions (n,N), recon time ms/image, light dose)}}
    Timing covers only computing the image from bucket readings, not the simulated measurement or scoring.
    Light dose = sum of the mean intensity of every pattern (unit: one full-field exposure)."""
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    out = {"GI": {}, "DGI": {}, "Rand-LS": {}}
    for M in Ms:
        P = rng.random((M, X.shape[1]), dtype=np.float32)
        dose = float(P.mean() * M)
        b = measure_np(P, X, noise, rng)
        for name, fn, mode in [("GI", gi_np, "minmax"), ("DGI", dgi_np, "minmax"), ("Rand-LS", ls_np, "clip")]:
            fn(P[:8], b[:, :8])                      # warm-up
            t0 = time.perf_counter()
            r = fn(P, b)
            ms = (time.perf_counter() - t0) * 1000 / len(X)
            s, p = eval_recs(r, X, mode)
            img = np.array([minmax(v) for v in r]) if mode == "minmax" else np.clip(r, 0, 1)
            out[name][M] = (s, p, img.astype(np.float32), ms, dose)
    return out


def patterns_needed(curve_M, curve_ssim, target):
    """Interpolate on a log scale to find the fewest patterns that reach the target SSIM; None if never reached."""
    M = np.asarray(curve_M, float); s = np.asarray(curve_ssim, float)
    for i in range(len(M)):
        if s[i] >= target:
            if i == 0:
                return float(M[0])
            f = (target - s[i - 1]) / (s[i] - s[i - 1] + 1e-12)
            return float(np.exp(np.log(M[i - 1]) + f * (np.log(M[i]) - np.log(M[i - 1]))))
    return None
