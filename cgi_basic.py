"""
Classic computational ghost imaging (CGI) simulation: second-order correlation GI + differential GI (DGI).
Follows the accumulation loop of the original paper (random pattern -> bucket reading -> accumulate ->
one reconstruction every report_every patterns). Only needs numpy / matplotlib / scikit-image; runs in seconds.

Usage:  python cgi_basic.py --n_patterns 20000 --idx 0
        python cgi_basic.py --dataset mnist
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_dataset, minmax, ssim_psnr


def cgi_simulate(obj, n_patterns, report_every=1000, noise=0.0, seed=0):
    """obj: (H,W) object transmittance in [0,1]. Returns {N: (GI image, DGI image)}."""
    rng = np.random.default_rng(seed)
    H, W = obj.shape
    SP = np.zeros((H, W))      # sum B*I   -> <BI>
    CCDz = np.zeros((H, W))    # sum I     -> <I>
    SPS = np.zeros((H, W))     # sum S*I   -> <SI>   (for DGI; S = total pattern intensity)
    Ibz = 0.0                  # sum B     -> <B>
    Sz = 0.0                   # sum S     -> <S>
    snaps = {}
    for i in range(1, n_patterns + 1):
        D = rng.random((H, W))              # random pattern I_n loaded on the SLM/DMD
        Ib = float((obj * D).sum())         # bucket detector reading B_n
        if noise > 0:
            Ib += noise * rng.standard_normal() * Ib
        S = float(D.sum())                  # total reference intensity (computed, no camera needed)
        Ibz += Ib; Sz += S
        CCDz += D; SP += Ib * D; SPS += S * D
        if i % report_every == 0 or i == n_patterns:
            mB, mS, mI = Ibz / i, Sz / i, CCDz / i
            G_gi = SP / i - mB * mI                          # GI:  <BI> - <B><I>
            G_dgi = SP / i - (mB / mS) * (SPS / i)           # DGI: <BI> - <B>/<S> <SI>
            snaps[i] = (minmax(G_gi), minmax(G_dgi))
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bloodmnist", help="bloodmnist / pathmnist / mnist ...")
    ap.add_argument("--n_patterns", type=int, default=20000)
    ap.add_argument("--report_every", type=int, default=1000)
    ap.add_argument("--idx", type=int, default=0, help="index of the test image")
    ap.add_argument("--noise", type=float, default=0.0, help="relative bucket-detector noise")
    ap.add_argument("--out", default="cgi_basic_result.png")
    a = ap.parse_args()

    imgs, name = load_dataset(a.dataset, "test")
    obj = imgs[a.idx]
    print(f"Dataset: {name}, object size {obj.shape}, {a.n_patterns} illuminations")
    snaps = cgi_simulate(obj, a.n_patterns, a.report_every, a.noise)

    Ns = sorted(snaps)
    s_gi = [ssim_psnr(snaps[n][0], obj)[0] for n in Ns]
    s_dgi = [ssim_psnr(snaps[n][1], obj)[0] for n in Ns]
    for n in [Ns[0], Ns[len(Ns) // 4], Ns[len(Ns) // 2], Ns[-1]]:
        k = Ns.index(n)
        print(f"N={n:6d}  GI SSIM={s_gi[k]:.3f}   DGI SSIM={s_dgi[k]:.3f}")

    show = [Ns[0], Ns[len(Ns) // 4], Ns[len(Ns) // 2], Ns[-1]]
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(2, 6, 1); ax.imshow(obj, cmap="gray"); ax.set_title("Object"); ax.axis("off")
    for j, n in enumerate(show):
        for r, lab in enumerate(["GI", "DGI"]):
            ax = fig.add_subplot(2, 6, 2 + j + 6 * r)
            ax.imshow(snaps[n][r], cmap="gray"); ax.set_title(f"{lab}  N={n}"); ax.axis("off")
    ax = fig.add_subplot(1, 6, 6)
    ax.plot(Ns, s_gi, label="GI"); ax.plot(Ns, s_dgi, label="DGI")
    ax.set_xlabel("N (illuminations)"); ax.set_ylabel("SSIM"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print("Figure saved:", a.out)


if __name__ == "__main__":
    main()
