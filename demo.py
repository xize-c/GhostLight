"""
Demo script: loads a trained model and creates the animation and comparison figures for presenting.

  python demo.py --run runs/bloodmnist_ssim
  python demo.py --run runs/bloodmnist_ssim --idx 3 7 12 25    # choose which test images to show

Outputs (in the --run directory):
  demo.gif          round-by-round animation: random vs AI-designed patterns at the same number of exposures;
                    the last frame shows how many exposures random patterns need to catch up
  demo_summary.png  the whole result in one figure, for slides and Devpost
  demo_patterns.png the patterns the AI generates for one sample, round by round
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from baselines import ls_np, measure_np
from data_utils import ssim_psnr

RED, GRAY = "#c0392b", "#555555"


def _mean_ssim(recs, gts):
    return float(np.mean([ssim_psnr(np.clip(r, 0, 1).astype(np.float64), g.astype(np.float64))[0]
                          for r, g in zip(recs, gts)]))


def make_gif(gt, rand_frames, dm_frames, Ms, dm_doses, m_need, rand_need, path, fps=1.2):
    """gt: (n,S,S); rand_frames/dm_frames: list[(n,S,S)], one frame per round; Ms: cumulative exposures per round;
    m_need/rand_need: exposures random patterns need to match DMGI, and that reconstruction (n,S,S)."""
    n = len(gt)
    T = len(Ms)
    fig, axs = plt.subplots(3, n, figsize=(1.9 * n + 1.6, 6.4))
    axs = np.asarray(axs).reshape(3, n)
    ims = []
    for r in range(3):
        row = []
        for c in range(n):
            im = axs[r, c].imshow(gt[c], cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            row.append(im)
        ims.append(row)
    labels = [axs[r, 0].set_ylabel("", fontsize=10) for r in range(3)]
    axs[0, 0].set_ylabel("Sample\n(ground truth)", fontsize=10)
    title = fig.suptitle("", fontsize=12, fontweight="bold")
    note = fig.text(0.5, 0.015, "", ha="center", fontsize=9, color=GRAY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    # frame sequence: one frame per round, hold the last round, then show "random needs N exposures"
    seq = list(range(T)) + [T - 1] * 2 + [T] * 4

    def update(f):
        if f < T:
            M = Ms[f]
            for c in range(n):
                ims[1][c].set_data(rand_frames[f][c]); ims[2][c].set_data(dm_frames[f][c])
            s_r, s_d = _mean_ssim(rand_frames[f], gt), _mean_ssim(dm_frames[f], gt)
            labels[1].set_text(f"Random patterns\n{M} exposures\nSSIM {s_r:.2f}")
            labels[2].set_text(f"AI-designed (DMGI)\n{M} exposures\nSSIM {s_d:.2f}")
            labels[1].set_color(GRAY); labels[2].set_color(RED)
            title.set_text(f"Same number of exposures: {M}")
            note.set_text(f"light dose on sample  —  random: {0.5 * M:.0f}   DMGI: {dm_doses[f]:.0f}   "
                          "(units: full-field exposures)")
        else:
            for c in range(n):
                ims[1][c].set_data(rand_need[c]); ims[2][c].set_data(dm_frames[-1][c])
            s_r, s_d = _mean_ssim(rand_need, gt), _mean_ssim(dm_frames[-1], gt)
            labels[1].set_text(f"Random patterns\n{m_need} exposures\nSSIM {s_r:.2f}")
            labels[2].set_text(f"AI-designed (DMGI)\n{Ms[-1]} exposures\nSSIM {s_d:.2f}")
            title.set_text(f"Same image quality: random needs {m_need} exposures, DMGI needs {Ms[-1]} "
                           f"(≈{m_need / Ms[-1]:.0f}× fewer)")
            note.set_text(f"light dose on sample  —  random: {0.5 * m_need:.0f}   DMGI: {dm_doses[-1]:.0f}   "
                          f"(≈{0.5 * m_need / dm_doses[-1]:.0f}× less light)")
        return []

    anim = FuncAnimation(fig, update, frames=seq, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_summary(gt, dm_last, rand_same, rand_need, M, m_need, dm_dose, path):
    n = len(gt)
    cols = [("Sample", gt, "black"),
            (f"Random patterns\n{M} exposures", rand_same, GRAY),
            (f"AI-designed (DMGI)\n{M} exposures", dm_last, RED),
            (f"Random patterns\n{m_need} exposures", rand_need, GRAY)]
    fig, axs = plt.subplots(n, 4, figsize=(8.2, 2.0 * n + 1.3))
    axs = np.asarray(axs).reshape(n, 4)
    for c, (lab, imgs, col) in enumerate(cols):
        for r in range(n):
            axs[r, c].imshow(np.clip(imgs[r], 0, 1), cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        sub = "" if c == 0 else f"\nSSIM {_mean_ssim(imgs, gt):.2f}"
        axs[0, c].set_title(lab + sub, fontsize=10, color=col)
    fig.suptitle(f"Same image quality with ≈{m_need / M:.0f}× fewer exposures "
                 f"and ≈{0.5 * m_need / dm_dose:.0f}× less light on the sample",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def make_patterns(P_one, recs_one, gt_one, k, path):
    T = len(recs_one)
    fig, axs = plt.subplots(2, T + 1, figsize=(1.55 * (T + 1), 3.5))
    axs[0, 0].imshow(gt_one, cmap="gray", vmin=0, vmax=1); axs[0, 0].set_title("sample", fontsize=8)
    axs[1, 0].axis("off")
    for t in range(T):
        axs[0, t + 1].imshow(P_one[t * k], cmap="gray", vmin=0, vmax=1)
        axs[0, t + 1].set_title("learned start" if t == 0 else f"round {t}", fontsize=8)
        axs[1, t + 1].imshow(np.clip(recs_one[t], 0, 1), cmap="gray", vmin=0, vmax=1)
        axs[1, t + 1].set_title(f"{k * (t + 1)} exp.", fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 1].set_ylabel("pattern", fontsize=8); axs[1, 1].set_ylabel("image", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    import torch
    from data_utils import load_dataset
    from dmgi_core import DMGI, light_dose

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bloodmnist_ssim", help="output directory of dmgi.py")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="indices of test images to show (default: 4 random ones)")
    ap.add_argument("--fps", type=float, default=1.2)
    a = ap.parse_args()

    ck = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")
    cfg = ck["args"]
    res = json.load(open(os.path.join(a.run, "results.json")))
    Xte, _ = load_dataset(cfg["dataset"], "test", cfg["size"], cfg["invert_resolved"])
    idx = a.idx if a.idx else list(np.random.default_rng(1).permutation(len(Xte))[:4])
    gt = Xte[idx]
    S = cfg["size"]; k = cfg["k"]; T = cfg["rounds"]

    model = DMGI(k, T, cfg["base"], cfg["noise"], S, learn_p0=not cfg["fixed_p0"], seed=cfg["seed"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        recs, P_all = model(torch.from_numpy(gt).unsqueeze(1))
        dm_doses = [float(light_dose(P_all[:, :k * (t + 1)]).mean()) for t in range(T)]
    dm_frames = [r.clamp(0, 1).numpy()[:, 0] for r in recs]
    Ms = [k * (t + 1) for t in range(T)]

    # random patterns + the same least-squares reconstructor; uses the first M patterns of one random sequence
    need = res["random_patterns_needed"].get("Rand-LS")
    m_need = int(round(need)) if need else 32 * S * S
    rng = np.random.default_rng(7)
    Pr = rng.random((max(m_need, Ms[-1]), S * S), dtype=np.float32)
    Xf = gt.reshape(len(gt), -1)
    rand_frames = []
    for M in Ms + [m_need]:
        b = measure_np(Pr[:M], Xf, cfg["noise"], rng)
        rand_frames.append(np.clip(ls_np(Pr[:M], b), 0, 1).reshape(-1, S, S))
    rand_need = rand_frames.pop()

    make_gif(gt, rand_frames, dm_frames, Ms, dm_doses, m_need, rand_need, os.path.join(a.run, "demo.gif"), a.fps)
    make_summary(gt, dm_frames[-1], rand_frames[-1], rand_need, Ms[-1], m_need, dm_doses[-1],
                 os.path.join(a.run, "demo_summary.png"))
    make_patterns(P_all[0].numpy(), [f[0] for f in dm_frames], gt[0], k, os.path.join(a.run, "demo_patterns.png"))
    print(f"Created: {a.run}/demo.gif  demo_summary.png  demo_patterns.png")
    if not need:
        print("Note: random patterns did not match DMGI within the tested range; the figure uses the largest tested count, so the true number is higher.")


if __name__ == "__main__":
    main()
