"""
DMGI training + evaluation: AI-designed adaptive illumination patterns that reduce the light dose
of single-pixel (ghost) imaging.

Use case: low-light fluorescence / microscopy single-pixel imaging. More exposures mean more light on
the sample (phototoxicity, photobleaching). DMGI looks at the current blurry reconstruction and designs
the next patterns round by round, reaching the same image quality with far fewer exposures.

Usage (results are saved to runs/<dataset>_<loss>/):
  python dmgi.py --epochs 1 --n_train 256 --n_test 50     # smoke test, 1-2 minutes
  python dmgi.py --loss ssim                               # main run: BloodMNIST 28x28
  python dmgi.py --dataset pathmnist                       # other dataset
  python demo.py --run runs/bloodmnist_ssim                # demo animation and figures
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from baselines import default_Ms, eval_recs, patterns_needed, random_baselines
from data_utils import load_dataset
from dmgi_core import DMGI, light_dose, rec_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bloodmnist", help="bloodmnist / pathmnist / mnist / other MedMNIST 2D")
    ap.add_argument("--size", type=int, default=28, help="image size 28 or 64 (64 is much slower)")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="invert microscopy images to fluorescence style (dark background, bright cells); auto = invert microscopy datasets")
    ap.add_argument("--k", type=int, default=8, help="patterns per round")
    ap.add_argument("--rounds", type=int, default=8, help="number of rounds T; total patterns = k*rounds")
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels (64 in the paper)")
    ap.add_argument("--fixed_p0", action="store_true", help="use fixed random patterns in round 0 (default: round-0 patterns are learned)")
    ap.add_argument("--loss", choices=["mse", "ssim"], default="mse")
    ap.add_argument("--n_train", type=int, default=4800)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--noise", type=float, default=0.01, help="relative bucket-detector noise")
    ap.add_argument("--pattern_hz", type=float, default=1000, help="DMD pattern rate (Hz), used to estimate acquisition time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory, default runs/<dataset>_<loss>")
    a = ap.parse_args()

    out = a.out or os.path.join("runs", f"{a.dataset}_{a.loss}")
    os.makedirs(out, exist_ok=True)
    P = lambda f: os.path.join(out, f)
    invert = {"auto": None, "yes": True, "no": False}[a.invert]

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, name = load_dataset(a.dataset, "train", a.size, invert)
    Xte, _ = load_dataset(a.dataset, "test", a.size, invert)
    rng = np.random.default_rng(a.seed)
    Xtr = Xtr[rng.permutation(len(Xtr))[:a.n_train]]
    Xte = Xte[np.random.default_rng(1).permutation(len(Xte))[:a.n_test]]
    S = a.size; N = S * S
    M_total = a.k * a.rounds
    print(f"Data: {name}  train {len(Xtr)} / test {len(Xte)}  device: {dev}")
    print(f"Exposures per sample M = {a.k} x {a.rounds} = {M_total} ({N} pixels, sampling ratio {M_total / N:.1%})")

    model = DMGI(a.k, a.rounds, a.base, a.noise, S, learn_p0=not a.fixed_p0, seed=a.seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    xtr = torch.from_numpy(Xtr).unsqueeze(1)
    w = torch.arange(1, a.rounds + 1, dtype=torch.float32); w = (w / w.sum()).tolist()  # later rounds weigh more

    # ---------------------------------------------------------------- training
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), a.bs):
            x = xtr[perm[i:i + a.bs]].to(dev)
            recs, _ = model(x)
            loss = sum(wt * rec_loss(r, x, a.loss) for wt, r in zip(w, recs))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(x)
        sched.step()
        hist.append(tot / len(xtr))
        el = time.time() - t0
        print(f"epoch {ep + 1:3d}/{a.epochs}  loss {hist[-1]:.5f}  elapsed {el:.0f}s  remaining ~{el / (ep + 1) * (a.epochs - ep - 1):.0f}s")
    cfg = {k_: v for k_, v in vars(a).items()}
    cfg["invert_resolved"] = invert
    torch.save({"state_dict": model.state_dict(), "args": cfg}, P("model.pt"))

    # ---------------------------------------------------------------- test: DMGI
    model.eval()
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    with torch.no_grad():
        xte = torch.from_numpy(Xte).unsqueeze(1).to(dev)
        model(xte[:8]); sync()
        t1 = time.perf_counter()
        recs, P_all = model(xte)
        sync()
        dm_ms = (time.perf_counter() - t1) * 1000 / len(Xte)
        dm_dose_round = [float(light_dose(P_all[:, :a.k * (t + 1)]).mean()) for t in range(a.rounds)]
    X_flat = Xte.reshape(len(Xte), -1)
    dm_round = [eval_recs(r.cpu().numpy().reshape(len(Xte), -1), X_flat, "clip") for r in recs]
    dm_ssim, dm_psnr = dm_round[-1]
    dm_dose = dm_dose_round[-1]
    dm_img = recs[-1].clamp(0, 1).cpu().numpy()[:, 0]

    # ---------------------------------------------------------------- test: random-pattern baselines
    Ms = default_Ms(N, extra=[a.k * (t + 1) for t in range(a.rounds)])
    print("Computing random-pattern baselines ...")
    base = random_baselines(X_flat, Ms, a.noise)
    acq = lambda m: m / a.pattern_hz * 1000

    # ---------------------------------------------------------------- result tables
    print(f"\n===== Same number of exposures M = {M_total} (test-set mean) =====")
    print(f"{'Method':<10}{'SSIM':>8}{'PSNR':>8}{'LightDose':>10}{'Recon ms':>11}{'Acq. ms':>11}")
    for k_ in ["GI", "DGI", "Rand-LS"]:
        s, p, _, ms, dose = base[k_][M_total]
        print(f"{k_:<10}{s.mean():8.4f}{p.mean():8.2f}{dose:10.1f}{ms:11.4f}{acq(M_total):11.1f}")
    print(f"{'DMGI':<10}{dm_ssim.mean():8.4f}{dm_psnr.mean():8.2f}{dm_dose:10.1f}{dm_ms:11.4f}{acq(M_total):11.1f}")
    print("  Light dose unit: one full-field exposure. DMGI recon time includes U-Net pattern generation in every round.")

    target = float(dm_ssim.mean())
    print(f"\n===== To match DMGI image quality (SSIM={target:.3f}) each method needs =====")
    print(f"{'Method':<10}{'Exposures':>10}{'LightDose':>10}{'Dose vs DMGI':>14}{'Acq. ms':>12}")
    need, summary = {}, {}
    for k_ in ["GI", "DGI", "Rand-LS"]:
        n = patterns_needed(Ms, [base[k_][m][0].mean() for m in Ms], target)
        need[k_] = n
        if n is None:
            print(f"{k_:<10}{'>' + str(Ms[-1]):>10}{'>' + format(0.5 * Ms[-1], '.0f'):>10}"
                  f"{'>' + format(0.5 * Ms[-1] / dm_dose, '.0f') + 'x':>14}{'-':>12}  (not reached in tested range)")
            continue
        summary[k_] = {"patterns": n, "dose": 0.5 * n, "dose_ratio": 0.5 * n / dm_dose,
                       "pattern_ratio": n / M_total}
        print(f"{k_:<10}{n:10.0f}{0.5 * n:10.1f}{0.5 * n / dm_dose:13.1f}x{acq(n):12.1f}")
    print(f"{'DMGI':<10}{M_total:10d}{dm_dose:10.1f}{1:13.1f}x{acq(M_total):12.1f}")

    # ---------------------------------------------------------------- fig 1: SSIM vs exposures / light dose
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, xkey in zip(axs, ["patterns", "dose"]):
        for k_, c in [("GI", "tab:blue"), ("DGI", "tab:orange"), ("Rand-LS", "tab:green")]:
            xs = Ms if xkey == "patterns" else [base[k_][m][4] for m in Ms]
            ax.plot(xs, [base[k_][m][0].mean() for m in Ms], "o-", ms=3, c=c, label=f"{k_} (random patterns)")
        xs = [a.k * (t + 1) for t in range(a.rounds)] if xkey == "patterns" else dm_dose_round
        ax.plot(xs, [s.mean() for s, _ in dm_round], "s-", c="tab:red", label="DMGI (AI-designed patterns)")
        ax.axhline(target, ls="--", c="tab:red", lw=.8)
        ax.set_xscale("log"); ax.set_ylabel("mean SSIM"); ax.grid(alpha=.3, which="both")
        ax.set_xlabel("number of exposures" if xkey == "patterns" else "light dose on sample (full-field exposures)")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(P("ssim_vs_exposures.png"), dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- fig 2: reconstructions
    m_match = next((m for m in Ms if need["Rand-LS"] is not None and m >= need["Rand-LS"]), Ms[-1])
    rows = [("Ground truth", Xte), (f"DGI  {M_total} exp.", base["DGI"][M_total][2]),
            (f"Random+LS  {M_total} exp.", base["Rand-LS"][M_total][2]),
            (f"DMGI  {M_total} exp.", dm_img), (f"Random+LS  {m_match} exp.", base["Rand-LS"][m_match][2])]
    n_show = 6
    fig, axs = plt.subplots(len(rows), n_show, figsize=(n_show * 1.6, len(rows) * 1.7))
    for r, (lab, imgs) in enumerate(rows):
        for c in range(n_show):
            axs[r, c].imshow(imgs[c].reshape(S, S), cmap="gray", vmin=0, vmax=1)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(lab, fontsize=8)
    fig.tight_layout(); fig.savefig(P("reconstructions.png"), dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- fig 3: patterns and reconstructions per round
    P0 = P_all[0].cpu().numpy()
    fig, axs = plt.subplots(2, a.rounds, figsize=(a.rounds * 1.5, 3.3))
    for t in range(a.rounds):
        axs[0, t].imshow(P0[t * a.k], cmap="gray", vmin=0, vmax=1); axs[0, t].set_title(f"round {t}", fontsize=8)
        axs[1, t].imshow(recs[t][0, 0].clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axs[1, t].set_title(f"{a.k * (t + 1)} exp.", fontsize=8)
        for r in range(2):
            axs[r, t].axis("off")
    fig.tight_layout(); fig.savefig(P("rounds.png"), dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- fig 4: training loss
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(range(1, len(hist) + 1), hist, "o-"); ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({a.loss})")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(P("loss.png"), dpi=140); plt.close(fig)

    json.dump({"dataset": name, "M": M_total, "dmgi_ssim": target, "dmgi_psnr": float(dm_psnr.mean()),
               "dmgi_light_dose": dm_dose, "dmgi_ssim_per_round": [float(s.mean()) for s, _ in dm_round],
               "dmgi_dose_per_round": dm_dose_round,
               "baseline_ssim_at_M": {k_: float(base[k_][M_total][0].mean()) for k_ in base},
               "random_patterns_needed": need, "matched_quality": summary,
               "recon_ms": {**{k_: float(base[k_][M_total][3]) for k_ in base}, "DMGI": dm_ms},
               "pattern_hz": a.pattern_hz},
              open(P("results.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}/ : ssim_vs_exposures.png  reconstructions.png  rounds.png  loss.png  "
          "results.json  model.pt")
    print(f"Next, create the demo:  python demo.py --run {out}")


if __name__ == "__main__":
    main()
