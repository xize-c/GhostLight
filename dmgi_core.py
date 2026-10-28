"""
DMGI core: differentiable single-pixel imaging simulation + adaptive pattern-generation network.
Shared by the training script dmgi.py and the demo script demo.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- physics simulation (differentiable)
def measure(P, x, noise):
    """P: (B,K,H,W) pattern intensities in [0,1], x: (B,1,H,W) sample -> bucket readings (B,K).
    Relative Gaussian noise."""
    b = (P * x).flatten(2).sum(-1)
    if noise > 0:
        b = b + noise * b.detach().abs().mean(1, keepdim=True) * torch.randn_like(b)
    return b


def ls_recon(P, b, lam=1e-3):
    """Regularized least-squares reconstruction x = A^T (A A^T + lam*I)^-1 b.
    P: (B,M,H,W), b: (B,M) -> (B,1,H,W)"""
    B, M, H, W = P.shape
    A = P.reshape(B, M, H * W)
    G = A @ A.transpose(1, 2)
    scale = G.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
    G = G + lam * scale * torch.eye(M, device=G.device, dtype=G.dtype)
    y = torch.linalg.solve(G, b.unsqueeze(-1))
    return (A.transpose(1, 2) @ y).view(B, 1, H, W)


def light_dose(P):
    """Cumulative light on the sample, in units of one full-field exposure (a random pattern is ~0.5).
    P: (B,M,H,W) -> (B,)"""
    return P.mean(dim=(2, 3)).sum(1)


# ----------------------------------------------------------------- Small U-Net (Fig. 5 of the paper)
def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """With base=64 the channels are 64/128/256 and the decoder concatenations are 384->128 and 192->64,
    matching the paper."""

    def __init__(self, in_ch=1, out_ch=8, base=16, n_rounds=8):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.bott = conv_block(c2, c3)
        self.t_emb = nn.Embedding(n_rounds, c3)        # round embedding (like the diffusion time step t)
        self.up2 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.dec2 = conv_block(c3 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.dec1 = conv_block(c2 + c1, c1)
        self.out = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bott(F.max_pool2d(e2, 2)) + self.t_emb(t)[:, :, None, None]
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))             # pattern intensities in [0,1] (DMD-displayable)


class DMGI(nn.Module):
    """
    Round 0: K starting patterns (learned from data if learn_p0=True, shared by all samples;
             otherwise fixed random patterns)
    Round t: U-Net(current reconstruction, t) -> K new patterns for THIS sample only
    After every round, least squares is re-solved with all patterns and readings so far,
    and the new reconstruction is fed back to the network.
    """

    def __init__(self, k=8, rounds=8, base=16, noise=0.01, size=28, learn_p0=True, seed=0):
        super().__init__()
        assert size % 4 == 0, "image size must be divisible by 4 (two U-Net downsamplings)"
        self.k, self.rounds, self.noise, self.size = k, rounds, noise, size
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(1, k, size, size, generator=g).clamp(0.02, 0.98)
        logits = torch.log(u / (1 - u))                # sigmoid(logits) = uniform random patterns
        if learn_p0:
            self.p0_logits = nn.Parameter(logits)
        else:
            self.register_buffer("p0_logits", logits)
        self.net = SmallUNet(1, k, base, rounds)

    def forward(self, x):
        B = x.shape[0]
        Ps = [torch.sigmoid(self.p0_logits).expand(B, -1, -1, -1)]
        bs = [measure(Ps[0], x, self.noise)]
        rec = ls_recon(Ps[0], bs[0])
        recs = [rec]
        for t in range(1, self.rounds):
            tt = torch.full((B,), t, dtype=torch.long, device=x.device)
            P = self.net(rec.clamp(0, 1), tt)
            Ps.append(P)
            bs.append(measure(P, x, self.noise))
            rec = ls_recon(torch.cat(Ps, 1), torch.cat(bs, 1))
            recs.append(rec)
        return recs, torch.cat(Ps, 1)


# ----------------------------------------------------------------- losses
def ssim_torch(x, y, k=7, C1=0.01 ** 2, C2=0.03 ** 2):
    pad = k // 2
    mu_x = F.avg_pool2d(x, k, 1, pad, count_include_pad=False)
    mu_y = F.avg_pool2d(y, k, 1, pad, count_include_pad=False)
    sxx = F.avg_pool2d(x * x, k, 1, pad, count_include_pad=False) - mu_x ** 2
    syy = F.avg_pool2d(y * y, k, 1, pad, count_include_pad=False) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, k, 1, pad, count_include_pad=False) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


def rec_loss(rec, x, kind):
    if kind == "mse":
        return F.mse_loss(rec, x)
    return (1 - ssim_torch(rec, x)) / 2 + 0.5 * F.mse_loss(rec, x)   # DSSIM (+ a little MSE to anchor brightness)
