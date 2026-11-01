#!/usr/bin/env python3
"""Self-supervised control: SimCLR training of a ResNet-18 without labels, for the superclass-structure analysis.

SimCLR (Chen et al. 2020, ICML, arXiv:2002.05709) on the 50,000 CIFAR-100 training images; the labels are never read
by the training loop. Encoder = the project's ResNet-18 (transfer_lib.build_model("ResNet-18"): torchvision resnet18,
ImageNet stem on 32 x 32 inputs, so layer4 is 1 x 1 and the avgpool vector is 512-d), with the classifier replaced by
the identity. Inputs are normalised with the project's CIFAR-100 statistics (transfer_lib.MEAN/STD, cifar_batch).

Recipe (standard CIFAR SimCLR; sources):
  views        two views per image: RandomResizedCrop(32, scale (0.2, 1), ratio (3/4, 4/3)), horizontal flip p 0.5,
               ColorJitter(0.4, 0.4, 0.4, 0.1) with p 0.8 (random op order), RandomGrayscale p 0.2, no blur.
               Colour strength 0.5 and no blur = SimCLR's CIFAR-10 setting (Chen et al. 2020, App. B.9) and the
               Lightly CIFAR-10 benchmark (SimCLRTransform(input_size=32, cj_strength=0.5, gaussian_blur=0.0));
               crop scale (0.2, 1) as in the MoCo-v2/SimSiam CIFAR recipes (Chen & He 2021, CVPR).
  head         Linear(512, 512) - ReLU - Linear(512, 128), no BN (SimCLR g(.); lightly SimCLRProjectionHead(512, 512, 128)).
  loss         NT-Xent, temperature 0.5 (Chen et al. 2020 App. B.9; lightly NTXentLoss default).
  optimiser    SGD, momentum 0.9, weight decay 5e-4 (all parameters), lr = 0.06 x batch / 256 (= 0.06 at batch 256),
               cosine annealing stepped per epoch over --total-epochs (lr_e = lr0 * (1 + cos(pi e / E)) / 2, e = 0..E-1),
               no warm-up; batch 256, last partial batch dropped (195 updates per epoch). Lightly CIFAR-10 benchmark
               (lightly-ai/lightly, benchmarks/cifar10_benchmark.py, v1.4: SGD lr 6e-2 x batch/128, momentum 0.9,
               wd 5e-4, CosineAnnealingLR(max_epochs)); the base-lr reference batch here is 256
               (--lr-ref 128 reproduces the Lightly scaling).
  epochs       100 (--total-epochs fixes the schedule; --epochs is where this invocation stops, so a short run is
               exactly the first epochs of the full run and the run can be resumed to 100).
Augmentation is computed batched on the CPU (faster than on a shared MPS device; the views are then
moved to the device): crop + flip by an affine grid (bilinear, continuous crop box; the
torchvision crop rounds the box to whole pixels), colour ops with torchvision's tensor formulas (brightness, contrast
around the grey mean, saturation against grey, hue in HSV via torchvision's _rgb2hsv/_hsv2rgb), per-sample factors.
Randomness: torch/numpy/random seeded with --seed before the model is built; the data order of epoch e is
RandomState(1000 * seed + e).permutation(50000) and the augmentation draws of epoch e come from a CPU generator seeded
1000003 * seed + e, so a resumed run draws the same views (MPS kernels are not bit-deterministic).

Checkpoints at epochs 0 (initialisation), 1, 2, 3, 5, 10, 20, ..., 100: <out>/seed<S>/epoch_<EEE>/
  encoder.pt            encoder state_dict (torchvision resnet18 keys, no fc)
  projection_head.pt    head state_dict
  pooled_test.npz       X (10000, 512) float16 avgpool activations of the CIFAR-100 test set in eval mode, canonical
                        class-major order (transfer_lib.cifar_test); y_fine, y_super (CIFAR-100 coarse labels)
<out>/seed<S>/train_state.pt (latest epoch; resume), train_log.jsonl (per epoch: loss, positive-pair accuracy, lr,
seconds), config.json. Never writes under EXCLUDED.

Usage: python scripts/paper/simclr_train.py --seed 7 [--epochs 100] [--out DIR] [--device mps]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
RDT_DATA = os.environ.get("RDT_DATA", "/data/rdt")  # per-run data (README.md)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_lib import CIFAR, MEAN, STD, build_model, cifar_batch, cifar_test  # noqa: E402

try:
    from torchvision.transforms._functional_tensor import _hsv2rgb, _rgb2hsv
except ImportError:  # older torchvision
    from torchvision.transforms.functional_tensor import _hsv2rgb, _rgb2hsv

OUT_DEFAULT = Path(RDT_DATA + "/extended/experiments/simclr_train")
EXCLUDED = RDT_DATA + "/core"
CKPT_EPOCHS = (0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
GRAY_W = (0.2989, 0.587, 0.114)  # torchvision rgb_to_grayscale


# data
def cifar_train_images() -> np.ndarray:
    """CIFAR-100 training images (50,000, 3, 32, 32) uint8, file order; same source as transfer_lib.cifar_test."""
    with open(CIFAR / "train", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["data"].reshape(-1, 3, 32, 32)


def cifar_test_coarse(y_fine: np.ndarray) -> np.ndarray:
    """Coarse labels of the canonical test order, read from the CIFAR-100 file and checked against the fine labels."""
    with open(CIFAR / "test", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    yf, yc = np.array(d["fine_labels"]), np.array(d["coarse_labels"])
    sc = np.full(100, -1)
    sc[yf] = yc
    assert all((yc[yf == c] == sc[c]).all() for c in range(100))
    return sc[y_fine]


# augmentation (batched)
def _gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(GRAY_W, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _u(n, lo, hi, g):
    return torch.empty(n).uniform_(lo, hi, generator=g)


def augment(x: torch.Tensor, g: torch.Generator, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3), jitter=(0.4, 0.4, 0.4, 0.1),
            p_jitter=0.8, p_gray=0.2) -> torch.Tensor:
    """One SimCLR view of a batch x (B, 3, 32, 32) in [0, 1]; returns the view in [0, 1]."""
    B, dev = x.shape[0], x.device
    # RandomResizedCrop: torchvision's sampler (10 attempts, then the whole image), box as fractions of the side
    area = torch.empty(B, 10).uniform_(*scale, generator=g)
    r = torch.exp(torch.empty(B, 10).uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=g))
    w, h = torch.sqrt(area * r), torch.sqrt(area / r)
    ok = (w <= 1) & (h <= 1)
    k = ok.float().argmax(1)
    ar = torch.arange(B)
    w, h = torch.where(ok.any(1), w[ar, k], 1.0), torch.where(ok.any(1), h[ar, k], 1.0)
    x0, y0 = torch.rand(B, generator=g) * (1 - w), torch.rand(B, generator=g) * (1 - h)
    flip = torch.rand(B, generator=g) < 0.5
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = torch.where(flip, -w, w)
    theta[:, 0, 2] = 2 * x0 + w - 1
    theta[:, 1, 1] = h
    theta[:, 1, 2] = 2 * y0 + h - 1
    grid = F.affine_grid(theta.to(dev), list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # ColorJitter with probability p_jitter, the four ops in a random order per sample
    bj, cj, sj, hj = jitter
    fac = torch.stack([_u(B, max(0, 1 - bj), 1 + bj, g), _u(B, max(0, 1 - cj), 1 + cj, g),
                       _u(B, max(0, 1 - sj), 1 + sj, g), _u(B, -hj, hj, g)], 1)
    order = torch.argsort(torch.rand(B, 4, generator=g), 1)
    apply = torch.rand(B, generator=g) < p_jitter
    for pos in range(4):  # masks stay on the CPU (no device syncs); rows are gathered/scattered by index
        for op in range(4):
            idx = torch.nonzero(apply & (order[:, pos] == op)).flatten()
            if idx.numel() == 0:
                continue
            f = fac[idx, op].view(-1, 1, 1, 1).to(dev)
            idx = idx.to(dev)
            xm = x.index_select(0, idx)
            if op == 0:  # brightness
                xm = (f * xm).clamp(0, 1)
            elif op == 1:  # contrast (blend with the mean grey level)
                xm = (f * xm + (1 - f) * _gray(xm).mean((1, 2, 3), keepdim=True)).clamp(0, 1)
            elif op == 2:  # saturation (blend with the grey image)
                xm = (f * xm + (1 - f) * _gray(xm)).clamp(0, 1)
            else:  # hue
                hsv = _rgb2hsv(xm)
                hsv = torch.cat([(hsv[:, :1] + f) % 1.0, hsv[:, 1:]], 1)
                xm = _hsv2rgb(hsv)
            x = x.index_copy(0, idx, xm)
    gray = torch.nonzero(torch.rand(B, generator=g) < p_gray).flatten()
    if gray.numel():
        gray = gray.to(dev)
        x = x.index_copy(0, gray, _gray(x.index_select(0, gray)).expand(-1, 3, -1, -1).contiguous())
    return x


def normalise(x01: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


# model / loss
def build_encoder() -> nn.Module:
    enc = build_model("ResNet-18", 100)
    enc.fc = nn.Identity()  # forward returns the 512-d avgpool vector
    return enc


def build_head() -> nn.Module:
    return nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, t: float):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    s = z @ z.T / t
    s = s.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(s, tgt), (s.argmax(1) == tgt).float().mean()


# checkpoints
def _atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def pooled_test(enc: nn.Module, dev: torch.device, X: np.ndarray) -> np.ndarray:
    was = enc.training
    enc.eval()
    out = np.concatenate([enc(cifar_batch(X[s:s + 500]).to(dev)).float().cpu().numpy() for s in range(0, len(X), 500)])
    enc.train(was)
    assert out.shape == (len(X), 512) and np.isfinite(out).all()
    assert np.abs(out).max() < 6e4, "activations overflow float16"
    return out.astype(np.float16)


def save_checkpoint(d: Path, epoch: int, enc, head, dev, test):
    e = d / f"epoch_{epoch:03d}"
    e.mkdir(parents=True, exist_ok=True)
    _atomic_save({k: v.detach().cpu() for k, v in enc.state_dict().items()}, e / "encoder.pt")
    _atomic_save({k: v.detach().cpu() for k, v in head.state_dict().items()}, e / "projection_head.pt")
    X = pooled_test(enc, dev, test["X"])
    tmp = e / "pooled_test.tmp.npz"
    np.savez(tmp, X=X, y_fine=test["y"].astype(np.int16), y_super=test["y_super"].astype(np.int16), epoch=epoch)
    os.replace(tmp, e / "pooled_test.npz")


def lr_at(cfg, epoch0: int) -> float:
    lr0 = cfg["lr_base"] * cfg["batch"] / cfg["lr_ref"]
    return lr0 * 0.5 * (1 + math.cos(math.pi * epoch0 / cfg["total_epochs"]))


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=100, help="stop after this epoch (resumable)")
    ap.add_argument("--total-epochs", type=int, default=100, help="length of the cosine schedule")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="root; the run goes to <out>/seed<S>")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr-base", type=float, default=0.06)
    ap.add_argument("--lr-ref", type=float, default=256)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    d = (a.out / f"seed{a.seed}").resolve()
    assert not str(d).startswith(EXCLUDED), f"read-only data volume, not writing under {EXCLUDED}"
    assert a.epochs <= a.total_epochs
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(seed=a.seed, total_epochs=a.total_epochs, batch=a.batch, lr_base=a.lr_base, lr_ref=a.lr_ref,
               momentum=0.9, weight_decay=5e-4, temperature=a.temperature, crop_scale=[0.2, 1.0],
               jitter=[0.4, 0.4, 0.4, 0.1], p_jitter=0.8, p_gray=0.2, head=[512, 512, 128], arch="ResNet-18",
               ckpt_epochs=list(CKPT_EPOCHS))
    if (d / "config.json").exists():
        old = json.load(open(d / "config.json"))
        assert old == cfg, f"config differs from the stored run: {old} vs {cfg}"
    else:
        json.dump(cfg, open(d / "config.json", "w"), indent=1)

    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    enc, head = build_encoder().to(dev), build_head().to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=lr_at(cfg, 0), momentum=0.9, weight_decay=5e-4)

    test = cifar_test()
    test["y_super"] = cifar_test_coarse(test["y"])
    Xtr = torch.from_numpy(cifar_train_images())  # uint8, CPU; labels are never loaded
    n_steps = Xtr.shape[0] // a.batch

    start = 0
    if (d / "train_state.pt").exists():
        st = torch.load(d / "train_state.pt", map_location="cpu", weights_only=False)
        enc.load_state_dict(st["encoder"])
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        start = st["epoch"]
        print(f"resumed at epoch {start}", flush=True)
    elif not (d / "epoch_000" / "pooled_test.npz").exists():
        save_checkpoint(d, 0, enc, head, dev, test)
        print("saved epoch 0 (initialisation)", flush=True)

    enc.train()
    head.train()
    for ep in range(start + 1, a.epochs + 1):
        lr = lr_at(cfg, ep - 1)
        for pg in opt.param_groups:
            pg["lr"] = lr
        order = torch.from_numpy(np.random.RandomState(1000 * a.seed + ep).permutation(Xtr.shape[0]))
        g = torch.Generator().manual_seed(1000003 * a.seed + ep)
        tot_loss = torch.zeros((), device=dev)
        tot_acc = torch.zeros((), device=dev)
        t0 = time.time()
        for b in range(n_steps):
            x = Xtr[order[b * a.batch:(b + 1) * a.batch]].float() / 255.0
            v = normalise(torch.cat([augment(x, g), augment(x, g)])).to(dev)
            z = head(enc(v))
            loss, acc = nt_xent(z[:a.batch], z[a.batch:], a.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.detach()
            tot_acc += acc.detach()
        if dev.type == "mps":
            torch.mps.synchronize()
        sec = time.time() - t0
        rec = dict(epoch=ep, lr=lr, loss=float(tot_loss) / n_steps, pos_pair_acc=float(tot_acc) / n_steps,
                   seconds=round(sec, 2), steps=n_steps, device=a.device,
                   utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        assert math.isfinite(rec["loss"]), f"non-finite loss at epoch {ep}"
        with open(d / "train_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if ep in CKPT_EPOCHS:
            save_checkpoint(d, ep, enc, head, dev, test)
        _atomic_save(dict(epoch=ep, encoder=enc.state_dict(), head=head.state_dict(), opt=opt.state_dict(),
                          torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(), py_rng=random.getstate(),
                          config=cfg), d / "train_state.pt")
    print("done", d, flush=True)


if __name__ == "__main__":
    main()
