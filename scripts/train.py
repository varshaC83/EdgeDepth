"""Train EdgeDepthNet on the NYU Depth V2 subset (CPU-friendly).

    python scripts/train.py --iters 4000 --batch 8

* Training frames: data/nyu/train (scene-disjoint from val/test).
* Each iteration draws random 128x160 crops, applies a random degradation
  (2x/4x/8x low-res or 1-10% sparse; Kinect-model noise; random/blob/edge
  holes) and trains with edge-weighted L1 + gradient loss + boundary BCE.
* Every --val-every iterations the model is evaluated on the FULL validation
  frames at 2x/4x/8x (clean and noisy+missing, fixed seeds); the checkpoint
  with the lowest mean validation RMSE is kept.
* A JSON log is written to results/training_log.json.
On a GPU (Colab/Kaggle) pass --device cuda (e.g. --iters 20000 --batch 16).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.metrics import rmse  # noqa: E402
from edgedepth.model.dataset import load_split, sample_batch  # noqa: E402
from edgedepth.model.inference import predict  # noqa: E402
from edgedepth.model.losses import edgedepth_loss  # noqa: E402
from edgedepth.model.network import EdgeDepthNet, count_parameters  # noqa: E402

VAL_CONDS = {"clean": {}, "noisy_missing": dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)}


def validate(model, val_frames):
    model.eval()
    res = {}
    for i, (_, rgb, gt) in enumerate(val_frames):
        for s in (2, 4, 8):
            for c, kw in VAL_CONDS.items():
                obs = degrade(gt, DegradationConfig(scale=s, seed=1000 + i, **kw))
                out = predict(rgb, obs, model)
                res.setdefault(f"{s}x_{c}", []).append(rmse(out["refined"], gt))
    model.train()
    res = {k: float(np.mean(v)) for k, v in res.items()}
    res["mean"] = float(np.mean(list(res.values())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nyu")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--out", default="checkpoints/edgedepthnet.pt")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="stop early after this wall-clock budget")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    print(f"train frames {len(train)}  val frames {len(val)}")

    dev = torch.device(args.device)
    model = EdgeDepthNet(base=args.base).to(dev)
    n_params = count_parameters(model)
    print(f"EdgeDepthNet parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)) * 0.97 + 0.03)

    log = {"args": vars(args), "params": n_params, "train_frames": len(train), "val_frames": len(val), "history": [], "val": []}
    best = float("inf")
    t0 = time.time()
    run = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for it in range(1, args.iters + 1):
        b = {k: v.to(dev) for k, v in sample_batch(train, args.batch, rng=rng).items()}
        out = model(b["rgb"], b["d0"], b["m"], b["s"])
        loss, parts = edgedepth_loss(out, b["gt"], b["valid"], b["edge"], b["band"], b["mu"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in {"loss": float(loss), **parts}.items():
            run[k] = run.get(k, 0) + v
        if it % 50 == 0:
            avg = {k: v / 50 for k, v in run.items()}
            run = {}
            log["history"].append({"iter": it, "time_s": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0], **avg})
            print(f"it {it:5d}  {time.time() - t0:7.0f}s  " + "  ".join(f"{k} {v:.4f}" for k, v in avg.items()), flush=True)
        stop = (time.time() - t0) / 60 > args.max_minutes
        if it % args.val_every == 0 or it == args.iters or stop:
            v = validate(model, val)
            v["iter"] = it
            log["val"].append(v)
            print(f"  VAL it {it}: " + "  ".join(f"{k} {x:.4f}" for k, x in v.items() if k != "iter"), flush=True)
            if v["mean"] < best:
                best = v["mean"]
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "model_args": {"base": args.base}, "iter": it,
                            "val_rmse": v, "params": n_params, "train_frames": len(train),
                            "trained_on": "NYU Depth V2 subset (scene-disjoint train split)",
                            "train_time_min": round((time.time() - t0) / 60, 1)}, args.out)
                print(f"  saved best checkpoint (val mean RMSE {best:.4f})", flush=True)
            with open("results/training_log.json", "w") as f:
                json.dump(log, f, indent=1)
        if stop:
            print("time budget reached")
            break
    log["best_val_mean_rmse"] = best
    log["total_time_min"] = round((time.time() - t0) / 60, 1)
    with open("results/training_log.json", "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
