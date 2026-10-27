"""Fixed, development-only checkpoint selection for two public-input encoders."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import SCHEMA, WINDOW, Corpus, sha256, write_json
from .model import TrajectoryEncoder, augment, vicreg

CONFIG = dict(lr=.001, weight_decay=.0001, batch_size=128, max_epochs=50,
              patience=10, vicreg_weights=[25, 25, 1], noise_std=.01,
              pixel_drop_probability=.02, training_seeds=[0, 1, 2])


def tensors(data, ids, device):
    x, motion = data.batch_numpy(ids)
    return torch.from_numpy(x).to(device), torch.from_numpy(motion).to(device)


def loss_epoch(model, data, ids, device, generator, optimizer=None):
    model.train(optimizer is not None)
    totals = dict(loss=0., invariance=0., variance=0., covariance=0.)
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(ids), CONFIG["batch_size"]):
            batch = ids[start:start+CONFIG["batch_size"]]
            if len(batch) < 2:
                continue
            x, motion = tensors(data, batch, device)
            z1 = model(augment(x, generator), motion)
            z2 = model(augment(x, generator), motion)
            loss, parts = vicreg(z1, z2)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; stopping without exporting a fabricated key")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for key, value in {"loss": loss, **parts}.items():
                totals[key] += float(value.detach()) * len(batch)
            count += len(batch)
    if not count:
        raise ValueError("Empty train/development epoch")
    return {**{k: v / count for k, v in totals.items()}, "windows": count}


def train_one(data: Corpus, output, *, use_motion, seed, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TrajectoryEncoder(use_motion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    train_generator = torch.Generator(device=device).manual_seed(1000+seed)
    dev_ids = data.balanced("dev", np.random.default_rng(1729))
    best, best_epoch, stale = float("inf"), -1, 0
    started = time.monotonic()
    write_json(output / "config.json", {**CONFIG, "seed": seed, "use_motion": use_motion,
        "device": str(device), "window": WINDOW, "map_shape": [2,24,32],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection": "minimum fixed-augmentation development VICReg loss; no test/native feedback"})
    for epoch in range(CONFIG["max_epochs"]):
        tic = time.monotonic()
        train_ids = data.balanced("train", rng)
        train = loss_epoch(model, data, train_ids, device, train_generator, optimizer)
        dev = loss_epoch(model, data, dev_ids, device, torch.Generator(device=device).manual_seed(2718))
        improved = dev["loss"] < best
        if improved:
            best, best_epoch, stale = dev["loss"], epoch+1, 0
            bundle = dict(schema=SCHEMA, window=dict(WINDOW), use_motion=use_motion, seed=seed,
                          normalization=data.norm, sensor_sha256=data.manifest["sensor_sha256"],
                          state_dict={k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          best_epoch=best_epoch, development_loss=best,
                          config=dict(CONFIG), cache_manifest_sha256=sha256(data.path / "manifest.json"))
            torch.save(bundle, output / "best.pt")
        else:
            stale += 1
        row = dict(epoch=epoch+1, train=train, dev=dev, improved=improved,
                   seconds=time.monotonic()-tic, training_idle_fraction=float((~data.active[train_ids]).mean()))
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"train {'tactile_action' if use_motion else 'tactile'} seed={seed} epoch={epoch+1} "
              f"train={train['loss']:.5f} dev={dev['loss']:.5f} best={best:.5f} "
              f"seconds={row['seconds']:.1f}", flush=True)
        if stale >= CONFIG["patience"]:
            break
    summary = dict(seed=seed, use_motion=use_motion, epochs=epoch+1, best_epoch=best_epoch,
                   development_loss=best, training_seconds=time.monotonic()-started,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   checkpoint_sha256=sha256(output / "best.pt"), checkpoint_bytes=(output / "best.pt").stat().st_size)
    write_json(output / "training_summary.json", summary)
    return summary


def train_suite(cache, run, device="cpu"):
    data, run = Corpus(cache), Path(run)
    summaries = []
    for use_motion in (False, True):
        name = "tactile_action" if use_motion else "tactile"
        for seed in CONFIG["training_seeds"]:
            summaries.append(train_one(data, run / "models" / name / f"seed_{seed}",
                                       use_motion=use_motion, seed=seed, device=device))
    weights = {str(p.relative_to(run)): sha256(p) for p in sorted((run / "models").glob("*/seed_*/best.pt"))}
    write_json(run / "freeze.json", dict(schema=SCHEMA, weights=weights,
        normalization_sha256=sha256(data.path / "normalization.json"),
        cache_manifest_sha256=sha256(data.path / "manifest.json"),
        selection="development loss only", test_used_for_selection=False, private_used_for_selection=False))
    write_json(run / "training_summary.json", summaries)
    return summaries
