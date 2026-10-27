"""Frozen public-only representation diagnostics. No skill relevance is assumed."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import SCHEMA, WINDOW, Corpus, read_json, sha256, write_json
from .model import augment, load_checkpoint, unit_rows
from .train import tensors


def check_freeze(run):
    run = Path(run)
    freeze = read_json(run / "freeze.json")
    for relative, expected in freeze["weights"].items():
        if sha256(run / relative) != expected:
            raise ValueError(f"Frozen checkpoint changed: {relative}")
    for name in ("normalization", "manifest"):
        field = "normalization_sha256" if name == "normalization" else "cache_manifest_sha256"
        if sha256(run / "cache" / f"{name}.json") != freeze[field]:
            raise ValueError(f"Frozen {name} changed")
    return freeze


def nearest(queries, gallery, k=5):
    indices, scores = [], []
    k = min(k, len(gallery))
    if k <= 0:
        raise ValueError("Empty retrieval gallery")
    for start in range(0, len(queries), 128):
        sim = queries[start:start+128] @ gallery.T
        chosen = np.argpartition(-sim, k-1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, chosen, axis=1), axis=1, kind="stable")
        chosen = np.take_along_axis(chosen, order, axis=1)
        indices.append(chosen)
        scores.append(np.take_along_axis(sim, chosen, axis=1))
    return np.concatenate(indices), np.concatenate(scores)


def encode_ids(model, data, ids, device, *, perturb=False, shuffle=False):
    generator = torch.Generator(device=device).manual_seed(12345)
    # Last frame stays fixed: a last-frame-only model cannot pass as temporal.
    order = np.r_[np.random.default_rng(1729).permutation(15), 15]
    results = []
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            x, motion = tensors(data, ids[start:start+128], device)
            if perturb:
                x = augment(x, generator)
            if shuffle:
                x, motion = x[:, order], motion[:, order]
            results.append(model(x, motion).cpu().numpy())
    return np.concatenate(results)


def distribution(z):
    centered = np.asarray(z, np.float64)-np.mean(z, axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T@centered/max(len(z)-1, 1)), 0)
    p = eigenvalues / max(eigenvalues.sum(), 1e-30)
    entropy = -np.sum(p[p > 0]*np.log(p[p > 0]))
    std = np.std(z, axis=0)
    return dict(mean_dimension_std=float(std.mean()), max_dimension_std=float(std.max()),
                effective_rank=float(np.exp(entropy)) if eigenvalues.sum() > 1e-20 else 0.,
                constant_output_warning=bool(std.max() < 1e-3),
                interpretation="Low rank can reflect restricted data; it is not by itself proof of collapse.")


def benchmark(model, data, ids, device):
    x, motion = tensors(data, ids[:128], device)
    synchronize = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
    with torch.inference_mode():
        for _ in range(2): model(x, motion)
        timings = []
        for _ in range(5):
            synchronize(); tic = time.perf_counter()
            model(x, motion)
            synchronize(); timings.append(time.perf_counter()-tic)
    return dict(batch_size=len(x), batch_median_ms=float(np.median(timings)*1000),
                amortized_ms_per_window=float(np.median(timings)*1000/len(x)),
                preprocessing_and_io_included=False, device=str(device))


def _plot_neighbors(data, query_ids, retrieved_ids, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    active = np.flatnonzero(data.active[query_ids])
    choices = active[np.linspace(0, len(active)-1, min(4, len(active))).astype(int)] if len(active) else np.arange(min(4,len(query_ids)))
    fig, axes = plt.subplots(len(choices)*2, 16, figsize=(16, 2.3*len(choices)), squeeze=False)
    for j, choice in enumerate(choices):
        q, n = int(query_ids[choice]), int(retrieved_ids[choice, 0])
        frames = [np.asarray(data.maps[data.frames[i], 0]) for i in (q, n)]
        vmax = max(.005, max(float(np.max(np.abs(a))) for a in frames))
        for row, (wid, images) in enumerate(zip((q,n), frames)):
            ep = data.episodes[int(data.episode[wid])]
            for col, frame in enumerate(images):
                axes[j*2+row,col].imshow(frame, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[j*2+row,col].set_xticks([]); axes[j*2+row,col].set_yticks([])
            axes[j*2+row,0].set_ylabel(f"{'query' if row==0 else 'top1'}\nseed {ep['seed']}\nstep {data.steps[data.frames[wid,-1]]}", fontsize=7)
    fig.suptitle("Public signed depth (mm), oldest to newest; per-pair shared colour scale")
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate_method(run, data, name, raw_keys, *, encoder_version, model=None, device="cpu", baseline=None):
    folder = Path(run) / "evaluation" / name
    folder.mkdir(parents=True, exist_ok=False)
    keys = unit_rows(raw_keys)
    train_ids = data.ids("train")
    gallery = keys[train_ids]
    np.savez(folder / "index.npz", keys=gallery, window_ids=train_ids)
    write_json(str(folder / "index.npz") + ".json", dict(schema=SCHEMA, encoder_version=encoder_version,
        sensor_sha256=data.manifest["sensor_sha256"], window=WINDOW, dimension=keys.shape[1],
        cache_manifest_sha256=sha256(data.path / "manifest.json"),
        gallery_split="train", skill_ids_available=False))
    np.save(folder / "all_keys.npy", keys)
    result = dict(name=name, encoder_version=encoder_version, gallery_windows=len(train_ids), splits={})
    order = np.r_[np.random.default_rng(1729).permutation(15), 15]
    for split in ("test", "external"):
        qids = data.query_ids(split)
        qkeys = keys[qids]
        nn, sim = nearest(qkeys, gallery)
        neighbors = train_ids[nn]
        np.savez(folder / f"neighbors_{split}.npz", query_window_ids=qids, neighbor_window_ids=neighbors, similarities=sim)
        if model is not None:
            shuffled = unit_rows(encode_ids(model, data, qids, device, shuffle=True))
            # Separate identity diagnostic, not a cross-seed relevance label.
            identity_ids = data.query_ids(split, min_spacing_s=.5)
            perturbed = unit_rows(encode_ids(model, data, identity_ids, device, perturb=True))
            identity_nn, _ = nearest(perturbed, keys[identity_ids])
            truth = np.arange(len(identity_ids))[:,None]
            correct1, correct5 = (identity_nn[:,:1] == truth).any(1), (identity_nn == truth).any(1)
            active = data.active[identity_ids]
            matching = dict(windows=len(identity_ids), recall_at_1=float(correct1.mean()), recall_at_5=float(correct5.mean()),
                            active_recall_at_1=float(correct1[active].mean()) if active.any() else None,
                            definition="Exact nonoverlapping window identity under mild augmentation; NOT skill relevance")
        else:
            if baseline == "current4":
                shuffled = qkeys.copy()
            else:
                features = data.features[data.frames[qids]][:,order]
                shuffled = unit_rows(((features-data.norm["feature_mean"])/data.norm["feature_std"]).reshape(len(qids),-1))
            matching = {"not_comparable": "Native-resolution 4D features and pooled-map pixel perturbations have different observation paths; no unfair identity comparison reported."}
        shuffled_nn, _ = nearest(shuffled, gallery, 1)
        result["splits"][split] = dict(queries=len(qids), active_queries=int(data.active[qids].sum()),
            distribution=distribution(raw_keys[qids]),
            exact_window_reidentification=matching,
            history_shuffle_last_frame_fixed_cosine=float(np.sum(qkeys*shuffled,1).mean()),
            history_shuffle_top1_changed_fraction=float((shuffled_nn[:,0] != nn[:,0]).mean()),
            mean_top1_similarity=float(sim[:,0].mean()))
        _plot_neighbors(data, qids, neighbors, folder / f"neighbors_{split}.png")
    if model is not None:
        result["encoding_benchmark"] = benchmark(model, data, data.ids("test"), device)
    write_json(folder / "metrics.json", result)
    print(f"evaluate {name} complete", flush=True)
    return result


def encode_cached_window(cache, checkpoint, episode, end_step, device="cpu"):
    data = Corpus(cache)
    model, bundle = load_checkpoint(checkpoint, device)
    if bundle["sensor_sha256"] != data.manifest["sensor_sha256"] or bundle["normalization"] != data.norm:
        raise ValueError("Cache and checkpoint contract differ")
    candidates = [i for i,e in enumerate(data.episodes) if e["id"] == episode]
    ids = np.flatnonzero(np.isin(data.episode,candidates) & (data.steps[data.frames[:,-1]] == end_step))
    result = dict(embedding=None, quality="unavailable", encoder_version=sha256(checkpoint),
                  sensor_sha256=data.manifest["sensor_sha256"], window=WINDOW)
    if len(ids) != 1:
        return result
    z = unit_rows(encode_ids(model, data, ids, device))[0]
    return {**result, "quality":"ok", "embedding":z.tolist(), "episode":episode, "end_step":end_step}


def evaluate_suite(run, device="cpu"):
    run = Path(run)
    freeze = check_freeze(run)
    data = Corpus(run / "cache")
    out = run / "evaluation"
    out.mkdir(exist_ok=False)
    with (out / "window_records.jsonl").open("w", encoding="utf-8") as stream:
        for i, frames in enumerate(data.frames):
            ep = data.episodes[int(data.episode[i])]
            stream.write(json.dumps(dict(window_id=i, episode=ep["id"], seed=ep["seed"], split=ep["split"],
                start_step=int(data.steps[frames[0]]), end_step=int(data.steps[frames[-1]]),
                sample_steps=data.steps[frames].tolist(), skill_id=None)) + "\n")
    results=[]
    normalized = (data.features[data.frames]-np.asarray(data.norm["feature_mean"])) / np.asarray(data.norm["feature_std"])
    for kind, values in (("current4", normalized[:,-1]), ("window4", normalized.reshape(len(normalized),-1))):
        results.append(evaluate_method(run,data,kind,values.astype(np.float32),encoder_version=f"{kind}:{freeze['normalization_sha256']}",baseline=kind))
    for relative, digest in freeze["weights"].items():
        checkpoint=run/relative
        model,bundle=load_checkpoint(checkpoint,device)
        if bundle["normalization"] != data.norm or bundle["sensor_sha256"] != data.manifest["sensor_sha256"]:
            raise ValueError("Checkpoint does not match corpus")
        name=checkpoint.parent.parent.name+"_"+checkpoint.parent.name
        raw=encode_ids(model,data,np.arange(len(data.frames)),device)
        results.append(evaluate_method(run,data,name,raw,encoder_version=digest,model=model,device=device))
    write_json(out/"summary.json",results)
    grouped={}
    for prefix in ("tactile_seed_", "tactile_action_seed_"):
        members=[r for r in results if r["name"].startswith(prefix)]
        grouped[prefix.rstrip('_')]={}
        for split in ("test","external"):
            grouped[prefix.rstrip('_')][split]={}
            for metric in ("history_shuffle_last_frame_fixed_cosine","history_shuffle_top1_changed_fraction"):
                values=[m["splits"][split][metric] for m in members]
                grouped[prefix.rstrip('_')][split][metric]={"mean":float(np.mean(values)),"std":float(np.std(values)),"n":len(values)}
            for metric in ("recall_at_1","recall_at_5","active_recall_at_1"):
                values=[m["splits"][split]["exact_window_reidentification"][metric] for m in members]
                grouped[prefix.rstrip('_')][split][metric]={"mean":float(np.mean(values)),"std":float(np.std(values)),"n":len(values)}
    write_json(out/"three_seed_aggregate.json",grouped)
    check_freeze(run)
    return results
