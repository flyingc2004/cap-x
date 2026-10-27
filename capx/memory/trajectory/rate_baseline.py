"""Post-freeze, public-only 16 x 5D baseline with exact causal area-rate history."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from capx.integrations.opentac_shadow_force import AREA_RATE_FEATURE_NAMES, AREA_RATE_HISTORY
from .data import Corpus, WINDOW, read_json, sha256, write_json
from .evaluate import check_freeze


def area_rate_features(data):
    """No interpolation, future samples, zero padding, or cross-episode history."""
    result = np.full((len(data.steps), 5), np.nan, np.float64)
    result[:, :4] = data.features
    previous = np.full(len(data.steps), -1, np.int64)
    lag = AREA_RATE_HISTORY["window_physics_steps"]
    seconds = AREA_RATE_HISTORY["window_seconds"]
    for ep in data.episodes:
        if not np.isclose(ep["physics_dt_s"], AREA_RATE_HISTORY["physics_dt_seconds"], rtol=0, atol=1e-12):
            raise ValueError("Unsupported physics time unit")
        offset, n = ep["offset"], ep["frames"]
        steps = np.asarray(data.steps[offset:offset+n])
        if np.any(np.diff(steps) <= 0):
            raise ValueError("Nonmonotonic episode steps")
        rows = np.searchsorted(steps, steps-lag)
        safe = np.clip(rows, 0, n-1)
        good = np.asarray(data.good[offset:offset+n]) & np.isfinite(result[offset:offset+n, :4]).all(1)
        valid = (rows < n) & (steps[safe] == steps-lag) & good & good[safe]
        current = offset + np.flatnonzero(valid)
        past = offset + rows[valid]
        if not np.allclose(data.times[current]-data.times[past], seconds, rtol=0, atol=1e-8):
            raise ValueError("Physics step and observation time disagree")
        previous[current] = past
        result[current, 4] = (result[current, 2]-result[past, 2]) / seconds
    return result, previous


def fit_rate_normalization(data, features):
    selected = np.concatenate([np.arange(ep["offset"], ep["offset"]+ep["frames"])
                               for ep in data.episodes if ep["split"] == "train"])
    selected = selected[np.isfinite(features[selected]).all(1)]
    if not len(selected):
        raise ValueError("No valid training history")
    # Reuse frozen 4D scaling; fit ONLY the added coordinate on training frames.
    return dict(mean=[*data.norm["feature_mean"], float(features[selected, 4].mean())],
                std=[*data.norm["feature_std"], max(float(features[selected, 4].std()), 1e-6)],
                fifth_dimension_fitted_frames=len(selected),
                fitted_seeds=[ep["seed"] for ep in data.episodes if ep["split"] == "train"],
                feature_names=list(AREA_RATE_FEATURE_NAMES), history=dict(AREA_RATE_HISTORY))


def normalize_keys(values):
    values = np.asarray(values, np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms < 1e-12):
        raise ValueError("Invalid retrieval key")
    return values / norms


def stable_top1(queries, gallery, gallery_ids):
    """Float64 cosine; ties resolved by ascending original window id, not partition k."""
    gallery_ids = np.asarray(gallery_ids)
    order = np.argsort(gallery_ids, kind="stable")
    gallery = normalize_keys(gallery[order])
    queries = normalize_keys(queries)
    neighbors, scores = [], []
    for start in range(0, len(queries), 128):
        cosine = queries[start:start+128] @ gallery.T
        rows = np.argmax(cosine, axis=1)
        neighbors.append(gallery_ids[order[rows]])
        scores.append(cosine[np.arange(len(rows)), rows])
    return np.concatenate(neighbors), np.concatenate(scores)


def public_evaluation(run, output):
    run, output = Path(run), Path(output)
    freeze = check_freeze(run)
    output.mkdir(parents=True, exist_ok=False)
    data = Corpus(run/"cache")
    features, previous = area_rate_features(data)
    eligible = np.isfinite(features[data.frames]).all((1, 2))
    norm = fit_rate_normalization(data, features)
    write_json(output/"normalization.json", norm)
    gallery = data.ids("train")
    gallery = gallery[eligible[gallery]]
    queries, coverage = {}, {}
    for split in ("train", "dev", "test", "external"):
        ids = data.ids(split)
        coverage[split] = dict(original_windows=len(ids), retained_windows=int(eligible[ids].sum()))
        if split in ("test", "external"):
            ids = data.query_ids(split)
            queries[split] = ids[eligible[ids]]
            coverage[split].update(original_queries=len(ids), retained_queries=len(queries[split]))
    needed = np.unique(np.concatenate([gallery, *queries.values()]))
    local = np.full(len(data.frames), -1, np.int64)
    local[needed] = np.arange(len(needed))
    frames = data.frames[needed]
    z = (features[frames]-norm["mean"]) / norm["std"]
    past = previous[frames[:, 0]]
    first4 = (np.asarray(data.features[past])-np.asarray(norm["mean"][:4])) / np.asarray(norm["std"][:4])
    values = {
        "window16x4": z[:, :, :4].reshape(len(z), 64),
        "window16x5": z.reshape(len(z), 80),
        "window17x4": np.concatenate([first4[:, None], z[:, :, :4]], axis=1).reshape(len(z), 68),
    }
    inputs = {str(run/"freeze.json"): sha256(run/"freeze.json")}
    for name in ("features.npy", "times.npy", "steps.npy", "good.npy", "windows.npz", "normalization.json", "manifest.json"):
        path = run/"cache"/name
        inputs[str(path)] = sha256(path)
    # Re-evaluate frozen learned keys on exactly the same gallery/query population.
    for relative in freeze["weights"]:
        cp = Path(relative)
        name = cp.parent.parent.name+"_"+cp.parent.name
        path = run/"evaluation"/name/"all_keys.npy"
        values[name] = np.load(path, mmap_mode="r", allow_pickle=False)[needed]
        inputs[str(path)] = sha256(path)
    results, artifacts = [], {}
    for name, keys in values.items():
        tic = time.monotonic()
        folder = output/name
        folder.mkdir()
        np.savez(folder/"index.npz", keys=normalize_keys(keys[local[gallery]]), window_ids=gallery)
        history = dict(WINDOW)
        if name in ("window16x5", "window17x4"):
            history.update(raw_history_seconds=WINDOW["seconds"]+AREA_RATE_HISTORY["window_seconds"],
                           area_rate=dict(AREA_RATE_HISTORY))
        write_json(folder/"index.npz.json", dict(dimension=keys.shape[1], sensor_sha256=data.manifest["sensor_sha256"],
            window=history, normalization_sha256=sha256(output/"normalization.json"),
            private_labels_in_index=False, skill_ids_available=False))
        item = dict(name=name, dimension=keys.shape[1], splits={})
        for split, ids in queries.items():
            neighbors, scores = stable_top1(keys[local[ids]], keys[local[gallery]], gallery)
            if np.any(data.split[neighbors] != "train"):
                raise AssertionError("Non-training gallery entry")
            if np.any([data.episodes[int(data.episode[q])]["seed"] == data.episodes[int(data.episode[n])]["seed"]
                       for q, n in zip(ids, neighbors)]):
                raise AssertionError("Same-seed neighbor leakage")
            path = folder/f"neighbors_{split}.npz"
            np.savez(path, query_window_ids=ids, neighbor_window_ids=neighbors, similarities=scores)
            artifacts[str(path.relative_to(output))] = sha256(path)
            item["splits"][split] = dict(queries=len(ids), mean_similarity=float(scores.mean()))
        item["retrieval_seconds"] = time.monotonic()-tic
        results.append(item)
        print(f"public {name}: complete ({item['retrieval_seconds']:.1f}s)", flush=True)
    check_freeze(run)
    report = dict(base_run=str(run.resolve()), methods=results, coverage=coverage,
                  inputs_sha256=inputs, neighbors_sha256=artifacts,
                  labels_used=False, selection_feedback_to_training=False,
                  nearest_contract="float64 cosine, exact ties choose smallest training window id",
                  scope="Post-hoc baseline comparison; no tuning of weights or features against private labels",
                  history_note="16x5 requires an extra 1/30s before its first frame; 17x4 controls for this extra history.")
    write_json(output/"public_manifest.json", report)
    return report
