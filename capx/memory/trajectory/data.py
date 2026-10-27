"""Read-only, allowlisted DIGIT data preparation. No native labels are read here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "digit_trajectory_key.v1"
WINDOW = {"seconds": .5, "samples": 16, "hz": 30, "max_age_seconds": 1 / 30}
SPLITS = {**{s: "train" for s in range(3100, 3130)},
          **{s: "dev" for s in range(3130, 3140)},
          **{s: "test" for s in range(3140, 3150)},
          **{s: "external" for s in range(3200, 3204)}}
DEFAULT_ROOT = Path("/mnt/data/ljz/opentacbench")
DEFAULT_SOURCES = (DEFAULT_ROOT / "shadow_digit_force_v1/balanced_50seed_v1/episodes",
                   DEFAULT_ROOT / "shadow_digit_force_v3/matched_60hz_v1/episodes")
ALLOWED = ("time_s", "physics_step", "action_target", "hand_state",
           "tactile/index/reference_depth_m", "tactile/index/depth_m",
           "tactile/index/valid_mask")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reduce_depth(depth, reference, mask):
    """Signed millimetres, area-weighted 10x10 pooling, original 4D features."""
    depth = np.asarray(depth, np.float32)
    reference = np.asarray(reference, np.float32)
    mask = np.asarray(mask)
    if depth.ndim != 3 or depth.shape[1:] != (240, 320) or reference.shape != (240, 320) or mask.shape != depth.shape:
        raise ValueError("Expected N x 240 x 320 depth/mask and a 240 x 320 reference")
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0) & np.isfinite(reference)[None] & (reference[None] > 0)
    delta = np.zeros_like(depth)
    np.subtract(reference[None], depth, out=delta, where=valid)
    delta *= 1000
    count = valid.sum((1, 2))
    absolute = np.abs(delta)
    features = np.stack([absolute.sum((1, 2)) / np.maximum(count, 1)] +
                        [((absolute > threshold) & valid).sum((1, 2)) / np.maximum(count, 1)
                         for threshold in (.01, .05, .2)], axis=1).astype(np.float32)
    n = len(depth)
    coverage = valid.reshape(n, 24, 10, 32, 10).mean((2, 4)).astype(np.float32)
    summed = delta.reshape(n, 24, 10, 32, 10).mean((2, 4))
    pooled = np.divide(summed, coverage, out=np.zeros_like(summed), where=coverage > 0)
    maps = np.stack([pooled, coverage], axis=1)
    return maps, features, count / (240 * 320) >= .60


def causal_windows(times, good):
    """Only past samples; windows never pad, extrapolate, or cross episodes."""
    times = np.asarray(times, np.float64)
    good = np.asarray(good, bool)
    if times.ndim != 1 or good.shape != times.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid time/quality sequence")
    if len(times) == 0:
        return np.empty((0, 16), np.int64)
    endpoints = np.flatnonzero(times >= times[0] + .5 - 1e-10)
    targets = times[endpoints, None] - np.arange(15, -1, -1)[None] / 30
    # Tolerance only resolves equivalent floating point timestamps, not real future samples.
    ids = np.searchsorted(times, targets + 1e-10, side="right") - 1
    safe = np.maximum(ids, 0)
    ages = targets - times[safe]
    keep = (ids >= 0).all(1) & (ages >= -1e-10).all(1) & (ages <= 1 / 30 + 1e-10).all(1) & good[safe].all(1)
    return ids[keep]


def relative_motion(raw):
    """Target xyz/flexion + actual xyz/flexion; xyz relative to first actual."""
    value = np.array(raw, np.float32, copy=True)
    if value.ndim != 3 or value.shape[2] != 8:
        raise ValueError("Expected B x T x 8 public motion")
    origin = value[:, :1, 4:7].copy()
    value[:, :, :3] -= origin
    value[:, :, 4:7] -= origin
    return value


def no_contact(features):
    return (features[..., 0] < .005) & (features[..., 1:] == 0).all(-1)


def prepare(output, roots=DEFAULT_SOURCES, *, require_full=True):
    """Only public.h5 is opened. Metadata is restricted to provenance/split fields."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for root in roots:
        for path in sorted(Path(root).glob("*/public.h5")):
            with h5py.File(path, "r") as source:
                meta = json.loads(source.attrs["metadata"])
                seed = int(meta["seed"])
                if seed not in SPLITS:
                    continue
                for name in ALLOWED:
                    if name not in source:
                        raise ValueError(f"Missing public dataset {name}: {path}")
                episodes.append(dict(id=path.parent.name, seed=seed, split=SPLITS[seed],
                                     source=str(path.resolve()), frames=len(source["time_s"]),
                                     sensor_sha256=meta["piano_sha256"],
                                     physics_dt_s=float(meta["physics_dt_s"])))
    if not episodes or len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Empty or duplicate episode identifiers")
    if len({e["sensor_sha256"] for e in episodes}) != 1:
        raise ValueError("Sensor configurations must not be mixed")
    if require_full:
        counts = {s: sum(e["seed"] == s for e in episodes) for s in SPLITS}
        if any(counts[s] != (2 if s >= 3200 else 1) for s in SPLITS):
            raise ValueError(f"Expected 50 expert plus 8 external episodes: {counts}")
    total = sum(e["frames"] for e in episodes)
    shapes = {"maps": ((total, 2, 24, 32), np.float32), "features": ((total, 4), np.float32),
              "motion": ((total, 8), np.float32), "times": ((total,), np.float64),
              "steps": ((total,), np.int64), "good": ((total,), bool)}
    arrays = {k: np.lib.format.open_memmap(output / f"{k}.npy", mode="w+", dtype=dtype, shape=shape)
              for k, (shape, dtype) in shapes.items()}
    all_windows, episode_ids = [], []
    fingerprints = {}
    offset = 0
    for episode_id, episode in enumerate(episodes):
        n = episode["frames"]
        episode["offset"] = offset
        h = hashlib.sha256(b"public_depth_motion_v1_block32")
        with h5py.File(episode["source"], "r") as source:
            times = np.asarray(source["time_s"], np.float64)
            steps = np.asarray(source["physics_step"], np.int64)
            target = np.asarray(source["action_target"], np.float32)
            actual = np.asarray(source["hand_state"][:, :4], np.float32)
            reference = np.asarray(source["tactile/index/reference_depth_m"], np.float32)
            if (target.shape != (n, 4) or actual.shape != (n, 4) or not np.isfinite(target).all()
                    or not np.isfinite(actual).all() or np.any(np.diff(steps) <= 0)
                    or not np.allclose(times, steps * episode["physics_dt_s"], atol=1e-8, rtol=0)):
                raise ValueError(f"Invalid public state/timing: {episode['source']}")
            sl = slice(offset, offset + n)
            arrays["times"][sl], arrays["steps"][sl] = times, steps
            arrays["motion"][sl] = np.concatenate([target, actual], 1)
            for item in (times, steps, target, actual, reference):
                h.update(item.tobytes())
            for start in range(0, n, 32):
                stop = min(n, start + 32)
                depth = np.asarray(source["tactile/index/depth_m"][start:stop], np.float32)
                mask = np.asarray(source["tactile/index/valid_mask"][start:stop], np.uint8)
                h.update(depth.tobytes()); h.update(mask.tobytes())
                maps, features, good = reduce_depth(depth, reference, mask)
                dst = slice(offset + start, offset + stop)
                arrays["maps"][dst], arrays["features"][dst], arrays["good"][dst] = maps, features, good
        fingerprint = h.hexdigest()
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate allowed-input episode: {episode['id']} and {fingerprints[fingerprint]}")
        fingerprints[fingerprint] = episode["id"]
        episode["public_content_sha256"] = fingerprint
        windows = causal_windows(times, arrays["good"][offset:offset+n]) + offset
        episode["windows"] = len(windows)
        episode["median_dt_s"] = float(np.median(np.diff(times)))
        all_windows.append(windows.astype(np.int32))
        episode_ids.extend([episode_id] * len(windows))
        offset += n
        print(f"prepare {episode['id']} frames={n} windows={len(windows)}", flush=True)
    for value in arrays.values():
        value.flush()
    windows = np.concatenate(all_windows)
    episode_ids = np.asarray(episode_ids, np.int16)
    active = ~no_contact(arrays["features"][windows]).all(1)
    np.savez(output / "windows.npz", frames=windows, episode=episode_ids, active=active)
    manifest = dict(schema=SCHEMA, window=WINDOW, episodes=episodes,
                    sensor_sha256=episodes[0]["sensor_sha256"], raw_shape=[240, 320], map_shape=[2, 24, 32],
                    public_input_allowlist=list(ALLOWED), frames=total, windows=len(windows),
                    private_data_read=False, source_fingerprint="allowed arrays only; fixed 32-frame chunks")
    write_json(output / "manifest.json", manifest)
    data = Corpus(output)
    train_frames = np.concatenate([np.arange(e["offset"], e["offset"] + e["frames"])
                                   for e in episodes if e["split"] == "train"])
    train_frames = train_frames[data.good[train_frames]]
    count = s1 = s2 = 0.
    for start in range(0, len(train_frames), 512):
        x = data.maps[train_frames[start:start+512]].astype(np.float64)
        w = x[:, 1]
        count += w.sum(); s1 += (x[:, 0] * w).sum(); s2 += (x[:, 0] ** 2 * w).sum()
    mean = s1 / count
    feature_values = data.features[train_frames].astype(np.float64)
    motion_values = relative_motion(data.motion[data.frames[data.ids("train")]]).reshape(-1, 8).astype(np.float64)
    norm = dict(depth_mean=mean, depth_std=max(float(np.sqrt(max(0., s2/count-mean**2))), 1e-6),
                feature_mean=feature_values.mean(0).tolist(), feature_std=np.maximum(feature_values.std(0), 1e-6).tolist(),
                motion_mean=motion_values.mean(0).tolist(), motion_std=np.maximum(motion_values.std(0), 1e-6).tolist(),
                fitted_seeds=[e["seed"] for e in episodes if e["split"] == "train"])
    write_json(output / "normalization.json", norm)
    return manifest


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        if self.manifest["schema"] != SCHEMA or self.manifest["window"] != WINDOW:
            raise ValueError("Incompatible trajectory cache")
        self.episodes = self.manifest["episodes"]
        for name in ("maps", "features", "motion", "times", "steps", "good"):
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False))
        with np.load(self.path / "windows.npz", allow_pickle=False) as w:
            self.frames, self.episode, self.active = (w[name] for name in ("frames", "episode", "active"))
        self.split = np.array([self.episodes[i]["split"] for i in self.episode])
        self.norm = read_json(self.path / "normalization.json") if (self.path / "normalization.json").exists() else None

    def ids(self, split):
        return np.flatnonzero(self.split == split)

    def balanced(self, split, rng):
        ids = self.ids(split)
        active, idle = ids[self.active[ids]], ids[~self.active[ids]]
        if len(active) < 2:
            raise ValueError(f"Insufficient active public windows: {split}")
        idle = rng.choice(idle, min(len(idle), len(active) // 3), replace=False)
        return rng.permutation(np.r_[active, idle])

    def batch_numpy(self, ids):
        indices = self.frames[ids]
        x = np.array(self.maps[indices], copy=True)
        coverage = x[:, :, 1]
        x[:, :, 0] = np.where(coverage > 0, (x[:, :, 0] - self.norm["depth_mean"]) / self.norm["depth_std"], 0)
        motion = relative_motion(self.motion[indices])
        motion = (motion - np.asarray(self.norm["motion_mean"], np.float32)) / np.asarray(self.norm["motion_std"], np.float32)
        return x, motion

    def query_ids(self, split, limit_per_episode=256, min_spacing_s=0.):
        selected = []
        for ep in np.unique(self.episode[self.ids(split)]):
            candidates = np.flatnonzero((self.episode == ep) & (self.split == split))
            if min_spacing_s:
                kept, last = [], -np.inf
                for i in candidates:
                    t = float(self.times[self.frames[i, -1]])
                    if t - last >= min_spacing_s - 1e-8:
                        kept.append(i); last = t
                candidates = np.asarray(kept, int)
            if len(candidates) > limit_per_episode:
                candidates = candidates[np.linspace(0, len(candidates)-1, limit_per_episode).astype(int)]
            selected.extend(candidates.tolist())
        return np.asarray(selected, np.int64)
