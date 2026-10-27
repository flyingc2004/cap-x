"""Exact 5D history, fair population, and public-only supplementary retrieval."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from capx.memory.trajectory.data import SCHEMA, WINDOW, sha256, write_json
from capx.memory.trajectory.rate_baseline import (
    area_rate_features, fit_rate_normalization, public_evaluation, stable_top1,
)


def sample_data(stride=8):
    n = 40
    steps = np.tile(np.arange(n)*stride, 3)
    area = np.tile(np.arange(n)*.005, 3)
    features = np.stack([area/10, area, area/2, area/4], axis=1)
    episodes = [dict(id=f"{seed}_expert", seed=seed, split=split, offset=i*n, frames=n,
                     physics_dt_s=1/240) for i, (seed, split) in enumerate(
                         ((3100, "train"), (3140, "test"), (3200, "external")))]
    return SimpleNamespace(steps=steps, times=steps/240, features=features,
                           good=np.ones(3*n, bool), episodes=episodes,
                           norm=dict(feature_mean=[0]*4, feature_std=[1]*4))


@pytest.mark.parametrize("stride", [4, 8])
def test_exact_rate_and_causality(stride):
    data = sample_data(stride)
    f, past = area_rate_features(data)
    lag = 8//stride
    assert np.isnan(f[:lag, 4]).all()
    assert np.isnan(f[40:40+lag, 4]).all()  # Each episode resets history.
    assert f[lag, 4] == pytest.approx((data.features[lag, 2]-data.features[0, 2])*30)
    valid = past >= 0
    assert np.all(data.steps[valid]-data.steps[past[valid]] == 8)
    old = f.copy()
    data.features[30:40] = 999
    np.testing.assert_array_equal(area_rate_features(data)[0][:30], old[:30])
    data.good[5] = False
    changed, _ = area_rate_features(data)
    assert np.isnan(changed[5, 4]) and np.isnan(changed[5+lag, 4])


def test_missing_exact_step_not_interpolated():
    data = sample_data()
    data.steps[5] -= 1
    data.times = data.steps/240
    f, past = area_rate_features(data)
    assert past[5] == past[6] == -1
    assert np.isnan(f[[5, 6], 4]).all()


def test_rate_normalization_train_only():
    data = sample_data()
    norm = fit_rate_normalization(data, area_rate_features(data)[0])
    data.features[40:] *= 500
    assert norm == fit_rate_normalization(data, area_rate_features(data)[0])
    assert norm["fitted_seeds"] == [3100]


def test_exact_ties_choose_window_id():
    ids, scores = stable_top1(np.array([[1., 0.]]), np.array([[1., 0.], [1., 0.], [0., 1.]]), np.array([9, 3, 1]))
    assert ids.tolist() == [3]
    assert scores[0] == pytest.approx(1)


def test_public_comparison_without_private_files(tmp_path, monkeypatch):
    from capx.memory.trajectory.data import causal_windows
    d = sample_data()
    run = tmp_path/"base"
    cache = run/"cache"
    cache.mkdir(parents=True)
    frames = np.concatenate([causal_windows(d.times[i:i+40], d.good[i:i+40])+i for i in (0, 40, 80)])
    epids = np.repeat(np.arange(3), len(frames)//3)
    arrays = dict(features=d.features, times=d.times, steps=d.steps, good=d.good,
                  maps=np.zeros((120, 2, 24, 32), np.float32), motion=np.zeros((120, 8), np.float32))
    for name, value in arrays.items():
        np.save(cache/f"{name}.npy", value)
    np.savez(cache/"windows.npz", frames=frames, episode=epids, active=np.ones(len(frames), bool))
    write_json(cache/"manifest.json", dict(schema=SCHEMA, window=WINDOW, episodes=d.episodes, sensor_sha256="test"))
    write_json(cache/"normalization.json", d.norm)
    write_json(run/"freeze.json", dict(weights={}, normalization_sha256=sha256(cache/"normalization.json"),
                                      cache_manifest_sha256=sha256(cache/"manifest.json")))
    before = sha256(run/"freeze.json")
    def no_hdf(*args, **kwargs):
        raise AssertionError("Public evaluation must use cached public data only")
    monkeypatch.setattr(h5py, "File", no_hdf)
    out = tmp_path/"supplement"
    report = public_evaluation(run, out)
    assert not report["labels_used"]
    assert sha256(run/"freeze.json") == before
    assert [x["dimension"] for x in report["methods"]] == [64, 80, 68]
    queries, galleries = [], []
    for name in ("window16x4", "window16x5", "window17x4"):
        with np.load(out/name/"index.npz") as f:
            galleries.append(f["window_ids"])
        with np.load(out/name/"neighbors_test.npz") as f:
            queries.append(f["query_window_ids"])
            assert np.isin(f["neighbor_window_ids"], galleries[0]).all()
    for v in queries[1:]: np.testing.assert_array_equal(v, queries[0])
    for v in galleries[1:]: np.testing.assert_array_equal(v, galleries[0])
    assert report["coverage"]["test"]["retained_queries"] < report["coverage"]["test"]["original_queries"]
