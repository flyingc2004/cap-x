"""Offline 4D versus 5D DIGIT key-force ablation, with whole-seed holdout.

The fifth feature is the causal change of area_gt_0p05mm over 8 physics
ticks (1/30 second). Robot state, desired dynamics, score, and private force
are never feature inputs. Private forces are offline labels/strata only.
No runtime model, prompt, OpenTacBench source, or existing data is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path('/mnt/data/ljz/opentacbench/shadow_digit_force_v1/balanced_50seed_v1')
NAMES = ('mean_mm', 'area_gt_0p01mm', 'area_gt_0p05mm', 'area_gt_0p2mm',
         'area_gt_0p05mm_rate_per_s')


def read_contract():
    path = Path(__file__).resolve().parents[2] / 'capx/integrations/opentac_shadow_force.py'
    spec = importlib.util.spec_from_file_location('digit_force_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def causal_area_rate(features, steps, physics_dt):
    """Exact past-time matching; missing history is unavailable, never zero."""
    steps = np.asarray(steps, np.int64)
    if not len(steps) or np.any(np.diff(steps) <= 0):
        raise ValueError('Expected increasing steps within one episode')
    lag = round((1 / 30) / physics_dt)
    if lag < 1 or not np.isclose(lag * physics_dt, 1 / 30):
        raise ValueError('Physics ticks do not represent a 1/30 second window')
    previous = np.searchsorted(steps, steps - lag)
    valid = (steps - lag >= steps[0]) & (steps[previous] == steps - lag)
    rate = np.full(len(steps), np.nan)
    rate[valid] = (features[valid, 2] - features[previous[valid], 2]) / (lag * physics_dt)
    return rate, valid


def extract(path, contract, expected_sha):
    result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
    meta = result['metadata']
    if (result['status'] != 'complete' or not result['native_info']['success'] or
            meta['piano_sha256'] != expected_sha or meta['range_force_N'] != [0, 2] or
            meta['action_repeat'] != 8):
        raise ValueError(f'Unexpected source collection: {path}')
    sequence = result['native_info']['piano_sequence']
    dt = float(meta['physics_dt_s'])
    with h5py.File(path / 'public.h5', 'r') as public:
        steps = public['physics_step'][:]
        reference = public['tactile/index/reference_depth_m'][:]
        features = np.empty((len(steps), 4))
        for index in range(len(steps)):
            feature, quality = contract.public_depth_features(
                public['tactile/index/depth_m'][index], reference)
            if quality != 'ok':
                raise ValueError(f'Invalid depth at {path} frame {index}: {quality}')
            features[index] = feature
    # Extract all public features and their past-only history BEFORE labels.
    rate, history = causal_area_rate(features, steps, dt)
    with h5py.File(path / 'private_labels.h5', 'r') as private:
        private_steps = private['physics_step'][:]
        aligned = np.searchsorted(private_steps, steps)
        if np.any(aligned >= len(private_steps)) or not np.array_equal(private_steps[aligned], steps):
            raise ValueError(f'Private/public step misalignment: {path}')
        active = private['score_count'][:].astype(int)
        requested = private['requested_midi'][:].astype(int)
        forces = private['key_contact_force_N'][:]
        frame_note = active[aligned]
        selected = ((frame_note >= 0) & (frame_note < len(sequence['midi'])) &
                    (requested[aligned] == np.asarray(sequence['midi'])[np.clip(frame_note, 0, len(sequence['midi']) - 1)]))
        # Preserve original per-episode force-bin sampling, then remove missing
        # history from BOTH variants. Neither gets different training rows.
        ids = np.flatnonzero(selected)
        y_all = forces[aligned[ids], requested[aligned[ids]] - 21].astype(float)
        sampled = np.zeros(len(ids), bool)
        bins = np.floor(y_all / .1).astype(int)
        for value in np.unique(bins):
            group = np.flatnonzero(bins == value)
            sampled[group[np.linspace(0, len(group) - 1, min(30, len(group))).astype(int)]] = True
        keep = history[ids]
        sampled = sampled[keep]
        excluded = int(np.count_nonzero(~keep))
        ids = ids[keep]
        index = frame_note[ids]
        y = forces[aligned[ids], requested[aligned[ids]] - 21].astype(float)
        phase = np.full(len(ids), 'no_contact', dtype='U16')
        # Offline strata only: contact onset and force trend are private labels,
        # not predictor inputs or a runtime phase API.
        for note_index, midi in enumerate(sequence['midi']):
            physics = np.flatnonzero((active == note_index) & (requested == midi) &
                                     (forces[:, midi - 21] >= .05))
            if not len(physics):
                continue
            relevant = np.flatnonzero((index == note_index) & (y >= .05))
            p = aligned[ids[relevant]]
            previous = np.maximum(p - 8, 0)
            trend = (forces[p, midi - 21] - forces[previous, midi - 21]) / np.maximum((p - previous) * dt, dt)
            phase[relevant] = np.where(trend > .05, 'loading', np.where(trend < -.05, 'unloading', 'steady'))
            initial = (steps[ids[relevant]] - private_steps[physics[0]]) * dt <= .12
            phase[relevant[initial]] = 'initial_contact'
    data = dict(seed=np.full(len(ids), meta['seed'], np.int64),
                x=np.column_stack((features[ids], rate[ids])), y=y, sampled=sampled,
                steps=steps[ids], time_s=steps[ids] * dt, note_index=index,
                key=np.asarray(sequence['midi'])[index],
                black=np.asarray(['#' in sequence['notes'][i] for i in index]),
                heavy=np.asarray([sequence['dynamics'][i] == 'heavy' for i in index]), phase=phase)
    if not np.isfinite(data['x']).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError(f'Invalid features/labels: {path}')
    return data, dict(seed=meta['seed'], source=str(path), public_frames=len(steps),
                      aligned_frames=len(ids), missing_history_frames=excluded,
                      selected_training_frames=int(sampled.sum()), sensor_sha256=meta['piano_sha256'])


def metrics(y, prediction):
    if not len(y):
        return {'count': 0}
    error = prediction - y
    return dict(count=len(y), mae_N=float(np.abs(error).mean()),
                rmse_N=float(np.sqrt((error ** 2).mean())),
                p95_absolute_error_N=float(np.percentile(np.abs(error), 95)), bias_N=float(error.mean()))


def fit(data, dimensions):
    train = (data['seed'] < 3140) & data['sampled']
    prediction = np.full(len(data['y']), np.nan)
    ood = np.zeros(len(data['y']), bool)
    per_feature_ood = np.zeros((len(data['y']), dimensions), bool)
    models = {}
    for color, black in (('white', False), ('black', True)):
        use = train & (data['black'] == black)
        target = data['black'] == black
        x, y = data['x'][use, :dimensions], data['y'][use]
        median = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        iqr[iqr < 1e-9] = 1
        z = (x - median) / iqr
        regression = Ridge(alpha=.01).fit(np.c_[z, z * z], y)
        ztest = (data['x'][target, :dimensions] - median) / iqr
        prediction[target] = regression.predict(np.c_[ztest, ztest * ztest])
        lower, upper = x.min(axis=0) - .05 * np.ptp(x, axis=0), x.max(axis=0) + .05 * np.ptp(x, axis=0)
        lower[:4] = np.maximum(0, lower[:4])  # The derivative is signed.
        outside = (data['x'][target, :dimensions] < lower) | (data['x'][target, :dimensions] > upper)
        per_feature_ood[target] = outside
        ood[target] = np.any(outside, axis=1)
        models[color] = dict(median=median.tolist(), iqr=iqr.tolist(),
                             coef=regression.coef_.tolist(), intercept=float(regression.intercept_),
                             feature_min=lower.tolist(), feature_max=upper.tolist(), training_frames=int(use.sum()))
    return prediction, ood, per_feature_ood, models


def evaluate(data, prediction, ood, per_feature_ood):
    holdout = data['seed'] >= 3140
    y, black, heavy = data['y'], data['black'], data['heavy']
    near = np.abs(y - np.where(black, np.where(heavy, 1., .7), np.where(heavy, .5, .35))) <= np.where(black, .14, .07)
    upper = np.where(black, np.where(heavy, 1.14, .84), np.where(heavy, .57, .42))
    contact = y >= .1
    masks = dict(overall=holdout, contact=holdout & contact,
                 near_native_target=holdout & near,
                 low_force_0p05_to_0p25_N=holdout & (y >= .05) & (y <= .25),
                 white_heavy_initial=holdout & ~black & heavy & (data['phase'] == 'initial_contact'),
                 white_heavy_native_band=holdout & ~black & heavy & near)
    for phase in ('no_contact', 'initial_contact', 'loading', 'steady', 'unloading'):
        masks[f'phase_{phase}'] = holdout & (data['phase'] == phase)
    for color, color_flag in (('white', False), ('black', True)):
        for dynamic, dynamic_flag in (('light', False), ('heavy', True)):
            group = holdout & (black == color_flag) & (heavy == dynamic_flag)
            masks[f'{color}_{dynamic}_contact'] = group & contact
            masks[f'{color}_{dynamic}_initial'] = group & (data['phase'] == 'initial_contact')
    for key in np.unique(data['key'][holdout]):
        masks[f'key_{key}_contact'] = holdout & (data['key'] == key) & contact
    report = {name: metrics(y[mask], prediction[mask]) for name, mask in masks.items()}
    report['by_seed_contact'] = {str(seed): metrics(y[holdout & (data['seed'] == seed) & contact],
                                                   prediction[holdout & (data['seed'] == seed) & contact])
                                  for seed in range(3140, 3150)}
    # Replicate the existing runtime gates separately from raw regression errors.
    depth_no_contact = ((data['x'][:, 0] < .005) & (data['x'][:, 1:4] == 0).all(axis=1))
    available = ~ood & ~depth_no_contact & np.isfinite(prediction) & (prediction >= .05)
    no_contact = holdout & (y < .05)
    overforce = holdout & (y > upper)
    report['quality'] = dict(
        heldout_ood_frames=int((holdout & ood).sum()),
        heldout_ood_by_feature=dict(zip(NAMES, per_feature_ood[holdout].sum(axis=0).tolist())),
        contact_ood_frames=int((holdout & contact & ood).sum()),
        contact_unavailable_frames=int((holdout & contact & ~available).sum()),
        contact_frames=int((holdout & contact).sum()),
        accepted_contact=metrics(y[holdout & contact & available], prediction[holdout & contact & available]),
        no_contact_frames=int(no_contact.sum()),
        no_contact_false_positive_frames=int((no_contact & available & (prediction >= .1)).sum()),
        overforce_frames=int(overforce.sum()),
        overforce_unavailable_frames=int((overforce & ~available).sum()),
        overforce_underestimated_frames=int((overforce & available & (prediction <= upper)).sum()),
        predictions_clipped=False)
    lags = {}
    for seed in range(3140, 3150):
        mask = data['seed'] == seed
        truth, estimate = y[mask], prediction[mask]
        errors = {}
        for lag in range(-4, 5):
            actual = truth[:len(truth) - lag] if lag >= 0 else truth[-lag:]
            shifted = estimate[lag:] if lag >= 0 else estimate[:len(estimate) + lag]
            relevant = (actual >= .05) | (shifted >= .05)
            errors[lag] = float(np.abs(shifted[relevant] - actual[relevant]).mean())
        lags[str(seed)] = min(errors, key=errors.get) / 30
    report['best_alignment_lag_seconds_by_seed'] = lags
    report['lag_correction_applied'] = False
    return report


def plot(data, predictions, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for column, seed in enumerate((3140, 3145)):
        candidates = np.flatnonzero((data['seed'] == seed) & ~data['black'] & data['heavy'] & (data['y'] >= .05))
        index = data['note_index'][candidates[0]]
        mask = (data['seed'] == seed) & (data['note_index'] == index)
        x = data['time_s'][mask]
        x = x - x[0]
        axes[0, column].plot(x, data['y'][mask], label='Private measured key force', color='black')
        for name, prediction in predictions.items():
            axes[0, column].plot(x, prediction[mask], label=name, alpha=.85)
        axes[0, column].set(title=f'Heldout seed {seed}: white heavy, note index {index}',
                            xlabel='Time in note (s)', ylabel='Key-top force (N)')
        axes[1, column].plot(x, data['x'][mask, 2], label='Area > 0.05 mm', color='tab:green')
        right = axes[1, column].twinx()
        right.plot(x, data['x'][mask, 4], label='Causal area rate', color='tab:red', alpha=.7)
        right.set_ylabel('Area fraction / second')
        axes[1, column].set(xlabel='Time in note (s)', ylabel='Contact area fraction')
        for row in range(2):
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(fontsize=8, loc='upper left')
    fig.savefig(output / 'heldout_alignment.png', dpi=150)
    plt.close(fig)
    mask = (data['seed'] >= 3140) & (data['y'] >= .05) & (data['phase'] == 'initial_contact')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, (name, prediction) in zip(axes, predictions.items()):
        for black, color in ((False, 'tab:blue'), (True, 'tab:orange')):
            rows = mask & (data['black'] == black)
            ax.scatter(data['y'][rows], prediction[rows], s=14, alpha=.5,
                       label='black keys' if black else 'white keys', color=color)
        ax.plot([0, 1.2], [0, 1.2], 'k--', linewidth=1)
        ax.set(title=f'{name}: heldout initial contact', xlabel='Private measured force (N)', ylabel='Estimated force (N)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / 'initial_contact_scatter.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    cache = args.output / 'private-feature-cache'
    cache.mkdir(mode=0o700)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frozen_path = args.root / 'shadow_digit_key_top_force_4d.v1.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    contract = read_contract()
    episodes, sources = [], []
    for seed in range(3100, 3150):
        paths = list((args.root / 'episodes').glob(f'{seed}_expert_*'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one source for seed {seed}')
        episode, source = extract(paths[0], contract, frozen['sensor_sha256'])
        with (cache / f'{seed}.private.npz').open('xb') as stream:
            np.savez_compressed(stream, **episode)
        episodes.append(episode)
        sources.append(source)
        print(f"seed={seed} frames={len(episode['y'])} history_excluded={source['missing_history_frames']}", flush=True)
    data = {key: np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]}
    report = dict(schema='shadow_digit_area_rate_ablation.v1',
                  source_root=str(args.root), script_sha256=script_sha,
                  sensor_sha256=frozen['sensor_sha256'], feature_names=list(NAMES),
                  feature_window_seconds=1/30, area_rate_units='fraction/second',
                  training_seeds=list(range(3100, 3140)), held_out_seeds=list(range(3140, 3150)),
                  fixed_ridge_alpha=.01, normalization='training Median/IQR; linear and elementwise squared terms',
                  training_sampling='same force-bin-selected rows for both; max 30 per 0.1 N bin per episode',
                  runtime_sidecar_emitted=False, future_samples_used=False, robot_state_used=False,
                  target_dynamics_used_as_input=False, source_episodes=sources, models={})
    models, predictions = {}, {}
    saved = {key: data[key] for key in ('seed', 'steps', 'y', 'black', 'heavy', 'phase', 'key')}
    for name, dimensions in (('depth_4d', 4), ('depth_area_rate_5d', 5)):
        prediction, ood, per_feature_ood, model = fit(data, dimensions)
        report['models'][name] = evaluate(data, prediction, ood, per_feature_ood)
        models[name] = dict(feature_names=list(NAMES[:dimensions]), groups=model)
        predictions[name] = prediction
        saved[f'{name}_prediction'] = prediction
        saved[f'{name}_ood'] = ood
    # Paired uncertainty is over complete seeds, not thousands of correlated frames.
    delta = np.array([report['models']['depth_area_rate_5d']['by_seed_contact'][str(seed)]['mae_N'] -
                      report['models']['depth_4d']['by_seed_contact'][str(seed)]['mae_N']
                      for seed in range(3140, 3150)])
    rng = np.random.default_rng(20261024)
    bootstrap = rng.choice(delta, size=(4000, len(delta)), replace=True).mean(axis=1)
    report['paired_seed_contact_MAE_difference_5d_minus_4d'] = dict(
        mean_N=float(delta.mean()), bootstrap_95_interval_N=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        improved_seeds=int((delta < 0).sum()), seed_differences_N=delta.tolist())
    report['limitations'] = [
        'Successful expert trajectories at 30 Hz only; no new CaP-X or 60 Hz contact-geometry coverage.',
        'Historical CaP-X OOD raw depth and its causal history were not saved, so those failures cannot be replayed.',
        'Adding a fifth box constraint cannot remove existing four-dimensional OOD at unchanged bounds.',
        'Private-force-based phase/target strata are evaluation labels only; not predictor inputs.',
        'Current key-top force is a TACTO rendering input: this tests simulated image inversion, not independent force measurement.',
        'Existing holdout seeds have been audited previously; no hyperparameter was selected on this comparison.',
    ]
    (args.output / 'comparison_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (args.output / 'models_offline_only.json').write_text(json.dumps(models, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'predictions.private.npz').open('xb') as stream:
        np.savez_compressed(stream, **saved)
    plot(data, predictions, args.output)
    for name, result in report['models'].items():
        print(name, json.dumps({key: result[key] for key in ('overall', 'contact', 'phase_initial_contact',
                                                          'white_heavy_initial', 'quality')}, ensure_ascii=False), flush=True)
    print(f'REPORT={args.output / "comparison_report.json"}', flush=True)


if __name__ == '__main__':
    main()
