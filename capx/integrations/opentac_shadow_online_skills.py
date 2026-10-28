"""Episode-boundary skill reflection from a strictly public evidence file.

This module never opens native audits, run.log, rewards or trial summaries.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import ast
import dis
import sys
import uuid
from collections.abc import Mapping
import numpy as np

import yaml

from capx.integrations.opentac_shadow_keyboard import atomic_json
from capx.llm.deadline import request_deadline
from capx.integrations.opentac_shadow_skills import (
    SCHEMA, MAX_FILE_BYTES, MAX_SKILLS, _PRIVATE_MARKERS, inject_shadow_piano_public_skills,
)

REFLECTION_TIMEOUT_SECONDS = 120


def _evidence_hash(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class _UnavailableTrajectoryEncoder:
    def __init__(self, config, method, error):
        self.contract = {'version': 'unavailable:' + _evidence_hash(dict(config)),
            'representation': method, 'preprocessing': 'unavailable', 'modalities': ['depth_m']}
        self.error_type = type(error).__name__

    def encode(self, window):
        return {'vector': None, 'quality': 'encoder_unavailable', 'error_type': self.error_type}


def trajectory_encoders(config):
    """Resolve an opt-in YAML configuration, or retain the existing plugin API."""
    if config is None:
        return []
    if not isinstance(config, Mapping):
        return list(config) if isinstance(config, (list, tuple)) else [config]
    methods = config.get('representations', ['tactile'])
    if not isinstance(methods, (list, tuple)) or not methods or len(set(methods)) != len(methods):
        raise ValueError('Specify distinct trajectory representations')
    encoders = []
    for method in methods:
        try:
            from capx.memory.trajectory.model import EvidenceTrajectoryEncoder
            encoders.append(EvidenceTrajectoryEncoder(config['run_dir'], method))
        except Exception as exc:
            # Optional retrieval cannot disable the existing 5D controller.
            encoders.append(_UnavailableTrajectoryEncoder(config, method, exc))
    return encoders


def _clip_window(directory, samples, clip, contract, encoder, episode_id):
    """Use same-episode history up to this clip, including previous phases."""
    selected = [s for s in samples if s['physics_step'] <= clip['end_physics_step']]
    spec = encoder.contract.get('window')
    if spec and selected:
        # Keep the previous observation for a causal 30Hz target; never synthesize it.
        start = selected[-1]['timestamp_s'] - spec['seconds'] - spec['max_age_seconds']
        selected = [s for s in selected if s['timestamp_s'] >= start - 1e-10]
    else:
        selected = [s for s in selected if s['clip_id'] == clip['clip_id']]
    return load_public_tactile_window(directory, selected, contract=contract, episode_id=episode_id)


def _temporal_contract(contract, encoder):
    return {**contract, 'encoder': dict(encoder.contract),
            'window': encoder.contract.get('window', 'commanded_phase_segment.v1')}


class TactileEvidenceIndex:
    """Append-only public provenance; representations never replace their source clips."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'associations.jsonl'

    def append(self, kind, payload, links=()):
        record = {'kind': kind, 'payload': payload, 'links': list(links)}
        record['id'] = kind + ':' + _evidence_hash(record)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
        return record['id']

    def records(self):
        if not self.path.exists():
            return []
        # A truncated final append is not a valid node, never guess its contents.
        result = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                content = {k: row[k] for k in ('kind', 'payload', 'links')}
                if row['id'] == row['kind'] + ':' + _evidence_hash(content):
                    result.append(row)
            except (ValueError, KeyError, TypeError):
                continue
        return result

    def related(self, identifier):
        return [r for r in self.records() if r['id'] == identifier or identifier in r['links']]

    def encode_clip(self, clip_id, encoder, *, records=None):
        """Re-encode archived PUBLIC frames without changing the clip or 5D nodes."""
        records = self.records() if records is None else records
        clips = [r['payload'] for r in records if r['kind'] == 'clip' and r['payload']['clip_id'] == clip_id]
        if not clips:
            raise ValueError('Unknown tactile clip')
        clip = clips[-1]
        path = Path(clip['manifest']).resolve()
        if not path.is_relative_to(self.root / 'episodes') or path.name != 'manifest.json':
            raise ValueError('Clip must reference a public episode')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['episode_id'] != clip['episode_id']:
            raise ValueError('Clip episode mismatch')
        window = _clip_window(path.parent, manifest['samples'], clip, manifest['contract'], encoder, manifest['episode_id'])
        result = encode_tactile_window(window, encoder)
        contracts = [r['payload']['contract'] for r in records if r['kind'] == 'representation'
                     and r['payload']['clip_id'] == clip_id and r['payload']['type'] == 'instantaneous_5d']
        contract = _temporal_contract(contracts[-1], encoder)
        return self.put_representation(clip_id, 'temporal', result['vector'], contract,
            result['quality'], clip['end_physics_step'], provenance=result.get('provenance'))

    def encode_history(self, encoders):
        """Backfill CaP-X public clips only, idempotently; no expert gallery import."""
        records = self.records()
        results = []
        for row in records:
            if row['kind'] != 'clip':
                continue
            clip = row['payload']
            for encoder in encoders:
                existing = [r for r in records if r['kind'] == 'representation'
                    and r['payload']['clip_id'] == clip['clip_id'] and r['payload']['type'] == 'temporal'
                    and r['payload']['contract'].get('encoder') == encoder.contract]
                if existing:
                    continue
                try:
                    identifier = self.encode_clip(clip['clip_id'], encoder, records=records)
                    results.append({'clip_id': clip['clip_id'], 'representation_id': identifier})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    results.append({'clip_id': clip['clip_id'], 'error_type': type(exc).__name__})
        return results

    def resolve_skill_evidence(self, clip_id, *, records=None):
        """Follow actual clip -> invocation -> function; never infer a skill label."""
        records = self.records() if records is None else records
        clips = [r['payload'] for r in records if r['kind'] == 'clip' and r['payload']['clip_id'] == clip_id]
        if not clips:
            return None
        clip = clips[-1]
        calls = [r['payload'] for r in records if r['kind'] == 'invocation'
                 and r['payload']['invocation_id'] in clip['invocation_ids']]
        functions = {}
        for row in records:
            f = row['payload']
            if row['kind'] == 'function' and any(i['function_id'] == f['function_id'] for i in calls):
                if _evidence_hash(f['source'].encode()) == f['source_hash']:
                    functions[f['function_id']] = {k: f[k] for k in
                        ('function_id', 'name', 'source', 'source_hash', 'ast_hash')}
        return {'clip_id': clip_id, 'episode_id': clip['episode_id'], 'manifest': clip['manifest'],
                'invocations': [{k: i[k] for k in ('invocation_id', 'function_id', 'start', 'end', 'outcome', 'exceptions')}
                                for i in calls if i['function_id'] in functions],
                'functions': list(functions.values())}

    def put_representation(self, clip_id, kind, vector, contract, quality, physics_step, *, provenance=None):
        if kind not in ('instantaneous_5d', 'temporal'):
            raise ValueError('Unknown tactile representation type')
        if vector is not None:
            value = np.asarray(vector, float)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError('Invalid tactile representation')
            if kind == 'instantaneous_5d' and value.size != 5:
                raise ValueError('The existing instantaneous representation stays 5D')
            vector = value.tolist()
        return self.append('representation', {'clip_id': clip_id, 'type': kind,
            'vector': vector, 'contract': contract, 'quality': quality,
            'physics_step': int(physics_step), 'provenance': provenance}, [clip_id])

    def retrieve(self, kind, vector, contract, *, candidates=None, scales=None, limit=3):
        query = np.asarray(vector, float)
        if query.ndim != 1 or not np.isfinite(query).all():
            return []
        scale = np.ones(query.size) if scales is None else np.asarray(scales, float)
        if scale.shape != query.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid representation distance scales')
        found = []
        for row in self.records() if candidates is None else candidates:
            p = row['payload']
            if row['kind'] != 'representation' or p['type'] != kind or p['contract'] != contract:
                continue
            if p['quality'] != 'ok' or p['vector'] is None:
                continue
            value = np.asarray(p['vector'], float)
            if value.shape != query.shape:
                continue
            delta = (value - query) / scale
            found.append({'representation_id': row['id'], 'clip_id': p['clip_id'],
                          'distance': float(np.sqrt(np.mean(delta * delta)))})
        return sorted(found, key=lambda r: (r['distance'], r['representation_id']))[:limit]


def encode_tactile_window(window, encoder=None):
    """Optional parallel encoder. Inputs are public; sampling remains causal.

    Host plugins implement encode(window) and contract={version, preprocessing,
    modalities, ...}. A window contains frames, timestamps_s, physics_steps,
    valid_masks and instantaneous_5d. Missing channels remain None. The learned
    representation has no fixed dimensionality and does not feed the force model.
    """
    if encoder is None:
        return {'vector': None, 'quality': 'not_configured', 'contract': None}
    contract = None
    try:
        steps = window['physics_steps']
        if not steps or any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError('Window timestamps must be strictly increasing')
        if steps[-1] > window['current_physics_step']:
            raise ValueError('Future samples are not a causal query')
        if any(len(window[k]) != len(steps) for k in ('timestamps_s', 'frames', 'valid_masks', 'instantaneous_5d')):
            raise ValueError('Window channels are not aligned')
        times = window['timestamps_s']
        if not np.isfinite(times).all() or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Invalid window time order')
        contract = dict(encoder.contract)
        if not all(contract.get(k) for k in ('version', 'preprocessing', 'modalities')):
            raise ValueError('Encoder needs version, preprocessing and modalities')
        result = encoder.encode(window)
        value = result.get('vector')
        if not isinstance(result.get('quality'), str):
            raise ValueError('Encoder quality must be explicit')
        if result.get('quality') != 'ok':
            return {'vector': None, 'quality': result.get('quality', 'unavailable'), 'contract': contract,
                    'provenance': result.get('provenance'), 'error_type': result.get('error_type')}
        value = np.asarray(value, float)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError('Encoder returned invalid representation')
        return {'vector': value.tolist(), 'quality': 'ok', 'contract': contract,
                'provenance': result.get('provenance')}
    except Exception as exc:
        return {'vector': None, 'quality': 'encoder_error', 'contract': contract,
                'error_type': type(exc).__name__}


def load_public_tactile_window(directory, samples, *, contract=None, episode_id=None):
    directory = Path(directory).resolve()
    if not samples:
        raise ValueError('An empty clip has no encoded representation')
    if contract is not None:
        hz = float(contract['physics_hz'])
        if not np.isfinite(hz) or hz <= 0 or any(
                not np.isclose(s['timestamp_s'], s['physics_step']/hz, rtol=0, atol=1e-9) for s in samples):
            raise ValueError('Public timestamps do not match the physical clock')
    frames, masks = [], []
    for sample in samples:
        path = (directory / sample['frame']['path']).resolve()
        if path.parent != directory or _evidence_hash(path.read_bytes()) != sample['frame']['sha256']:
            raise ValueError('Tactile frame reference failed verification')
        with np.load(path, allow_pickle=False) as data:
            # Explicit modality whitelist, never trust arbitrary archive fields.
            frames.append({k: data[k] for k in ('rgb', 'depth_m', 'reference_depth_m') if k in data})
            masks.append(data['valid_mask'] if 'valid_mask' in data else None)
    def motion(value):
        if value is None or any(v is None for v in value):
            return None
        array = np.asarray(value, float)
        return array.tolist() if array.shape == (4,) and np.isfinite(array).all() else None
    return {'frames': frames, 'valid_masks': masks,
        'episode_id': episode_id, 'sensor_sha256': (contract or {}).get('sensor'),
        'actions': [motion(s.get('action')) for s in samples],
        'actual_state': [motion([s.get('hand', {}).get(k) for k in
            ('wrist_x_offset_m', 'wrist_y_m', 'wrist_z_m', 'index_flexion_rad')]) for s in samples],
        'sources': [{k: s[k] for k in ('sample_id', 'clip_id', 'physics_step', 'timestamp_s', 'frame')} for s in samples],
        'physics_steps': [s['physics_step'] for s in samples],
        'timestamps_s': [s['timestamp_s'] for s in samples],
        'instantaneous_5d': [list(s['instantaneous_5d']) if s['instantaneous_5d'] is not None else None for s in samples],
        'current_physics_step': samples[-1]['physics_step']}


class PublicTactileEvidence:
    """Host recorder, separate from force estimation and private evaluation.

    One NPZ per actual sample permits crash recovery and future image encoders;
    JSON only carries references. An invocation is recorded even if it throws.
    Phase names describe commanded motion, not native scoring stages.
    """

    def __init__(self, root, run_dir, contract, request, *, encoder=None, retrieve_history=True,
                 failure_triggered=True):
        self.index = TactileEvidenceIndex(root)
        self.failure_triggered = bool(failure_triggered)
        self.encoders = trajectory_encoders(encoder)
        # Existing public CaP-X executions gain keys in the same versioned spaces.
        # No native labels or expert-index skill IDs are consumed here.
        self.backfill = self.index.encode_history(self.encoders) if retrieve_history and self.encoders and not self.failure_triggered else []
        self.history = self.index.records() if retrieve_history else []
        self.episode_id = Path(run_dir).name + '_' + uuid.uuid4().hex
        self.directory = self.index.root / 'episodes' / self.episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.contract, self.request, self.encoder = contract, request, encoder
        self.frames, self.clips, self.invocations, self.functions = [], [], [], {}
        self.events, self.retrievals, self.active_calls = [], [], []
        self.sequence = 0
        self.execution_id = None
        self.executions = []
        self._text_buffer = ''
        self.current_clip = None
        self.last_step = None
        self.last_action = None
        self.attempt = None
        self.releasing = False
        self.clock = {'action_step': 0, 'physics_step': 0}
        self.manifest_path = self.directory / 'manifest.json'
        self.index.append('episode', {'episode_id': self.episode_id, 'request': request,
            'contract': contract, 'manifest': str(self.manifest_path)})
        atomic_json(self.run_dir / 'piano_tactile_evidence.json', {
            'schema': 'public_tactile_evidence.v1', 'root': str(self.index.root),
            'episode_id': self.episode_id, 'manifest': str(self.manifest_path)})
        self.flush()

    def stamp(self):
        self.sequence += 1
        return {'event_seq': self.sequence, 'execution_id': self.execution_id, **self.clock}

    def write(self, text):
        """Observe public stop markers at their call site, before the stack unwinds."""
        self._text_buffer += text
        while '\n' in self._text_buffer:
            line, self._text_buffer = self._text_buffer.split('\n', 1)
            if line.startswith(('PIANO_STOP ', 'CAPX_FAILURE ')) and not any(m in line.lower() for m in _PRIVATE_MARKERS):
                self.event('public_stop', message=line[:1200])
        self._text_buffer = self._text_buffer[-1200:]
        return len(text)

    def _phase(self, attempt, action):
        if attempt != self.attempt:
            self.releasing, self.last_action = False, None
            self.attempt = attempt
        previous = self.last_action
        self.last_action = list(action)
        if previous is not None and (action[3] < previous[3] - 1.e-5 or
                (previous[3] > .05 and action[2] > previous[2] + .00005)):
            self.releasing = True
        if self.releasing:
            return 'release'
        if action[3] <= .05:
            return 'approach'
        if previous is not None and abs(action[3] - previous[3]) <= 1.e-5 and abs(action[2] - previous[2]) <= .00005:
            return 'hold'
        return 'press'

    def sample(self, *, action_step, physics_step, attempt, note, dynamics, action,
               hand, estimate, features, rgb, depth, reference, scales):
        self.instantaneous_scales = list(scales)
        if self.last_step == physics_step:
            return  # Repeated queries do not create new physical observations.
        if self.last_step is not None and physics_step < self.last_step:
            raise ValueError('Evidence physical time went backwards')
        self.last_step = int(physics_step)
        self.clock = {'action_step': int(action_step), 'physics_step': int(physics_step)}
        phase = self._phase(attempt, action)
        if self.current_clip is None or (self.current_clip['attempt'], self.current_clip['phase']) != (attempt, phase):
            self._finish_clip()
            self.current_clip = {'clip_id': 'clip:' + _evidence_hash([self.episode_id, attempt, phase, physics_step]),
                'attempt': attempt, 'note': note, 'dynamics': dynamics, 'phase': phase,
                'phase_basis': 'commanded_motion_not_native_stage', 'samples': []}
        depth = None if depth is None else np.asarray(depth)
        reference = None if reference is None else np.asarray(reference)
        valid = None
        if depth is not None and reference is not None and depth.ndim == 2 and depth.shape == reference.shape:
            valid = np.isfinite(depth) & np.isfinite(reference) & (depth > 0) & (reference > 0)
        arrays = {k: np.asarray(v) for k, v in {'rgb': rgb, 'depth_m': depth,
            'reference_depth_m': reference, 'valid_mask': valid}.items() if v is not None}
        if any(v.dtype.hasobject for v in arrays.values()):
            raise ValueError('Object arrays are not public tactile frames')
        path = self.directory / f'frame_{physics_step:09d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        vector = None
        if features is not None:
            value = np.asarray(features, float)
            if value.shape == (5,) and np.isfinite(value).all():
                vector = value.tolist()
        public_hand = {k: hand.get(k) for k in ('index_flexion_rad', 'wrist_x_offset_m', 'wrist_y_m', 'wrist_z_m')}
        force = estimate.get('estimated_key_force_N')
        if force is not None and not np.isfinite(force):
            force = None
        row = {'sample_id': 'sample:' + _evidence_hash([self.episode_id, physics_step]),
            **self.stamp(), 'timestamp_s': physics_step / self.contract['physics_hz'],
            'clip_id': self.current_clip['clip_id'], 'attempt': attempt, 'phase': phase,
            'action': list(action), 'hand': public_hand, 'instantaneous_5d': vector,
            'estimate': {'estimated_key_force_N': force, 'quality': estimate.get('quality', 'unavailable'),
                         'calibration_version': estimate.get('calibration_version')},
            'frame': {'path': path.name, 'sha256': _evidence_hash(path.read_bytes()),
                      'channels': list(arrays)}, 'invocation_ids': list(self.active_calls)}
        self.frames.append(row)
        self.current_clip['samples'].append(row)
        with (self.directory / 'samples.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        # Query once per segment using fresh instantaneous evidence. A retrieved
        # clip remains advisory, not an automatically approved parameter change.
        if not self.failure_triggered and vector is not None and estimate.get('quality') == 'ok' and not self.current_clip.get('queried'):
            self.current_clip['queried'] = True
            rep_contract = self._representation_contract(note, dynamics, phase)
            matches = self.index.retrieve('instantaneous_5d', vector, rep_contract,
                candidates=self.history, scales=scales)
            event = {'query_sample': row['sample_id'], 'type': 'instantaneous_5d', 'matches': matches}
            self.retrievals.append(event)
            self.index.append('retrieval', event, [row['clip_id'], *(m['representation_id'] for m in matches)])

    def _representation_contract(self, note, dynamics, phase):
        from capx.integrations.opentac_shadow_vision import parse_note
        return {**self.contract, 'feature_schema': 'instantaneous_5d.v1',
                'color': 'black' if parse_note(note)[1] else 'white', 'dynamics': dynamics, 'phase': phase}

    def window(self, samples):
        return load_public_tactile_window(self.directory, samples, contract=self.contract, episode_id=self.episode_id)

    def _finish_clip(self):
        clip = self.current_clip
        if not clip or not clip['samples']:
            return
        samples = clip.pop('samples')
        clip.pop('queried', None)
        clip.update(sample_ids=[r['sample_id'] for r in samples],
            start_physics_step=samples[0]['physics_step'], end_physics_step=samples[-1]['physics_step'],
            invocation_ids=sorted({i for r in samples for i in r['invocation_ids']}))
        self.clips.append(clip)
        self.index.append('clip', {**clip, 'episode_id': self.episode_id, 'manifest': str(self.manifest_path)},
            [clip['clip_id'], *clip['invocation_ids']])
        contract = self._representation_contract(clip['note'], clip['dynamics'], clip['phase'])
        # Keep every 5D sample; this additional representative is explicitly the
        # last valid sample, not a learned temporal embedding or a replacement.
        valid = [r for r in samples if r['instantaneous_5d'] is not None and r['estimate']['quality'] == 'ok']
        rep = valid[-1] if valid else samples[-1]
        self.index.put_representation(clip['clip_id'], 'instantaneous_5d', rep['instantaneous_5d'],
            contract, rep['estimate']['quality'], rep['physics_step'])
        if not self.encoders:
            self.index.put_representation(clip['clip_id'], 'temporal', None,
                {**contract, 'encoder': None, 'window': 'commanded_phase_segment.v1'},
                'not_configured', samples[-1]['physics_step'])
        for encoder in ([] if self.failure_triggered else self.encoders):
            try:
                window = _clip_window(self.directory, self.frames, clip, self.contract, encoder, self.episode_id)
                result = encode_tactile_window(window, encoder)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {'vector': None, 'quality': 'invalid_public_history', 'error_type': type(exc).__name__}
            temporal_contract = _temporal_contract(contract, encoder)
            rep_id = self.index.put_representation(clip['clip_id'], 'temporal', result['vector'], temporal_contract,
                result['quality'], samples[-1]['physics_step'], provenance=result.get('provenance'))
            if result['quality'] == 'ok':
                matches = self.index.retrieve('temporal', result['vector'], temporal_contract, candidates=self.history)
                self.retrievals.append({'query_representation': rep_id, 'type': 'temporal',
                    'encoder': dict(encoder.contract), 'matches': matches})
                self.index.append('retrieval', self.retrievals[-1], [rep_id, *(m['representation_id'] for m in matches)])
        self.current_clip = None
        self.flush()

    def event(self, kind, **fields):
        # Never serialize arbitrary generated values or native simulator info.
        if kind in ('attempt_finalized', 'release_rejected'):
            fields = {k: fields[k] for k in ('index', 'release_verified', 'continuous_release_seconds',
                'finalization_source', 'terminal_before_finalize') if k in fields}
        elif kind == 'native_terminal_end':
            fields = {'finalized_indices': list(fields.get('finalized_indices', []))}
        elif kind in ('tactile_skill_query', 'tactile_skill_applied', 'tactile_skill_recorded'):
            fields = {k: fields[k] for k in ('skill_id', 'function_name', 'function_hash', 'source_version',
                'status', 'z_correction_m', 'queried_step', 'applied_step', 'entry_features',
                'color', 'dynamics', 'hold_start_step', 'hold_end_step') if k in fields}
        elif kind == 'code_execution':
            fields = {'has_exception': bool(fields.get('has_exception')),
                'public_messages': [s[:8192] if s.startswith('PIANO_POLICY_STATE ') else s[:1200] for s in fields.get('public_messages', [])[-24:]
                    if not any(m in s.lower() for m in _PRIVATE_MARKERS)]}
        elif kind == 'recovery_read':
            if fields.get('modality') not in ('hand','tactile'):
                return
            fields = {'modality': fields['modality']}
        elif kind == 'public_stop':
            fields = {'message': fields['message'][:1200], 'invocation_ids': list(self.active_calls)}
        else:
            return
        event = {'kind': kind, **self.stamp(), **fields}
        self.events.append(event)
        links = [*self.active_calls, *([fields['skill_id']] if fields.get('skill_id') else [])]
        if kind == 'tactile_skill_recorded':
            links += [f['function_id'] for f in self.functions.values()
                      if f['ast_hash'] == fields.get('function_hash')]
            self.index.append('tactile_parameter', {'episode_id': self.episode_id, **event}, links)
        else:
            self.index.append('public_outcome', {'episode_id': self.episode_id, **event}, links)
        if kind == 'attempt_finalized':
            self._finish_clip()
        self.flush()

    @contextmanager
    def trace_execution(self, code, namespace):
        filename = '<capx-public-' + _evidence_hash(code.encode()) + '>'
        self._finish_clip()
        self.execution_id = 'execution:' + _evidence_hash([self.episode_id, len(self.executions)])
        policy = re.search(r'^# capx-piano-policy:([0-9a-f]{64})$', code, re.M)
        repair = re.search(r'^# capx-piano-repair:([0-9a-f]{64})$', code, re.M)
        execution = {'execution_id': self.execution_id, 'source_version': policy.group(1) if policy else None,
            'repair_id': repair.group(1) if repair else None, 'start': self.stamp(), 'end': None}
        self.executions.append(execution)
        if repair:
            self.events.append({'kind': 'repair_execution_begin', 'repair_id': repair.group(1),
                                'source_version': execution['source_version'], **self.stamp()})
        tree = ast.parse(code)
        by_line = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                source = ast.get_source_segment(code, node)
                function_id = 'function:' + _evidence_hash(ast.dump(node))
                by_line[(node.lineno, node.name)] = function_id
                if function_id not in self.functions:
                    self.functions[function_id] = {'function_id': function_id, 'name': node.name,
                        'source': source, 'source_hash': _evidence_hash(source.encode()),
                        'ast_hash': hashlib.sha256(ast.dump(node).encode()).hexdigest()}
                    self.index.append('function', self.functions[function_id], [function_id])
        previous = sys.gettrace()
        frames = {}
        instructions = {}
        def trace(frame, event, arg):
            if frame.f_code.co_filename != filename or frame.f_globals is not namespace:
                return None
            frame.f_trace_lines = False
            key = (frame.f_code.co_firstlineno, frame.f_code.co_name)
            if event == 'call' and key in by_line:
                identifier = 'invocation:' + _evidence_hash([self.episode_id, len(self.invocations)])
                row = {'invocation_id': identifier, 'function_id': by_line[key], 'start': self.stamp(),
                       'execution_id': self.execution_id, 'depth': len(self.active_calls),
                       'parent_invocation_id': self.active_calls[-1] if self.active_calls else None,
                       'end': None, 'exceptions': [], 'outcome': 'running'}
                frames[id(frame)] = row
                self.invocations.append(row)
                self.active_calls.append(identifier)
            row = frames.get(id(frame))
            if row is not None and event == 'exception':
                row['exceptions'].append({'type': arg[0].__name__, 'generated_line': frame.f_lineno, **self.stamp()})
            if row is not None and event == 'return':
                if frame.f_code not in instructions:
                    instructions[frame.f_code] = {i.offset: i.opname for i in dis.get_instructions(frame.f_code)}
                opcode = instructions[frame.f_code].get(frame.f_lasti, '')
                row.update(end=self.stamp(), outcome='returned' if opcode.startswith('RETURN') else 'raised')
                # Exception events are retained even when handled by the helper.
                self.index.append('invocation', row, [row['invocation_id'], row['function_id']])
                self.active_calls.remove(row['invocation_id'])
                del frames[id(frame)]
            return trace
        try:
            sys.settrace(trace)
            yield filename
        finally:
            sys.settrace(previous)
            execution['end'] = self.stamp()
            self._finish_clip()
            self.flush()

    def flush(self):
        atomic_json(self.manifest_path, {'schema': 'public_tactile_evidence.v1',
            'episode_id': self.episode_id, 'contract': self.contract, 'request': self.request,
            'failure_triggered': self.failure_triggered, 'executions': self.executions,
            'trajectory_encoders': [{'contract': e.contract,
                'load_error_type': getattr(e, 'error_type', None)} for e in self.encoders],
            'clips': self.clips, 'samples': self.frames, 'invocations': self.invocations,
            'functions': list(self.functions.values()), 'events': self.events, 'retrievals': self.retrievals})

    def close(self):
        self._finish_clip()
        self.flush()


@contextmanager
def _reflection_deadline(seconds):
    """The text-library organizer retains its independent 120-second budget."""
    with request_deadline(seconds, phase='skill_reflection'):
        yield


def library_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_library(raw: bytes) -> dict:
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("Library exceeds the fixed 8 KB budget")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or set(data) != {"schema", "skills"} or data["schema"] != SCHEMA:
        raise ValueError("Unsupported library schema")
    skills = data["skills"]
    if not isinstance(skills, list) or not 0 <= len(skills) <= MAX_SKILLS:
        raise ValueError("Library requires 0..16 skills")
    ids = set()
    for row in skills:
        if not isinstance(row, dict) or set(row) != {"id", "trigger", "practice"}:
            raise ValueError("Skill fields must be id/trigger/practice")
        if not isinstance(row["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", row["id"]) or row["id"] in ids:
            raise ValueError("Duplicate or invalid skill id")
        ids.add(row["id"])
        if not isinstance(row["trigger"], str) or not 1 <= len(row["trigger"]) <= 240:
            raise ValueError("Invalid trigger length")
        if not isinstance(row["practice"], str) or not 1 <= len(row["practice"]) <= 600:
            raise ValueError("Invalid practice length")
    text = raw.decode("utf-8").lower()
    if any(marker in text for marker in _PRIVATE_MARKERS):
        raise ValueError("Private audit markers are prohibited")
    if re.search(r"(?:ignore|disable|bypass|override|remove)\s+(?:the\s+)?(?:hard|safety|force|action|native|fixed|task)", text):
        raise ValueError("Skill attempts to override a fixed constraint")
    return data


def initialize_library(source: Path | None, destination: Path) -> None:
    """None starts with no historical entries; task constraints remain fixed."""
    data = {"schema": SCHEMA, "skills": []} if source is None else validate_library(source.read_bytes())
    for row in data["skills"]:
        if row["id"] == "visual_key_map":
            row["practice"] = ("Read get_keyboard_endpoints(). When calibration_ready is true, reuse mapped_keys without an image. "
                "Otherwise confirm the supplied A0/C8 endpoints in the clear HEAD image and pass only those two normalized "
                "white-key pixels to locate_visual_keys. Save the returned requested-key targets in piano_state['keys']. "
                "The public endpoint calibration works for new notes across compatible seeds; fresh hand/DIGIT feedback is still required.")
    raw = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
    validate_library(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)


def public_evidence(run_dir: Path) -> dict:
    episode_path = run_dir / "shadow_public_episode.json"
    if not episode_path.is_file():
        raise ValueError("No completed public reset record; retain the previous library")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    # Explicit field selection: do not pass arbitrary artifact fields onward.
    request = {k: episode[k] for k in ("seed", "sequence", "keyboard_cache_hit", "clear_action_steps")}
    source = run_dir / "shadow_public_observations.jsonl"
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines() if source.is_file() else []:
        row = json.loads(line)
        kind = row.get("kind")
        safe = {"kind": kind, "action_step": row["action_step"]}
        if kind == "force_estimate":
            hand = {k: row["hand_state"][k] for k in ("wrist_x_offset_m", "wrist_y_m", "wrist_z_m", "index_flexion_rad", "wrist_and_index_velocity")}
            safe.update(note=row["note"], target=row["target"], hand_state=hand,
                estimate={k: row["estimate"].get(k) for k in
                    ("estimated_key_force_N", "sample_step", "age_steps", "quality", "calibration_version")})
        elif kind == "action":
            safe.update(target=row["target"], hand_state=row["hand_state"], native_done=row["native_done"])
        elif kind == "attempt_finalized":
            safe["index"] = row["index"]
        elif kind == 'release_rejected':
            safe.update(index=row['index'], hand_state=row['hand_state'],
                continuous_release_seconds=row['continuous_release_seconds'])
        elif kind == "code_execution":
            safe["public_messages"] = row["public_messages"]
        else:
            continue
        safe["event_id"] = len(rows)
        rows.append(safe)
    forces = [r for r in rows if r["kind"] == "force_estimate"]
    quality = Counter(r["estimate"]["quality"] for r in forces)
    # Keep representative contact dynamics and all rare invalid states within a
    # bounded prompt. Full public telemetry remains available for reproduction.
    selected = {r["event_id"] for r in rows if r["kind"] in ("attempt_finalized", "code_execution")}
    mandatory = set(selected)
    # Preserve measured release trajectories, not only sparse press estimates.
    for event in rows:
        if event['kind'] in ('attempt_finalized', 'release_rejected'):
            mandatory.add(event['event_id'])
            prior_actions = [r for r in rows if r['kind']=='action' and r['event_id']<event['event_id']]
            mandatory.update(r['event_id'] for r in prior_actions[-8:])
    for note in dict.fromkeys(r["note"] for r in forces):
        group = [r for r in forces if r["note"] == note]
        stride = max(1, len(group) // 5)
        selected.update(r["event_id"] for r in group[::stride][:6])
        selected.update(r["event_id"] for r in group if r["estimate"]["quality"] not in ("ok", "no_contact"))
        selected.add(group[-1]["event_id"])
    selected.update(mandatory)
    keep = set(sorted(mandatory)[-128:])
    for i in sorted(selected - keep):
        if len(keep)>=128:break
        keep.add(i)
    selected_rows = [r for r in rows if r['event_id'] in keep]
    code_paths = sorted(run_dir.glob("trial_*/code.py")) + sorted(run_dir.glob("**/in_progress_code.py"))
    code = code_paths[0].read_text(encoding="utf-8", errors="replace") if code_paths else ""
    if any(marker in code.lower() for marker in _PRIVATE_MARKERS):
        code = "[Code omitted: contained a prohibited private marker.]"
    if len(code) > 16000:
        code = code[:10000] + "\n# [middle omitted]\n" + code[-6000:]
    result = {"schema": "shadow_piano_skill_public_evidence.v1", "run_id": run_dir.name,
              "request": request, "quality_counts": dict(quality), "public_events": selected_rows,
              "generated_code": code,
              "limitations": "No native score/true force is available. Public hold confidence is not native success."}
    # This verifies the whitelist output, rather than relying on prompts alone.
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if any(marker in encoded.lower() for marker in _PRIVATE_MARKERS):
        raise ValueError("Public evidence contains a prohibited field")
    return result


def validate_proposal(proposal: dict, previous: dict, evidence: dict) -> tuple[bytes, list]:
    if not isinstance(proposal, dict) or set(proposal) != {"skills", "evidence"}:
        raise ValueError("Reflection must return skills and evidence")
    raw = yaml.safe_dump({"schema": SCHEMA, "skills": proposal["skills"]}, sort_keys=False, allow_unicode=True).encode("utf-8")
    data = validate_library(raw)
    if not isinstance(proposal["evidence"], list):
        raise ValueError("Evidence must be a list")
    event_ids = {r["event_id"] for r in evidence["public_events"]}
    supported = {}
    for row in proposal["evidence"]:
        if not isinstance(row, dict) or set(row) != {"skill_id", "event_ids", "basis", "reason"}:
            raise ValueError("Invalid evidence record")
        if row["basis"] not in ("observed", "hypothesis") or not isinstance(row["event_ids"], list) or not row["event_ids"]:
            raise ValueError("Each update needs public events and an evidence basis")
        if any(isinstance(i, bool) or not isinstance(i, int) or i not in event_ids for i in row["event_ids"]):
            raise ValueError("Reflection cites an unavailable public event")
        if not isinstance(row["reason"], str) or not 1 <= len(row["reason"]) <= 500:
            raise ValueError("Invalid evidence explanation")
        supported[row["skill_id"]] = row
    old = {r["id"]: r for r in previous["skills"]}
    for row in data["skills"]:
        if old.get(row["id"]) != row and row["id"] not in supported:
            raise ValueError("Every changed skill requires traceable public evidence")
    if len({r["id"] for r in data["skills"]} & old.keys()) < min(4, len(old)):
        raise ValueError("Reflection discarded the fixed safety foundation")
    return raw, proposal["evidence"]


def reflect_once(library: Path, run_dir: Path, output: Path, query_args) -> dict:
    from capx.llm.client import collect_llm_usage, query_model, summarize_llm_usage
    output.mkdir(parents=True, exist_ok=False)
    before = library.read_bytes()
    (output / "library_input.yaml").write_bytes(before)
    previous = validate_library(before)
    events = []
    result = {"input_sha256": hashlib.sha256(before).hexdigest(), "accepted": False}
    try:
        evidence = public_evidence(run_dir)
        atomic_json(output / "public_evidence.json", evidence)
        instructions = (
            "整理钢琴公开技能库，输出一个 JSON 对象，仅有 skills 和 evidence。skills 是完整的下一版条目列表，"
            "每条仅 id/trigger/practice，保持英文便于后续代码生成，每条 trigger<=240字符、practice<=600字符，最多16条、总YAML<=8192字节。"
            "合并去重，不为更新而更新。维持现有安全基础，不修改固定动作边界、力阈值、估计器、原生判分或任务时限。"
            "可建议更好的规划、接近、等待、稳定、微调与释放，但不能读私有数据、绕过保护或声称已获官方成功。"
            "仅依据给出的公开反馈；每条新增或修改建议必须在 evidence 给出 {skill_id,event_ids:[整数],basis:'observed'或'hypothesis',reason:简短解释}。"
            "没有确证的控制改善必须标为 hypothesis，并在 practice 明确需要新触觉验证。未改条目无须 evidence。"
            "只有公开估计，没有真实力或原生得分；不能推断哪一音官方成功。端点标定跨兼容seed复用，缓存命中无需图片；"
            "get_hand_state没有native_done，终止标记来自motion返回。输出纯JSON，不要代码围栏。")
        prompt = [{"role": "system", "content": instructions},
                  {"role": "user", "content": json.dumps({"previous_library": previous, "public_evidence": evidence}, ensure_ascii=False)}]
        atomic_json(output / "reflection_prompt.json", {"messages": prompt})
        with collect_llm_usage(events, "skill_reflection"), _reflection_deadline(REFLECTION_TIMEOUT_SECONDS):
            response = query_model(query_args, prompt)
        (output / "response.txt").write_text(response["content"], encoding="utf-8")
        text = response["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        proposal = json.loads(text)
        atomic_json(output / "proposal.json", proposal)
        raw, references = validate_proposal(proposal, previous, evidence)
        atomic_json(output / "provenance.json", {"run_id": run_dir.name, "evidence": references})
        temp = library.with_suffix(".next.yaml")
        temp.write_bytes(raw)
        temp.replace(library)
        result.update(accepted=True, changed=raw != before)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    (output / "library_output.yaml").write_bytes(library.read_bytes())
    result["output_sha256"] = library_digest(library)
    result["usage"] = summarize_llm_usage(events)
    atomic_json(output / "result.json", result)
    return result
