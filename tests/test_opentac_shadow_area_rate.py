"""CPU checks for causal DIGIT history and the independent 5D smoke."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import yaml

from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
from capx.integrations.opentac_shadow import ShadowPianoVisualForceApi
from capx.integrations.opentac_shadow_force import (
    AREA_RATE_FEATURE_NAMES, AREA_RATE_HISTORY, AREA_RATE_SCHEMA, PublicDigitKeyForce,
)


def depth_with_area(fraction):
    reference = np.ones((10, 10), np.float32)
    depth = reference.copy()
    depth.flat[:round(fraction * depth.size)] -= .0001
    return depth, reference


def make_model(tmp_path, rate_upper=50):
    config = tmp_path / 'sensor.yml'
    config.write_text('force range: [0, 2]\n', encoding='utf-8')
    group = dict(median=[0] * 5, iqr=[1] * 5, coef=[0, 0, 0, 0, .05] + [0] * 5,
                 intercept=.5, feature_min=[0, 0, 0, 0, -50], feature_max=[2, 1, 1, 1, rate_upper])
    sidecar = tmp_path / 'force.json'
    sidecar.write_text(json.dumps(dict(schema=AREA_RATE_SCHEMA, feature_names=list(AREA_RATE_FEATURE_NAMES),
        history_contract=AREA_RATE_HISTORY, sensor_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        models={'white': group, 'black': group})), encoding='utf-8')
    return PublicDigitKeyForce(sidecar, config), sidecar, config


def test_causal_history_uses_physics_time_and_survives_repeated_reads(tmp_path):
    model, _, _ = make_model(tmp_path)
    model.observe_frame(*depth_with_area(.1), physics_step=0)
    early = model.predict(*depth_with_area(.3), color='white', sample_step=1, current_step=1, physics_step=4)
    assert early['quality'] == 'insufficient_history' and early['estimated_key_force_N'] is None
    diagnostics = {}
    reading = model.predict(*depth_with_area(.2), color='white', sample_step=2, current_step=2,
                            physics_step=8, diagnostics_out=diagnostics)
    assert diagnostics['features'][4] == pytest.approx(3.)  # compares tick 8 to 0, not 4
    assert diagnostics['history_physics_step'] == 0
    assert reading['estimated_key_force_N'] == pytest.approx(.65)
    assert model.predict(*depth_with_area(.2), color='white', sample_step=2, current_step=2, physics_step=8) == reading
    release = model.predict(*depth_with_area(.1), color='white', sample_step=3, current_step=3,
                            physics_step=12, diagnostics_out=diagnostics)
    assert diagnostics['features'][4] == pytest.approx(-6.)
    assert release['estimated_key_force_N'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='backwards'):
        model.observe_frame(*depth_with_area(.1), physics_step=0)
    model.reset_history()
    assert model.predict(*depth_with_area(.2), color='white', sample_step=0, current_step=0,
                         physics_step=0)['quality'] == 'insufficient_history'


def test_invalid_history_and_rate_ood_do_not_fabricate_force(tmp_path):
    model, _, _ = make_model(tmp_path, rate_upper=1)
    depth, reference = depth_with_area(.1)
    model.observe_frame(np.zeros_like(depth), reference, physics_step=0)
    diagnostics = {}
    invalid = model.predict(*depth_with_area(.2), color='white', sample_step=2, current_step=2,
                            physics_step=8, diagnostics_out=diagnostics)
    assert invalid['quality'] == 'insufficient_history'
    assert diagnostics['history_quality'] == 'invalid_depth'
    model.reset_history()
    model.observe_frame(depth, reference, physics_step=0)
    ood = model.predict(*depth_with_area(.2), color='white', sample_step=2, current_step=2,
                        physics_step=8, diagnostics_out=diagnostics)
    assert ood['quality'] == 'ood' and ood['estimated_key_force_N'] is None
    assert diagnostics['exceeded_bounds'][0]['feature'] == AREA_RATE_FEATURE_NAMES[-1]


def test_refitted_weights_report_their_version_without_changing_api(tmp_path):
    _, sidecar, config = make_model(tmp_path)
    frozen = json.loads(sidecar.read_text(encoding='utf-8'))
    frozen['calibration_version'] = 'shadow_digit_matched60hz.test.v1'
    sidecar.write_text(json.dumps(frozen), encoding='utf-8')
    model = PublicDigitKeyForce(sidecar, config)
    model.observe_frame(*depth_with_area(.1), physics_step=0)
    reading = model.predict(*depth_with_area(.2), color='white', sample_step=2,
                            current_step=2, physics_step=8)
    assert reading['calibration_version'] == frozen['calibration_version']
    assert reading['quality'] == 'ok'


def test_adapter_samples_unqueried_actions_and_reset_clears_history(tmp_path):
    _, sidecar, config = make_model(tmp_path)

    class Task:
        cfg = SimpleNamespace(physics_dt=1/240, action_repeat=4)
        action_space = gym.spaces.Box(np.array([-.01, -.32, -.006, 0], np.float32),
                                     np.array([.09, .26, .05, .5], np.float32))

        def __init__(self):
            self.step_count = 0
            self.tactile = SimpleNamespace()

        def _action_state(self):
            return np.array([0, -.09175, .03, 0], np.float32)

        def observation(self):
            depth, reference = depth_with_area({0: .1, 4: .3, 8: .2}[self.step_count])
            self.tactile.depth_m, self.tactile.reference_depth_m = depth, reference
            state = np.zeros(111, np.float32)
            state[:3] = [-.09175, .03, 0]
            return {'state': state, 'tactile': {'index': {'rgb': np.zeros((10, 10, 3), np.uint8)}}}

        def reset(self, **kwargs):
            self.step_count = 0
            return self.observation(), {}

        def step(self, action):
            self.step_count += 4
            return self.observation(), 0., False, False, {}

        def close(self):
            pass

    private = tmp_path / 'private'
    private.mkdir()
    env = ShadowPianoLowLevelEnv(univtac_root=str(tmp_path), piano_notes=['F4'], piano_dynamics=['heavy'],
        action_repeat=4, visual_key_mode=True, digit_piano_config=str(config), digit_force_sidecar=str(sidecar),
        task_factory=lambda _: Task(), private_audit_root=str(private))
    env._append_private_audit = lambda *args: None
    env._write_private_piano_timeline = lambda: None
    env._save_attempt_snapshots = lambda: None
    try:
        env.reset()
        env.command([0, -.09175, .03, 0], steps=2)  # no force query at all during these actions
        api = ShadowPianoVisualForceApi(env)
        reading = api.estimate_key_force('F4')
        assert reading['estimated_key_force_N'] == pytest.approx(.65)
        assert reading['calibration_version'] == AREA_RATE_SCHEMA
        assert set(reading) == {'estimated_key_force_N', 'sample_step', 'age_steps', 'quality', 'calibration_version'}
        assert ShadowPianoVisualForceApi(env).estimate_key_force('F4') == reading
        env.reset()
        assert api.estimate_key_force('F4')['quality'] == 'insufficient_history'
    finally:
        env.close()


def test_5d_smoke_keeps_control_and_scoring_and_matches_the_sensor(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    config_dir = root / 'env_configs/opentac'
    monkeypatch.setenv('CAPX_DIR', str(root))
    monkeypatch.setenv('CAPX_SHADOW_PRIVATE_AUDIT_ROOT', str(tmp_path / 'private'))
    base = yaml.safe_load((config_dir / 'play_piano_shadow_digit_visual_force_60hz_smoke.yaml').read_text())
    new = yaml.safe_load((config_dir / 'play_piano_shadow_digit_visual_force_5d_60hz_smoke.yaml').read_text())
    base_low, new_low = base['env']['cfg']['low_level'], new['env']['cfg']['low_level']
    assert new['env']['cfg']['prompt'] == base['env']['cfg']['prompt']
    assert new['env']['cfg']['multi_turn_prompt'] == base['env']['cfg']['multi_turn_prompt']
    portable_paths = {'digit_force_sidecar', 'digit_piano_config', 'private_audit_root'}
    assert {k: v for k, v in new_low.items() if k not in portable_paths} == {
        k: v for k, v in base_low.items() if k not in portable_paths}
    resolved = OmegaConf.to_container(OmegaConf.create(new), resolve=True)['env']['cfg']['low_level']
    assert resolved['digit_piano_config'] == str(config_dir / 'config_digit_piano_0_2.v1.yml')
    assert resolved['private_audit_root'] == str(tmp_path / 'private')
    sidecar = config_dir / 'shadow_digit_key_top_force_5d_area_rate.v1.json'
    estimator = PublicDigitKeyForce(sidecar, resolved['digit_piano_config'])
    assert estimator.uses_area_rate
    assert estimator.sidecar['frame_labels_included'] is False
    assert new_low['action_repeat'] == 4 and new_low['piano_require_dynamics']
