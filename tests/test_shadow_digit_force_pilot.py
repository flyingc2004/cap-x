"""GPU-free checks for the independent Shadow DIGIT force pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/opentac"))
from shadow_digit_pilot_core import PilotRecorder, audit_episode, prepare_piano_config, sha256


ROOT = Path(__file__).resolve().parents[3] / "OpenTacBench"
ORIGINAL = ROOT / ".cache/piano_tactile/digit/venv/lib/python3.11/site-packages/tacto/config_digit.yml"


def test_piano_config_changes_only_range_force(tmp_path):
    source = ORIGINAL if ORIGINAL.is_file() else tmp_path / "original.yml"
    if source != ORIGINAL:
        source.write_text("sensor:\n  force:\n    range_force: [0, 100]\n    max_deformation: 0.005\n")
    dest = tmp_path / "piano.yml"
    provenance = prepare_piano_config(source, dest)
    original = yaml.safe_load(source.read_text())
    piano = yaml.safe_load(dest.read_text())
    assert original["sensor"]["force"]["range_force"] == [0, 100]
    assert piano["sensor"]["force"]["range_force"] == [0, 2]
    original["sensor"]["force"]["range_force"] = [0, 2]
    assert original == piano
    assert provenance["piano_sha256"] == sha256(dest)
    assert provenance["max_deformation_m"] == .005
    assert prepare_piano_config(source, dest) == provenance
    dest.write_text("changed")
    with pytest.raises(FileExistsError):
        prepare_piano_config(source, dest)


def test_public_private_exact_step_separation_and_audit(tmp_path):
    episode = tmp_path / "episode"
    recorder = PilotRecorder(episode, metadata={"physics_dt_s": 1 / 240})
    reference = np.full((4, 6), .03, np.float32)
    for step in range(9):
        forces = np.zeros(88, np.float32)
        forces[60 - 21] = step * .3
        angles = np.zeros(88, np.float32)
        angles[60 - 21] = step * .002
        recorder.append_private(physics_step=step, forces=forces, angles=angles,
                                requested_midi=60, score_count=0, score_armed=False,
                                score_success=False, score_failure="")
        if step in (0, 8):
            rgb = np.full((4, 6, 3), 20 + step, np.uint8)
            depth = np.full((4, 6), .03 - step * .0001, np.float32)
            recorder.append_public(physics_step=step, rgb=rgb, depth_m=depth,
                                   reference_depth_m=reference,
                                   hand_state=np.zeros(21, np.float32),
                                   key_angles_rad=angles,
                                   action_target=np.zeros(4, np.float32),
                                   requested_midi=60, phase="hover")
    recorder.close()
    with h5py.File(episode / "public.h5") as public, h5py.File(episode / "private_labels.h5") as private:
        assert public["physics_step"][:].tolist() == [0, 8]
        assert private["physics_step"][:].tolist() == list(range(9))
        assert "key_contact_force_N" not in public
        assert "score_success" not in public
        assert "key_contact_force_N" in private
        assert public["tactile/index/depth_m"].shape == (2, 4, 6)
        assert public["tactile/index/reference_depth_m"].shape == (4, 6)
    report = audit_episode(episode)
    assert report["exact_step_alignment"] is True
    assert report["frames_at_or_above_2_N"] == 1
    assert report["private_steps_any_key_at_or_above_2_N"] == 2
    assert (episode / "alignment.png").is_file()
    assert json.loads((episode / "audit.json").read_text()) == report


def test_private_gap_or_missing_exact_step_rejected(tmp_path):
    recorder = PilotRecorder(tmp_path / "gap", metadata={"physics_dt_s": 1 / 240})
    values = np.zeros(88, np.float32)
    recorder.append_private(physics_step=0, forces=values, angles=values,
                            requested_midi=60, score_count=0, score_armed=False,
                            score_success=False, score_failure="")
    with pytest.raises(ValueError, match="consecutive"):
        recorder.append_private(physics_step=2, forces=values, angles=values,
                                requested_midi=60, score_count=0, score_armed=False,
                                score_success=False, score_failure="")
    with pytest.raises(ValueError, match="exact-step"):
        recorder.append_public(physics_step=1, rgb=np.zeros((4, 6, 3), np.uint8),
                               depth_m=np.ones((4, 6), np.float32),
                               reference_depth_m=np.ones((4, 6), np.float32),
                               hand_state=np.zeros(21, np.float32),
                               key_angles_rad=values, action_target=np.zeros(4, np.float32),
                               requested_midi=60, phase="hover")
    recorder.close()
