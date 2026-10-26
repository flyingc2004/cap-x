"""CPU checks for random requests, endpoint reuse and public-only reflection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from capx.integrations.opentac_shadow_keyboard import make_calibration, validate_calibration, targets_from_calibration
from capx.integrations.opentac_shadow_vision import head_world_to_display_pixel
from capx.integrations.opentac_shadow_online_skills import initialize_library, public_evidence, reflect_once, validate_proposal
from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.envs import trial
from test_opentac_shadow_adapter import FakeShadowTask

ROOT = Path(__file__).resolve().parents[1]


def endpoints():
    result = {"frame":"fixed_HEAD_display_image", "mirrored":True,
              "display_image_size_px":[960,662], "white_center_spacing_m":.0235}
    for note,midi,y in (("A0",21,-25.5*.0235),("C8",108,25.5*.0235)):
        point=[.515,y,.0725]
        result[note]={"midi":midi,"world_key_top_center_m":point,
                      "display_image_center_px":head_world_to_display_pixel(point,mirrored=True)}
    return result


def choices(e):
    return {k:{"u":e[k]["display_image_center_px"][0]/960,
               "v":(e[k]["display_image_center_px"][1]-22)/640,"color":"white"} for k in ("A0","C8")}


def test_endpoint_calibration_maps_unseen_notes_and_invalidates_geometry():
    e=endpoints(); record=make_calibration(choices(e),e,"first")
    notes=["C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4"]
    mapped=targets_from_calibration(record,e,notes)
    expected_indices=[23,23.5,24,24.5,25,26,26.5,27,27.5,28,28.5,29]
    for note,index in zip(notes,expected_indices):
        assert mapped[note]["wrist_y_m"] == pytest.approx((index-25.5)*.0235-.033)
    assert mapped["F#4"]["wrist_x_offset_m"] == .07
    changed=copy.deepcopy(e); changed["mirrored"]=False
    with pytest.raises(ValueError): validate_calibration(record,changed)
    wrong=choices(e); wrong["A0"]["u"]+=.03
    with pytest.raises(ValueError,match="8 px"): make_calibration(wrong,e,"bad")


def test_explicit_task_seed_and_prompt_follow_reset(tmp_path):
    fake=FakeShadowTask()
    original=fake.reset
    def reset(*,seed=None):
        obs,info=original(seed=seed)
        rng=np.random.default_rng(seed)
        notes=[str(n) for n in rng.choice(["C4","F#4","B4"],6)]
        info["piano_sequence"]={"midi":[60]*6,"notes":notes,"dynamics":["light"]*6,"hold_seconds":[.12]*6}
        return obs,info
    fake.reset=reset
    low=ShadowPianoLowLevelEnv(univtac_root=str(tmp_path), piano_notes=None,piano_dynamics=None,
        task_seed=3300,task_factory=lambda _:fake,output_dir=str(tmp_path))
    low.endpoint_calibration=True
    low.get_keyboard_endpoints=lambda:{"calibration_ready":False,"mapped_keys":{}}
    env=CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low,apis=[],prompt="Generic request."))
    obs,_=env.reset(seed=1)
    assert fake.seed==3300
    assert len(low.piano_notes)==6
    text=obs["full_prompt"][-1]["content"][0]["text"]
    assert "CURRENT PUBLIC REQUEST" in text and low.piano_notes[0] in text
    assert "actual_force" not in text
    env.reset(seed=17)
    assert fake.seed==3300
    assert env._task_prompt.count("CURRENT PUBLIC REQUEST")==1
    low.endpoint_calibration=False
    env.close()


def test_cached_initial_request_needs_no_image(monkeypatch):
    env=SimpleNamespace(low_level_env=SimpleNamespace(visual_key_map_ready=True))
    obs={"full_prompt":[{"content":[{"type":"text","text":"cached targets"}]}]}
    monkeypatch.setattr(trial,"_capture_raw_visual_snapshot",lambda *_:pytest.fail("Image captured on cache hit"))
    result=trial._capture_initial_visual_feedback(env,obs,{"visual_feedback_context":"shadow_piano","use_visual_feedback":True},None,None)
    assert result==([],[],"cached targets",[])


def test_reflection_whitelist_and_rejected_update_preserve_library(tmp_path,monkeypatch):
    library=tmp_path/"skills.yaml"
    initialize_library(ROOT/"env_configs/opentac/shadow_piano_public_skills_60hz.v1.yaml",library)
    original=library.read_bytes()
    run=tmp_path/"run";run.mkdir()
    (run/"shadow_public_episode.json").write_text(json.dumps({"seed":3300,"sequence":{"notes":["C4"]},
        "keyboard_cache_hit":False,"clear_action_steps":220,"native_notes_completed":999}))
    (run/"shadow_public_observations.jsonl").write_text(json.dumps({"kind":"force_estimate","action_step":1,
        "note":"C4","target":[0,0,0,0],"hand_state":{"wrist_x_offset_m":0,"wrist_y_m":0,"wrist_z_m":0,
        "index_flexion_rad":0,"wrist_and_index_velocity":[0,0,0]},
        "estimate":{"estimated_key_force_N":.3,"sample_step":1,"age_steps":0,"quality":"ok","calibration_version":"frozen"},
        "private_key_top_force_N":987})+'\n')
    (run/"run.log").write_text("native score 999 should never be opened")
    evidence=public_evidence(run)
    assert "987" not in json.dumps(evidence) and "999" not in json.dumps(evidence)
    previous=yaml.safe_load(original)
    proposal={"skills":previous["skills"],"evidence":[]}
    validate_proposal(proposal,previous,evidence)
    proposal=copy.deepcopy(proposal);proposal["skills"][0]["practice"]="Different strategy"
    with pytest.raises(ValueError,match="traceable"):validate_proposal(proposal,previous,evidence)
    from capx.llm import client
    monkeypatch.setattr(client,"query_model",lambda *_: {"content":'{"skills": [], "evidence": []}'})
    result=reflect_once(library,run,tmp_path/"reflection",SimpleNamespace())
    assert not result["accepted"]
    assert library.read_bytes()==original
    assert json.loads((tmp_path/"reflection"/"proposal.json").read_text()) == {"skills": [], "evidence": []}


def test_reflection_deadline_retains_previous_library(tmp_path,monkeypatch):
    import time
    from capx.integrations import opentac_shadow_online_skills as online
    from capx.llm import client
    library=tmp_path/"skills.yaml"
    initialize_library(ROOT/"env_configs/opentac/shadow_piano_public_skills_60hz.v1.yaml",library)
    original=library.read_bytes()
    monkeypatch.setattr(online,"public_evidence",lambda _: {"public_events":[]})
    monkeypatch.setattr(online,"REFLECTION_TIMEOUT_SECONDS",.01)
    monkeypatch.setattr(client,"query_model",lambda *_: time.sleep(1))
    result=reflect_once(library,tmp_path/"run",tmp_path/"reflection",SimpleNamespace())
    assert not result["accepted"] and result["error"].startswith("LLMRequestTimeout:")
    assert 'source=local_deadline phase=skill_reflection' in result['error']
    assert library.read_bytes()==original


def test_random_config_has_no_fixed_melody_and_uses_frozen_model():
    cfg=yaml.safe_load((ROOT/"env_configs/opentac/play_piano_shadow_digit_random6_online_skills.yaml").read_text())
    low=cfg["env"]["cfg"]["low_level"]
    assert low["piano_notes"] is None and low["piano_dynamics"] is None
    assert low["piano_note_count"]==6 and low["action_repeat"]==4
    assert "CAPX_SHADOW_TASK_SEED" in low["task_seed"]
    assert "matched60hz_8episode" in low["digit_force_sidecar"]
    assert "all eight" not in cfg["env"]["cfg"]["prompt"]


def test_empty_library_bootstraps_from_public_evidence_only(tmp_path):
    from capx.integrations.opentac_shadow_online_skills import validate_library
    from capx.integrations.opentac_shadow_skills import inject_shadow_piano_public_skills
    library=tmp_path/"skills.yaml"
    initialize_library(None,library)
    previous=validate_library(library.read_bytes())
    assert previous["skills"]==[]
    env={"cfg":{"prompt":"Fixed control requirements.","apis":["ShadowPianoVisualForceApi"],
        "low_level":{"_target_":"capx.envs.simulators.opentac_shadow.ShadowPianoLowLevelEnv",
            "endpoint_calibration":True}}}
    inject_shadow_piano_public_skills(env,config_path=tmp_path/"config.yaml",library_path=library)
    assert "library starts empty" in env["cfg"]["prompt"]
    assert "delayed_force_response" not in env["cfg"]["prompt"]
    proposal={"skills":[{"id":"fresh_contact_check","trigger":"A new contact was probed.",
        "practice":"Recheck fresh public DIGIT readings before acting; verify this hypothesis."}],
        "evidence":[{"skill_id":"fresh_contact_check","event_ids":[1],"basis":"hypothesis",
            "reason":"The first public contact sample was followed by another reading."}]}
    raw,refs=validate_proposal(proposal,previous,{"public_events":[{"event_id":1}]})
    assert len(validate_library(raw)["skills"])==1 and refs[0]["event_ids"]==[1]
