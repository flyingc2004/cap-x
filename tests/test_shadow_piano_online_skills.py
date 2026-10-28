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


def evidence_recorder(tmp_path, name='run', encoder=None, failure_triggered=False):
    from capx.integrations.opentac_shadow_online_skills import PublicTactileEvidence
    return PublicTactileEvidence(tmp_path/'evidence', tmp_path/name,
        {'sensor':'same', 'force_model':'frozen', 'action_repeat':4, 'physics_hz':240,
         'frame':'public_endpoint_geometry.v1'},
        {'notes':['F4'], 'dynamics':['light'], 'hold_seconds':[.12]}, encoder=encoder,
        failure_triggered=failure_triggered)


def record_sample(recorder, step, flex=.5, z=.01, quality='ok'):
    recorder.sample(action_step=step, physics_step=step*4, attempt=0, note='F4', dynamics='light',
        action=[0.,0.,z,flex], hand={'index_flexion_rad':flex,'wrist_x_offset_m':0.,'wrist_y_m':0.,'wrist_z_m':z,
                                  'key_forces':[987.], 'success':True},
        estimate={'estimated_key_force_N':.35 if quality=='ok' else None,'quality':quality,
                  'calibration_version':'frozen','reward':123}, features=[.1,.2,.3,.4,.5],
        rgb=np.ones((2,3,3),np.uint8), depth=np.full((2,3),.01),
        reference=np.full((2,3),.0101), scales=[1.]*5)


def test_evidence_keeps_continuous_frames_5d_and_optional_temporal_independent(tmp_path):
    from capx.integrations.opentac_shadow_online_skills import encode_tactile_window
    rec=evidence_recorder(tmp_path)
    for step,flex,z,q in [(1,.2,.01,'ok'),(2,.5,.01,'ood'),(3,.5,.01,'ok'),(4,0,.04,'no_contact')]:
        record_sample(rec,step,flex,z,q)
    record_sample(rec,4,0,.04)  # queries cannot invent a second frame
    rec.close()
    assert [s['physics_step'] for s in rec.frames]==[4,8,12,16]
    assert [c['phase'] for c in rec.clips]==['press','hold','release']
    assert rec.frames[1]['estimate']['estimated_key_force_N'] is None
    manifest=rec.manifest_path.read_text()
    assert all(x not in manifest for x in ('key_forces','success','reward','987'))
    window=rec.window(rec.frames)
    assert len(window['frames'])==4 and all(m.all() for m in window['valid_masks'])
    assert encode_tactile_window(window)['quality']=='not_configured'
    old=[r['id'] for r in rec.index.records() if r['kind']=='representation' and r['payload']['type']=='instantaneous_5d']
    class Encoder:
        contract={'version':'fake.v1','preprocessing':'raw.v1','modalities':['depth_m']}
        def encode(self, w):
            assert w['physics_steps'][-1]<=w['current_physics_step']
            return {'vector':[float(len(w['frames']))]*7,'quality':'ok'}
    rep_id=rec.index.encode_clip(rec.clips[0]['clip_id'],Encoder())
    rep=next(r for r in rec.index.records() if r['id']==rep_id)
    assert len(rep['payload']['vector'])==7
    assert all(any(r['id']==identifier for r in rec.index.records()) for identifier in old)
    assert rec.index.retrieve('temporal',rep['payload']['vector'],rep['payload']['contract'])
    changed={**rep['payload']['contract'],'encoder':{**Encoder.contract,'version':'fake.v2'}}
    assert rec.index.retrieve('temporal',rep['payload']['vector'],changed)==[]
    assert rec.index.retrieve('instantaneous_5d',rep['payload']['vector'],rep['payload']['contract'])==[]
    bad={**window,'current_physics_step':8}
    assert encode_tactile_window(bad,Encoder())['quality']=='encoder_error'
    class Broken(Encoder):
        def encode(self,w):
            w['instantaneous_5d'][0][0]=999.
            raise RuntimeError('offline')
    assert encode_tactile_window(window,Broken())['quality']=='encoder_error'
    assert rec.frames[0]['instantaneous_5d']==[.1,.2,.3,.4,.5]
    assert rec.frames[-1]['instantaneous_5d']==[.1,.2,.3,.4,.5]


def test_invocation_traces_distinguish_retrieval_call_exception_and_duplicate_calls(tmp_path):
    rec=evidence_recorder(tmp_path)
    source='''def press():
    take_sample()
    raise ValueError('public failure')
def unused():
    pass
for i in range(2):
    try:
        press()
    except ValueError:
        pass
'''
    count=[0]
    def sample():
        count[0]+=1
        record_sample(rec,count[0])
    scope={'take_sample':sample}
    import sys
    previous=sys.gettrace()
    with rec.trace_execution(source,scope) as filename:
        exec(compile(source,filename,'exec'),scope)
    assert sys.gettrace() is previous
    assert len(rec.invocations)==2
    assert rec.invocations[0]['invocation_id']!=rec.invocations[1]['invocation_id']
    assert all(i['exceptions'][0]['type']=='ValueError' and i['end'] is not None for i in rec.invocations)
    assert all(i['outcome']=='raised' for i in rec.invocations)
    assert all(s['invocation_ids'] for s in rec.frames)
    assert any(f['name']=='unused' for f in rec.functions.values())
    assert not any(i['function_id']==f['function_id'] for i in rec.invocations for f in rec.functions.values() if f['name']=='unused')
    assert rec.index.related(rec.clips[0]['clip_id'])
    warm=evidence_recorder(tmp_path,'warm')
    record_sample(warm,1)
    assert warm.retrievals[0]['matches']
    warm.close()


def test_archived_encoder_rejects_changed_or_escaped_frame(tmp_path):
    rec=evidence_recorder(tmp_path)
    record_sample(rec,1)
    rec.close()
    frame=rec.directory/rec.frames[0]['frame']['path']
    frame.write_bytes(b'not the recorded image')
    with pytest.raises(ValueError,match='verification'):
        rec.window(rec.frames)


def test_adapter_records_every_action_without_private_force_queries_or_extra_steps(tmp_path):
    class Task(FakeShadowTask):
        @property
        def step_count(self):return self.steps*4
    fake=Task(stop_after=20)
    low=ShadowPianoLowLevelEnv(univtac_root=str(tmp_path),piano_notes=['F4'],piano_dynamics=['light'],
        task_factory=lambda _:fake,output_dir=str(tmp_path/'run'))
    low.reset()
    fake.tactile=SimpleNamespace(depth_m=np.full((2,3),.01),reference_depth_m=np.full((2,3),.0101))
    calls=[]
    def predict(depth,reference,**kw):
        calls.append(kw['physics_step'])
        kw['diagnostics_out']['features']=[.1,.2,.3,.4,.5]
        return {'quality':'ok','estimated_key_force_N':.35,'calibration_version':'frozen'}
    low._force_estimator=SimpleNamespace(uses_area_rate=True,observe_frame=lambda *a,**kw:None,
        predict=predict,sidecar={'models':{'white':{'iqr':[1.]*5}}})
    low.estimate_key_force=lambda *a:pytest.fail('Recorder must not enter private force audit')
    low._tactile_evidence=evidence_recorder(tmp_path)
    for action in ([0,0,.01,.2],[0,0,.01,.5],[0,0,.01,.5],[0,0,.04,0]):
        low.step(action)
    low._tactile_evidence.close()
    assert fake.steps==4 and calls==[4,8,12,16]
    assert len(low._tactile_evidence.frames)==4
    assert {c['phase'] for c in low._tactile_evidence.clips}=={'press','hold','release'}
    env=CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low,apis=[],prompt='Public only.'))
    env._exec_globals['move_shadow_index']=lambda:low.step([0,0,.04,0])
    result=env._exec_user_code('def release_again():\n    move_shadow_index()\nrelease_again()\n')
    assert result['ok'] and fake.steps==5
    assert len(low._tactile_evidence.invocations)==1
    assert low._tactile_evidence.invocations[0]['outcome']=='returned'


def test_temporal_history_crosses_phase_not_reset_and_resolves_actual_code(tmp_path):
    from capx.memory.trajectory.data import WINDOW
    from capx.integrations.opentac_shadow_online_skills import load_public_tactile_window
    class Encoder:
        contract={'version':'temporal.test','preprocessing':'test','modalities':['depth_m'], 'window':WINDOW}
        def encode(self,w):
            assert w['sensor_sha256']=='same'
            if w['timestamps_s'][-1]-w['timestamps_s'][0] < .5-1e-10:
                return {'quality':'insufficient_valid_history','vector':None}
            assert all(x is not None for x in w['actual_state'])
            assert all(len(x)==4 for x in w['actions'])
            return {'quality':'ok','vector':[1.,0.,0.], 'provenance':{'episode_id':w['episode_id'],'samples':w['sources']}}
    rec=evidence_recorder(tmp_path,'first',Encoder())
    scope={'take_sample':lambda i:record_sample(rec,i,flex=.2 if i<31 else .5)}
    source='def press():\n    for i in range(1,34):\n        take_sample(i)\npress()\n'
    with rec.trace_execution(source,scope) as filename:
        exec(compile(source,filename,'exec'),scope)
    rec.close()
    reps=[r['payload'] for r in rec.index.records() if r['kind']=='representation' and r['payload']['type']=='temporal']
    assert reps[0]['quality']=='insufficient_valid_history'
    assert reps[-1]['quality']=='ok'  # hold clip is only two frames, history crosses press.
    assert len({s['clip_id'] for s in reps[-1]['provenance']['samples']})>1
    linked=rec.index.resolve_skill_evidence(rec.clips[-1]['clip_id'])
    assert linked['functions'][0]['name']=='press' and linked['invocations']
    assert 'take_sample' in linked['functions'][0]['source']
    before=len(rec.index.records())
    assert rec.index.encode_history([Encoder()])==[]
    assert len(rec.index.records())==before
    nxt=evidence_recorder(tmp_path,'second',Encoder())
    for i in range(1,4): record_sample(nxt,i)
    nxt.close()
    reps=[r['payload'] for r in nxt.index.records() if r['kind']=='representation' and r['payload']['type']=='temporal'
          and r['payload']['clip_id'] in {c['clip_id'] for c in nxt.clips}]
    assert all(r['vector'] is None for r in reps)  # never borrow previous reset history.
    sample=copy.deepcopy(rec.frames[0])
    sample['hand'].pop('wrist_x_offset_m')
    sample['hand']['wrist_x_m']=None
    loaded=load_public_tactile_window(rec.directory,[sample])
    assert loaded['actual_state']==[None] and loaded['actions'][0]==sample['action']
    assert not any(k in loaded for k in ('reward','success','key_forces'))


def test_missing_checkpoint_does_not_disable_5d(tmp_path):
    rec=evidence_recorder(tmp_path,encoder={'run_dir':str(tmp_path/'missing'),'representations':['tactile']})
    record_sample(rec,1)
    rec.close()
    reps=[r['payload'] for r in rec.index.records() if r['kind']=='representation']
    assert any(r['type']=='instantaneous_5d' and r['quality']=='ok' for r in reps)
    assert any(r['type']=='temporal' and r['quality']=='encoder_unavailable' and r['vector'] is None for r in reps)


def test_temporal_encoding_and_retrieval_are_lazy_until_error(tmp_path):
    from capx.memory.trajectory.data import WINDOW
    from capx.integrations.opentac_shadow_code_memory import _repair_query
    calls=[]
    class Encoder:
        contract={'version':'test','preprocessing':'raw','modalities':['depth_m'],'window':WINDOW}
        def encode(self,w):
            calls.append(w)
            assert all(s['physics_step']<=w['current_physics_step'] for s in w['sources'])
            return {'vector':[1.,0.], 'quality':'ok' if w['timestamps_s'][-1]-w['timestamps_s'][0]>=.5 else 'insufficient_valid_history'}
    rec=evidence_recorder(tmp_path,encoder=Encoder(),failure_triggered=True)
    for i in range(1,35):record_sample(rec,i,flex=.2 if i<31 else .5)
    rec.close()
    assert not calls and not rec.retrievals
    failure={'kind':'code_exception','physics_step':136,'error_event_seq':10000}
    result=_repair_query(rec,failure)
    assert len(calls)==1 and len({s['clip_id'] for s in calls[0]['sources']})>1
    assert result['miss_reason']=='no_compatible_verified_repair' and not result['matches']
    assert len(rec.retrievals)==1

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
