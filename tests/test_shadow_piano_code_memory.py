"""Executable-memory, public release and camera-only writer contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from capx.integrations import opentac_shadow_code_memory as mem
from capx.integrations.opentac_shadow_online_skills import public_evidence
from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
from capx.envs import trial

SOURCE='''def run_piano(request, piano_state):
    for i in range(piano_state['completed'], len(request['notes'])):
        print('NOTE', request['notes'][i])
        piano_state['completed'] = i + 1
'''


REGION_SOURCE='''def run_piano(request, piano_state):
    notes = request['notes']
    def command(x, y, z, flex):
        return move_shadow_index(x, y, z, flex)
    def measured_arrival(x, y, hover):
        return True, False
    def release(x, y, start_z, hover, start_flex, already_done):
        for k in range(8):
            command(x, y, hover, 0.)
        return True, False
    def finalize(i, note, quality, confidence, reason, done):
        finish_piano_note_attempt(i)
        piano_state['completed'] = i + 1
        return done
    for i in range(piano_state['completed'], len(notes)):
        note = notes[i]
        x = 0.
        y = 0.
        hover = .04
        z = .01
        flex = .5
        arrived, done = measured_arrival(x, y, hover)
        command(x, y, z, flex)
        released, done = release(x, y, z, hover, flex, done)
        done = finalize(i, note, 'ok', True, None, done)
'''

REGION_RELEASE='''def release(x, y, start_z, hover, start_flex, already_done):
    if already_done:
        return False, True
    elapsed = 0.
    ready_since = None
    for k in range(180):
        result = command(x, y, hover, 0.)
        if result.get('native_done'):
            return False, True
        elapsed += result.get('steps_executed', 0) * 4. / 240.
        hand = get_hand_state()
        ready = hand['index_flexion_rad'] <= .02 and abs(hand['wrist_z_m'] - hover) <= .002
        if not ready:
            ready_since = None
        elif ready_since is None:
            ready_since = elapsed
        elif elapsed - ready_since >= .12:
            return True, False
    print('PIANO_STOP release_timeout')
    return False, False
'''

REGION_RECOVERY='''hand = get_hand_state()
contact = estimate_key_force(note)
released, done = release(x, y, hand['wrist_z_m'], hover, hand['index_flexion_rad'], False)
if done:
    return
if not released:
    print('PIANO_STOP release_timeout')
    return
done = finalize(i, note, 'unavailable', None, 'release_recovered_prior_hold_unknown', done)
if done:
    return
continue
'''


def region_proposal(evidence):
    regions=evidence['editable_regions']
    return {'mode':'regions','base_source_sha256':regions['base_source_sha256'],
        'edits':[{'id':r['id'],'original_sha256':r['original_sha256'],
                  'source':REGION_RELEASE if r['id']=='release_body' else REGION_RECOVERY} for r in regions['regions']],
        'reason':'Eight observations have only seven physical intervals.',
        'expected_effect':'Complete release before finalization without replaying the press.',
        'evidence':{'error_event_seq':evidence['failure']['error_event_seq'],
                    'clip_ids':list({s['clip_id'] for s in evidence['samples']})}}


def test_region_repair_round_trip_and_fresh_seed(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path)
    cfg['shadow_piano_repair_scope']='release_regions_v1'
    monkeypatch.setattr(mem,'compatibility',lambda _: {'test':'regions'})
    seen=[]
    def query(args,messages,**kw):
        seen.append(kw['phase'])
        if kw['phase']=='code_memory_initial':return {'content':REGION_SOURCE}
        e=json.loads(messages[1]['content'])
        assert e['mode']=='regions' and e['current_source']==mem.validate_source(REGION_SOURCE)
        return {'content':json.dumps(region_proposal(e))}
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    _,_,_,_,info=env.step(code)
    assert info['sandbox_rc']!=0 and env.low_level_env._task.steps==9
    code=mem.repair(SimpleNamespace(),cfg,env=env)
    assert code
    _,_,_,_,info=env.step(code)
    assert info['sandbox_rc']==0,info
    assert env.low_level_env._task.steps==28
    result=mem.finish_episode(cfg)
    assert result['accepted'] and result['verification']['changed_functions_called'],result
    case=[r['payload'] for r in rec.index.records() if r['kind']=='repair_case'][-1]
    assert len(case['region_edits'])==2
    cfg2,rec2,env2=live_repair_fixture(tmp_path,'next')
    cfg2['shadow_piano_repair_scope']='release_regions_v1'
    code2,_,_=mem.initial_code(SimpleNamespace(),cfg2,{'full_prompt':[]})
    _,_,_,_,info=env2.step(code2)
    assert info['sandbox_rc']==0 and env2.low_level_env._task.steps==20
    assert not any(e['kind']=='recovery_read' for e in rec2.events)
    assert seen==['code_memory_initial','tactile_error_repair']
    assert mem.finish_episode(cfg2)['accepted']


@pytest.mark.parametrize('tamper',['shared_write','skip','full_module','wrong_hash','true_confidence','new_motion','signature'])
def test_region_rejects_out_of_scope_changes(tamper):
    source=mem.validate_source(REGION_SOURCE)
    e={'mode':'regions','editable_regions':mem._release_region_contract(source),
       'failure':{'error_event_seq':12},'samples':[]}
    p=region_proposal(e)
    if tamper=='shared_write':p['edits'][1]['source']="piano_state['completed'] = 99\n"+REGION_RECOVERY
    elif tamper=='skip':p['edits'][1]['source']='continue\n'+REGION_RECOVERY
    elif tamper=='full_module':p['source']=source
    elif tamper=='wrong_hash':p['edits'][0]['original_sha256']='0'*64
    elif tamper=='true_confidence':p['edits'][1]['source']=REGION_RECOVERY.replace("'unavailable', None", "'unavailable', True")
    elif tamper=='new_motion':p['edits'][1]['source']='move_shadow_index(0,0,0,0)\n'+REGION_RECOVERY
    else:p['edits'][0]['source']=REGION_RELEASE.replace('already_done):','already_done, extra=1):')
    with pytest.raises(ValueError):mem._validate_repair_proposal(p,e,source)


def test_region_history_offers_the_executed_local_correction(tmp_path,monkeypatch):
    test_region_repair_round_trip_and_fresh_seed(tmp_path,monkeypatch)
    cfg,rec,env=live_repair_fixture(tmp_path,'history_probe')
    p=tmp_path/'V0.py';p.write_text(REGION_SOURCE)
    cfg.update(shadow_piano_repair_scope='release_regions_v1',shadow_piano_initial_source=str(p),
               shadow_piano_initial_source_sha256=mem.digest(p.read_bytes()))
    seen=[]
    def query(args,messages,**kwargs):
        assert kwargs['phase']=='tactile_error_repair'
        e=json.loads(messages[1]['content']);seen.append(e)
        assert e['historical_repairs'] and e['historical_repairs'][0]['corrected_regions']
        assert e['historical_repairs'][0]['verification']['original_check_passed']
        assert len(json.dumps(e,ensure_ascii=False).encode())<=24*1024
        return {'content':json.dumps(region_proposal(e))}
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    env.step(code)
    repaired=mem.repair(SimpleNamespace(),cfg,env=env)
    assert repaired and len(seen)==1
    assert env.step(repaired)[-1]['sandbox_rc']==0
    assert mem.finish_episode(cfg)['accepted']


def test_region_preserves_frozen_code_and_supports_later_patch():
    source=mem.validate_source(REGION_SOURCE)
    e={'mode':'regions','editable_regions':mem._release_region_contract(source),
       'failure':{'error_event_seq':12},'samples':[]}
    assembled,changes=mem._validate_repair_proposal(region_proposal(e),e,source)
    old_release,(at,_),_=mem._release_region_layout(source)
    new_release,(start,end),_=mem._release_region_layout(assembled)
    old=source.splitlines(True);new=assembled.splitlines(True)
    assert old[:old_release.lineno-1]==new[:new_release.lineno-1]
    assert old[old_release.end_lineno:at]==new[new_release.end_lineno:start]
    assert old[at:]==new[end:]
    assert {c['qualified_name'] for c in changes}=={'run_piano','run_piano.release'}
    e['editable_regions']=mem._release_region_contract(assembled)
    with pytest.raises(ValueError,match='did not change'):mem._validate_repair_proposal(region_proposal(e),e,assembled)
    p=region_proposal(e);p['edits'][0]['source']=REGION_RELEASE.replace('range(180)','range(179)')
    again,_=mem._validate_repair_proposal(p,e,assembled)
    assert again.count(mem._RECOVERY_BEGIN)==1 and 'range(179)' in again


def test_local_repair_is_retrievable_despite_later_control_failure(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path)
    cfg['shadow_piano_repair_scope']='release_regions_v1'
    monkeypatch.setattr(mem,'compatibility',lambda _: {'test':'local-regions'})
    late_failure="    print('PIANO_NOTE', {'index':1,'quality':'ok','public_hold_confidence':False,'abort_reason':'hold_uncertain'})\n"
    source=REGION_SOURCE+late_failure
    def query(args,messages,**kw):
        if kw['phase']=='code_memory_initial':return {'content':source}
        return {'content':json.dumps(region_proposal(json.loads(messages[1]['content'])))}
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    assert env.step(code)[-1]['sandbox_rc']!=0
    repaired=mem.repair(SimpleNamespace(),cfg,env=env)
    assert env.step(repaired)[-1]['sandbox_rc']==0
    result=mem.finish_episode(cfg)
    assert not result['accepted'] and result['verification']['new_error_event_seqs']
    assert result['verification']['local_verification']['passed']
    assert not (tmp_path/'evidence'/'active_bundle.json').exists()
    case=[r['payload'] for r in rec.index.records() if r['kind']=='repair_case'][-1]
    assert case['status']=='local_repair_checked' and not case['verification']['passed']
    cfg2,rec2,env2=live_repair_fixture(tmp_path,'local-history')
    p=tmp_path/'original.py';p.write_text(source)
    cfg2.update(shadow_piano_repair_scope='release_regions_v1',shadow_piano_initial_source=str(p),
                shadow_piano_initial_source_sha256=mem.digest(p.read_bytes()))
    seen=[]
    def repair_query(args,messages,**kw):
        e=json.loads(messages[1]['content']);seen.append(e)
        h=e['historical_repairs'][0]
        assert h['qualification']=='local_repair_checked'
        assert h['local_verification']['passed'] and not h['verification']['passed']
        return {'content':json.dumps(region_proposal(e))}
    monkeypatch.setattr(mem,'_query',repair_query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg2,{'full_prompt':[]})
    env2.step(code)
    assert mem.repair(SimpleNamespace(),cfg2,env=env2) and len(seen)==1


@pytest.mark.parametrize('tamper',['missing_checkpoint','wrong_function','new_release_error'])
def test_local_repair_requires_executed_change_and_original_checkpoint(tmp_path,monkeypatch,tamper):
    test_region_repair_round_trip_and_fresh_seed(tmp_path,monkeypatch)
    cfg={'output_dir':str(tmp_path/'first')}
    session=json.loads(mem._session_path(cfg).read_text())
    manifest=json.loads(next((tmp_path/'evidence'/'episodes').glob('first*/manifest.json')).read_text())
    repair=session['repair'];source=session['source']
    if tamper=='missing_checkpoint':
        manifest['events']=[e for e in manifest['events'] if not (e['kind']=='attempt_finalized' and e['index']==0)]
    elif tamper=='wrong_function':repair['changes'][0]['new_hash']='0'*64
    else:
        checkpoint=next(e for e in manifest['events'] if e['kind']=='attempt_finalized' and e['index']==0)
        manifest['events'].append({**checkpoint,'kind':'release_rejected','event_seq':checkpoint['event_seq']-1})
    assert not mem._check_execution_suffix(manifest,source,repair)['local_verification']['passed']


@pytest.mark.parametrize('stopped',[False,True])
def test_policy_terminal_finalization_is_legal_but_not_release_verification(tmp_path,stopped):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    source=mem.validate_source(REGION_SOURCE)
    env.step(mem.wrapped(source,rec.request,{},host_progress=True))
    manifest=json.loads(rec.manifest_path.read_text())
    execution=manifest['executions'][-1]
    manifest['events']=[e for e in manifest['events'] if e['kind'] not in ('release_rejected','code_execution')]
    manifest['events'] += [
        {'kind':'attempt_finalized','index':0,'release_verified':False,'finalization_source':'policy',
         'terminal_before_finalize':stopped,'event_seq':10000,'physics_step':40},
        {'kind':'code_execution','has_exception':False,'public_messages':[],
         'event_seq':10001,'physics_step':40}]
    manifest['invocations']=[]
    check=mem._check_execution_suffix(manifest,source)
    assert check['public_chain_complete'] is stopped and check['passed'] is stopped
    assert not check['local_verification']['passed']


EVIDENCE_SOURCE='''def press(value):
    move_shadow_index(value)

def release(value):
    move_shadow_index(value)

def run_piano(request, piano_state):
    for i in range(piano_state['completed'], len(request['notes'])):
        press(0.2)
        press(0.5)
        press(0.5)
        release(0.0)
        finish_piano_note_attempt(i)
        piano_state['completed'] = i + 1
'''

REPAIR_SOURCE='''def press(index):
    move_shadow_index(0., 0., .01, .5)

def release(index):
    if index == 0:
        for k in range(9):
            move_shadow_index(0., 0., .04, 0.)
    finish_piano_note_attempt(index)

def run_piano(request, piano_state):
    if piano_state.get('recovery', {}).get('resuming'):
        get_hand_state()
        estimate_key_force(request['notes'][piano_state['completed']])
    if not piano_state.get('initialization_complete'):
        print('INIT_ONCE')
        piano_state['initialization_complete'] = True
    for i in range(piano_state['completed'], len(request['notes'])):
        press(i)
        release(i)
        piano_state['completed'] = i + 1
'''


def live_repair_fixture(tmp_path, name='first', count=2):
    from test_opentac_shadow_adapter import FakeShadowTask
    from test_shadow_piano_online_skills import evidence_recorder
    from capx.envs.tasks.base import CodeExecutionEnvBase, CodeExecEnvConfig
    class Task(FakeShadowTask):
        def __init__(self):
            self.target=np.array([0.,0.,.04,0.]);self._done=False
            super().__init__(stop_after=10000)
            self.tactile=SimpleNamespace(depth_m=np.full((2,3),.01),reference_depth_m=np.full((2,3),.0101))
        @property
        def step_count(self):return self.steps*4
        def _action_state(self):return self.target
        def _observation(self):
            obs=super()._observation();obs['state'][:3]=self.target[1:]
            return obs
        def step(self,action):
            self.target=np.asarray(action);return super().step(action)
        def finish_piano_note_attempt(self,index):return {}  # no scoring used by memory
    task=Task()
    low=ShadowPianoLowLevelEnv(univtac_root=str(tmp_path),piano_notes=['F4']*count,piano_dynamics=['light']*count,
        task_factory=lambda _:task,output_dir=str(tmp_path/name),action_repeat=4)
    low.reset();low.visual_key_mode=True;low.piano_continue_on_note_failure=True;low.require_public_release=True
    low._visual_key_map_ready=True;low._cached_key_targets={'F4':{'wrist_x_offset_m':0.}}
    for attr in ('_write_native_event','_write_private_piano_timeline','_write_private_force_audit','_save_attempt_snapshots','_save_tactile_snapshot'):
        setattr(low,attr,lambda *a,**kw:None)
    def predict(*args,**kw):
        kw['diagnostics_out']['features']=[.1,.2,.3,.4,.5]
        return {'estimated_key_force_N':.35,'quality':'ok','calibration_version':'frozen'}
    low._force_estimator=SimpleNamespace(uses_area_rate=True,observe_frame=lambda *a,**kw:None,predict=predict,
        sidecar={'models':{'white':{'iqr':[1.]*5}}})
    rec=evidence_recorder(tmp_path,name,failure_triggered=True)
    rec.request={'notes':['F4']*count,'dynamics':['light']*count,'hold_seconds':[.12]*count}
    rec.flush();low._tactile_evidence=rec
    def move(x,y,z,flex):
        low.step(np.array([x,y,z,flex]));return {'native_done':False,'steps_executed':1}
    def hand():
        state=low.get_observation()['hand_state'];low.note_public_recovery_read('hand');return state
    def force(note):
        low.note_public_recovery_read('tactile',note)
        return {'quality':'ok','estimated_key_force_N':.35}
    api=SimpleNamespace(functions=lambda:{'move_shadow_index':move,'finish_piano_note_attempt':low.finish_piano_note_attempt,
        'get_hand_state':hand,'estimate_key_force':force})
    env=CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low,apis=[],prompt='Public',generated_code_mode='host_module'))
    env._apis={'test':api}
    cfg={'shadow_piano_evidence_memory':str(tmp_path/'evidence'),'shadow_piano_code_memory':str(tmp_path/'code'),
        'output_dir':str(tmp_path/name),'shadow_piano_evidence_trigger':'repair_only',
        '_piano_public_request':rec.request,'_piano_key_map_ready':True}
    return cfg,rec,env


def repair_proposal(messages, *, fail_again=False):
    e=json.loads(messages[1]['content'])
    source=('def release(index):\n    finish_piano_note_attempt(index)\n' if fail_again else
        'def release(index):\n    get_hand_state()\n    for k in range(9):\n        move_shadow_index(0., 0., .04, 0.)\n    finish_piano_note_attempt(index)\n')
    return {'content':json.dumps({'mode':'functions','functions':[{'qualified_name':'release','source':source}],
        'source':None,'reason':'Release failed while the measured index was curled.',
        'expected_effect':'Inspect pose and extend at hover before retrying the public release guard.',
        'evidence':{'error_event_seq':e['failure']['error_event_seq'],
                    'clip_ids':list({s['clip_id'] for s in e['samples']})}})}


def test_pinned_historical_policy_is_not_a_memory_hit(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path)
    monkeypatch.setattr(mem,'compatibility',lambda _: {'test':'fixed'})
    source_path=tmp_path/'historical.py';source_path.write_text(REPAIR_SOURCE,encoding='utf-8')
    cfg.update(shadow_piano_initial_source=str(source_path),
        shadow_piano_initial_source_sha256=mem.digest(source_path.read_bytes()))
    def forbidden(*args,**kwargs):raise AssertionError('Pinned V0 must not read active code or call API')
    monkeypatch.setattr(mem,'_query',forbidden)
    monkeypatch.setattr(mem,'linked_initial_record',forbidden)
    code,_,error=mem.initial_code(None,cfg,{'full_prompt':[]})
    session=json.loads(mem._session_path(cfg).read_text())
    assert error is None and 'run_piano(' in code
    assert session['source']==mem.validate_source(REPAIR_SOURCE)
    assert not session['cache_hit'] and session['evidence_input_bundle'] is None
    assert session['parameters']=={} and session['initial_source']['initial_generation_calls']==0
    assert not (tmp_path/'evidence'/'active_bundle.json').exists()
    cfg['shadow_piano_initial_source_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):mem.initial_code(None,cfg,{'full_prompt':[]})


def test_error_only_repair_miss_resume_and_suffix_promotion(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path)
    monkeypatch.setattr(mem,'compatibility',lambda _: {'test':'fixed'})
    requests=[]
    def query(args,messages,**kw):
        requests.append(kw['phase'])
        if kw['phase']=='code_memory_initial':return {'content':REPAIR_SOURCE}
        assert kw['deadline_seconds']==300 and args.max_tokens==8192
        assert 'HTTPS_PROXY' not in mem.os.environ
        e=json.loads(messages[1]['content'])
        assert e['recovery']['current_attempt']==1
        assert e['retrieval_status']=='no_compatible_verified_repair'
        assert not e['historical_repairs'] and e['current_source'] is None
        assert not any(k in json.dumps(e) for k in ('key_forces','native_notes_completed','reward'))
        return repair_proposal(messages)
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    _,_,_,_,info=env.step(code)
    assert info['sandbox_rc']!=0
    assert [e['index'] for e in rec.events if e['kind']=='attempt_finalized']==[0]
    assert not rec.retrievals
    old_physics=rec.clock['physics_step']
    regenerated=mem.repair(SimpleNamespace(),cfg,env=env)
    assert regenerated and rec.clock['physics_step']==old_physics
    env._exec_globals['piano_state']['completed']=99  # generated counters are not authority
    env._exec_globals['obsolete_helper']=lambda:None
    _,_,_,_,info=env.step(regenerated)
    assert info['sandbox_rc']==0,info
    assert 'INIT_ONCE' not in info['stdout'] and 'obsolete_helper' not in env._exec_globals
    assert env.low_level_env._policy_end_submitted
    assert [e['index'] for e in rec.events if e['kind']=='attempt_finalized']==[0,1]
    assert any(e['kind']=='release_rejected' for e in rec.events)
    result=mem.finish_episode(cfg)
    assert result['status']=='promoted',result
    check=result['verification']
    assert check['changed_functions_called'] and check['original_check_passed'] and not check['new_error_event_seqs']
    assert check['boundary_physics_step']==old_physics
    assert any(e['kind']=='release_rejected' and e['event_seq']<check['boundary_event_seq'] for e in rec.events)
    assert requests==['code_memory_initial','tactile_error_repair']
    assert mem.repair(SimpleNamespace(),cfg,env=env) is None
    assert len([r for r in rec.index.records() if r['kind']=='repair_case'])==1
    cfg2,rec2,env2=live_repair_fixture(tmp_path,'next')
    code,_,_=mem.initial_code(SimpleNamespace(),cfg2,{'full_prompt':[]})
    assert len(requests)==2  # compatible active code executes without initial generation
    env2.step(code)
    result2=mem.finish_episode(cfg2)
    assert result2['status']=='promoted' and not rec2.retrievals
    assert len(requests)==2


def test_failed_repair_never_replaces_active_and_history_hit(tmp_path,monkeypatch):
    # Seed a verified correction with the complete public execution test first.
    test_error_only_repair_miss_resume_and_suffix_promotion(tmp_path,monkeypatch)
    active=(tmp_path/'evidence/active_bundle.json').read_bytes()
    cfg,rec,env=live_repair_fixture(tmp_path,'failed')
    source=mem.validate_source(REPAIR_SOURCE)
    mem.atomic_json(mem._session_path(cfg),{'source':source,'executed_version':mem.digest(source),
        'compatibility':{'test':'fixed'},'request':rec.request,'parameters':{},'repair_calls':0,'control_instructions':''})
    env.step(mem.wrapped(source,rec.request,{},host_progress=True))
    seen=[]
    def query(args,messages,**kw):
        e=json.loads(messages[1]['content']);seen.append(e)
        assert e['historical_repairs'] and e['historical_repairs'][0]['corrected_functions']
        return repair_proposal(messages,fail_again=True)
    monkeypatch.setattr(mem,'_query',query)
    code=mem.repair(SimpleNamespace(),cfg,env=env)
    assert code and len(seen)==1
    env.step(code)
    result=mem.finish_episode(cfg)
    assert result['status']=='candidate_not_promoted'
    assert result['verification']['new_error_event_seqs']
    assert (tmp_path/'evidence/active_bundle.json').read_bytes()==active
    assert [r for r in rec.index.records() if r['kind']=='repair_case'][-1]['payload']['status']=='failed_or_unverified'


@pytest.mark.parametrize('outcome',['fixed','api_failure','invalid_proposal'])
def test_no_window_static_error_still_gets_one_repair_and_is_archived(tmp_path,monkeypatch,outcome):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    monkeypatch.setattr(mem,'compatibility',lambda _: {'test':'fixed'})
    calls=[]
    def query(args,messages,**kw):
        calls.append(kw['phase'])
        if kw['phase']=='code_memory_initial':return {'content':'def run_piano(:\n'}
        e=json.loads(messages[1]['content'])
        assert e['mode']=='module' and not e['samples'] and e['retrieval_status']=='no_public_window'
        if outcome=='api_failure':raise TimeoutError('test provider')
        if outcome=='invalid_proposal':return {'content':'not JSON'}
        return {'content':json.dumps({'mode':'module','source':REPAIR_SOURCE,'functions':[],
            'reason':'Repair the invalid Python signature.','expected_effect':'Enter and execute the policy.',
            'evidence':{'error_event_seq':e['failure']['error_event_seq'],'clip_ids':[]}})}
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    env.step(code)
    assert not rec.frames
    code=mem.repair(SimpleNamespace(),cfg,env=env)
    if outcome=='fixed':
        assert code
        env.step(code)
    else:assert code is None
    assert mem.repair(SimpleNamespace(),cfg,env=env) is None
    result=mem.finish_episode(cfg)
    assert result['accepted']==(outcome=='fixed'),result
    cases=[r for r in rec.index.records() if r['kind']=='repair_case']
    assert len(cases)==1 and cases[0]['payload']['query_clip_id'] is None
    assert (tmp_path/'evidence/active_bundle.json').exists()==(outcome=='fixed')
    assert calls==['code_memory_initial','tactile_error_repair']


def test_stop_marker_keeps_actual_call_context_without_inventing_exception(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    source='''def probe():
    move_shadow_index(0.,0.,.01,.5)
    print('PIANO_STOP OOD_REPROBE')
def run_piano(request,piano_state):
    probe()
'''
    env.step(mem.wrapped(source,rec.request,{},host_progress=True))
    failure=mem.public_failure_context(json.loads(rec.manifest_path.read_text()))
    assert failure['function_name']=='probe' and failure['kind']=='public_stop'
    assert failure['exception'] is None
    assert not rec.retrievals


def test_reentry_cannot_move_before_actual_public_pose_and_contact_reads(tmp_path,monkeypatch):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    source='def run_piano(request,piano_state):\n    print("PIANO_STOP OOD_REPROBE")\n'
    env.step(mem.wrapped(source,rec.request,{},host_progress=True))
    before=env.low_level_env._task.step_count
    unchecked='def run_piano(request,piano_state):\n    move_shadow_index(0.,0.,.01,.5)\n'
    _,_,_,_,info=env.step(mem.wrapped(unchecked,rec.request,{},host_progress=True,repair_id='a'*64))
    assert info['sandbox_rc']!=0 and 'Recovery requires fresh' in info['stderr']
    assert env.low_level_env._task.step_count==before
    # An already stopped native environment must not call/retrieve for recovery.
    env.low_level_env._native_done=True
    mem.atomic_json(mem._session_path(cfg),{'source':source,'executed_version':mem.digest(source),
        'compatibility':{},'request':rec.request,'parameters':{},'repair_calls':0})
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:pytest.fail('No repair after native termination'))
    assert mem.repair(SimpleNamespace(),cfg,env=env) is None
    assert not rec.retrievals


def test_clean_checkpoint_or_unfinished_code_is_not_a_repair_trigger():
    config={'shadow_piano_evidence_memory':'unused','shadow_piano_evidence_trigger':'repair_only',
            'visual_checkpoint_markers':['finger_alignment'],'shadow_piano_recover_on_stop':True}
    for stdout in ('','CAPX_VISUAL_CHECKPOINT finger_alignment'):
        assert not trial._should_query_multiturn_after_block({'stdout':stdout,'stderr':'','sandbox_rc':0,'task_completed':False},
            code_block_idx=1,total_code_blocks=1,config=config)
    assert trial._should_query_multiturn_after_block({'stdout':'PIANO_STOP OOD_REPROBE','stderr':'','sandbox_rc':0},
        code_block_idx=1,total_code_blocks=1,config=config)


def test_host_metadata_preserves_valid_parameters_and_cannot_be_forged(tmp_path):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    parameters={'white_light':{'z_trim_m':-.0001}}
    source='def run_piano(request,piano_state):\n    return\n'
    env.step(mem.wrapped(source,rec.request,parameters,host_progress=True))
    assert env._exec_globals['piano_state']['policy_parameters']==parameters
    with pytest.raises(ValueError,match='metadata'):
        mem.validate_source('# capx-piano-repair:'+'a'*64+'\n'+source)


def test_pending_or_unchecked_bundle_never_becomes_default(tmp_path):
    cfg={'shadow_piano_evidence_memory':str(tmp_path),'shadow_piano_evidence_trigger':'repair_only'}
    bundle=mem._store_linked_bundle(tmp_path,mem.validate_source(REPAIR_SOURCE),{},
        {'schema':'shadow_piano_public_skills.v1','skills':[]},{},None,[],[])
    mem.atomic_json(tmp_path/'pending_bundle.json',{'bundle_id':bundle,'status':'syntax_checked_pending_execution'})
    assert mem.linked_initial_record(cfg,{})[0] is None
    mem.atomic_json(tmp_path/'active_bundle.json',{'bundle_id':bundle})
    assert mem.linked_initial_record(cfg,{})[0] is None
    mem.atomic_json(tmp_path/'active_bundle.json',{'bundle_id':bundle,'status':'public_execution_checked'})
    assert mem.linked_initial_record(cfg,{})[0]['bundle_id']==bundle


@pytest.mark.parametrize('confidence',[False,True])
def test_repaired_stop_requires_renewed_public_check_and_accepts_existing_dict_logs(tmp_path,confidence):
    cfg,rec,env=live_repair_fixture(tmp_path,count=1)
    source=mem.validate_source(REPAIR_SOURCE +
        "    print('PIANO_NOTE', {'index':0,'quality':'ok','public_hold_confidence':"+repr(confidence)+",'abort_reason':None})\n")
    env.step(mem.wrapped(source,rec.request,{},host_progress=True,repair_id='b'*64))
    repair={'repair_id':'b'*64,'changes':[], 'failure':{'kind':'public_stop','attempt':0,
            'function_name':'run_piano','messages':['PIANO_STOP OOD_REPROBE']}}
    result=mem._check_execution_suffix(json.loads(rec.manifest_path.read_text()),source,repair)
    assert result['original_check_passed']==confidence
    assert result['passed']==confidence


def run_public_evidence_fixture(tmp_path, name, source=EVIDENCE_SOURCE, config=None):
    from test_shadow_piano_online_skills import evidence_recorder, record_sample
    rec=evidence_recorder(tmp_path,name)
    cfg=config or {'shadow_piano_evidence_memory':str(tmp_path/'evidence'),
        'shadow_piano_code_memory':str(tmp_path/'old_code'),'output_dir':str(tmp_path/name)}
    cfg['output_dir']=str(tmp_path/name)
    cfg.setdefault('shadow_piano_evidence_trigger','episode_end')  # archived updater compatibility tests
    run=Path(cfg['output_dir'])
    request={'notes':['F4'],'dynamics':['light'],'hold_seconds':[.12]}
    (run/'shadow_public_episode.json').write_text(json.dumps({'sequence':request,'clear_action_steps':0}))
    stream=run/'shadow_public_observations.jsonl'
    stream.write_text('')
    count=[0]
    def emit(kind,**fields):
        row={'kind':kind,'action_step':count[0],'physics_step':count[0]*4,**fields}
        with stream.open('a') as f:f.write(json.dumps(row)+'\n')
        rec.event(kind,**fields)
    def motion(flex):
        count[0]+=1
        record_sample(rec,count[0],flex,.04 if flex==0 else .01)
        emit('action',target=[0,0,.04 if flex==0 else .01,flex])
    scope={'move_shadow_index':motion,
           'finish_piano_note_attempt':lambda i:emit('attempt_finalized',index=i,release_verified=True)}
    code=mem.wrapped(source,request,{})
    with rec.trace_execution(code,scope) as filename:
        exec(compile(code,filename,'exec'),scope)
    emit('code_execution',public_messages=[],has_exception=False)
    rec.close()
    mem.atomic_json(run/'piano_code_memory_session.json',{'source':mem.validate_source(source),
        'compatibility':{'fixed':'signature'},'executed_version':mem.digest(source),
        'evidence_input_bundle':cfg.get('_linked_input_bundle')})
    # The updater must never open these or copy their fields into its input.
    (run/'shadow_piano_native_audit_summary.json').write_text('{"success":true,"key_forces":[98765]}')
    return cfg,rec


def linked_proposal(evidence):
    f=next(f for f in evidence['functions'] if f['name']=='press')
    clip=evidence['clips'][0]
    refs=[{'clip_id':clip['clip_id'],'physics_steps':[clip['samples'][0]['physics_step']]}]
    return {'functions':[{'qualified_name':f['qualified_name'],
        'source':'def press(value):\n    if value > 0:\n        move_shadow_index(value)\n',
        'evidence':refs,'basis':'hypothesis','reason':'Separate positive pressing from release motion.',
        'expected_effect':'Keep press commands associated with contact samples.'}],
        'rules':[{'id':'positive_press','trigger':'Entering the press helper',
                  'practice':'Test this hypothesis with fresh contact: press only for positive flexion.'}],
        'rule_evidence':[{'rule_id':'positive_press','evidence':refs,'basis':'hypothesis',
                         'reason':'Public contact samples belong to positive flexion.',
                         'expected_effect':'Keep release separate from press.'}]}


def test_joint_loop_records_updates_and_executes_candidate_without_initial_api(tmp_path,monkeypatch):
    cfg,rec=run_public_evidence_fixture(tmp_path,'first')
    calls=[]
    import os
    monkeypatch.setenv('HTTPS_PROXY','http://must-not-be-used.invalid')
    def query(args,messages,**kwargs):
        assert os.environ['CAPX_DISABLE_PROXY']=='1' and 'HTTPS_PROXY' not in os.environ
        assert kwargs['deadline_seconds']==300
        evidence=json.loads(messages[1]['content'])
        assert len(messages[1]['content'].encode())<=24*1024
        assert '98765' not in json.dumps(evidence) and 'key_forces' not in json.dumps(evidence)
        assert 'generated_code' not in evidence and 'current_source' not in evidence
        calls.append(evidence)
        return {'content':json.dumps(linked_proposal(evidence))}
    monkeypatch.setattr(mem,'_query',query)
    result=mem.finish_linked_episode(cfg,SimpleNamespace())
    assert result['accepted'] and result['logic_change_proposed'],result
    assert result['public_execution_checked']
    assert os.environ['HTTPS_PROXY']=='http://must-not-be-used.invalid'
    assert mem.finish_linked_episode(cfg,SimpleNamespace())==result and len(calls)==1
    root=Path(cfg['shadow_piano_evidence_memory'])
    active_before=(root/'active_bundle.json').read_bytes()
    assert json.loads(active_before)['bundle_id']!=result['candidate_bundle']
    next_cfg={**cfg,'output_dir':str(tmp_path/'second'),'_piano_public_request':rec.request,
              '_piano_key_map_ready':True}
    monkeypatch.setattr(mem,'compatibility',lambda _: {'fixed':'signature'})
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:pytest.fail('Cache hit must not call initial API'))
    code,_,_=mem.initial_code(SimpleNamespace(),next_cfg,{'full_prompt':[]})
    assert 'if value > 0:' in code
    session=json.loads(mem._session_path(next_cfg).read_text())
    assert session['evidence_input_bundle']==result['candidate_bundle']
    next_cfg,rec2=run_public_evidence_fixture(tmp_path,'second',session['source'],next_cfg)
    monkeypatch.setattr(mem,'_query',lambda args,messages,**kwargs:{'content':json.dumps({
        'functions':[],'rules':json.loads(messages[1]['content'])['rules']['skills'],'rule_evidence':[]})})
    second=mem.finish_linked_episode(next_cfg,SimpleNamespace())
    assert second['status']=='no_change' and second['public_execution_checked'],second
    assert json.loads((root/'active_bundle.json').read_text())['bundle_id']==result['candidate_bundle']
    applications=[r for r in rec2.index.records() if r['kind']=='application']
    assert applications[-1]['payload']['changed_functions_called']
    assert applications[-1]['payload']['bundle_id']==result['candidate_bundle']
    assert rec2.index.related(result['candidate_bundle'])
    second_input=json.loads((Path(next_cfg['output_dir'])/'tactile_evidence_update'/'public_input.json').read_text())
    assert second_input['historical_clips']
    assert second_input['historical_clips'][0]['association_ids']


@pytest.mark.parametrize('failure',['timeout','bad_reference','bad_signature','oversize','unknown_global'])
def test_joint_update_failure_preserves_previous_valid_bundle(tmp_path,monkeypatch,failure):
    cfg,rec=run_public_evidence_fixture(tmp_path,'first')
    rules={'schema':'shadow_piano_public_skills.v1','skills':[]}
    root=Path(cfg['shadow_piano_evidence_memory'])
    identifier=mem._store_linked_bundle(root,mem.validate_source(EVIDENCE_SOURCE),{'fixed':'signature'},rules,{},None,[],[])
    mem.atomic_json(root/'active_bundle.json',{'bundle_id':identifier})
    cfg['_linked_input_bundle']=identifier
    def query(args,messages,**kw):
        if failure=='timeout':raise TimeoutError('test deadline')
        proposal=linked_proposal(json.loads(messages[1]['content']))
        if failure=='bad_reference':proposal['functions'][0]['evidence'][0]['physics_steps']=[999999]
        if failure=='bad_signature':proposal['functions'][0]['source']='def press(value, extra):\n    return 0\n'
        if failure=='oversize':proposal['functions']*=3
        if failure=='unknown_global':proposal['functions'][0]['source']='def press(value):\n    made_up_api(value)\n'
        return {'content':json.dumps(proposal)}
    monkeypatch.setattr(mem,'_query',query)
    result=mem.finish_linked_episode(cfg,SimpleNamespace())
    assert not result['accepted'],result
    assert json.loads((root/'active_bundle.json').read_text())['bundle_id']==identifier
    assert not (root/'pending_bundle.json').exists()
    assert (Path(cfg['output_dir'])/'tactile_evidence_update'/'validation.json').exists()


def test_joint_patch_cannot_change_frozen_force_or_release_parameters(tmp_path):
    cfg,rec=run_public_evidence_fixture(tmp_path,'first')
    source=EVIDENCE_SOURCE.replace('def press(value):','def press(value):\n    target_force_N = 0.35')
    manifest=json.loads(rec.manifest_path.read_text())
    evidence=mem._linked_update_input(source,manifest,{'schema':'shadow_piano_public_skills.v1','skills':[]},{})
    proposal=linked_proposal(evidence)
    proposal['functions'][0]['source']='def press(value):\n    target_force_N = 0.50\n    move_shadow_index(value)\n'
    with pytest.raises(ValueError,match='constraints'):
        mem.validate_linked_proposal(proposal,source,evidence)


def test_updater_selects_helpers_with_clip_evidence_not_pure_predicates(tmp_path):
    source=EVIDENCE_SOURCE.replace('def press(value):', 'def _release_pose():\n    return True\n\ndef hold():\n    return True\n\ndef press(value):\n    _release_pose()\n    hold()')
    _,rec=run_public_evidence_fixture(tmp_path,'linked',source)
    evidence=mem._linked_update_input(source,json.loads(rec.manifest_path.read_text()),{}, {})
    assert {f['name'] for f in evidence['functions']}=={'press','release'}


def test_linked_config_and_plain_runner_wiring(tmp_path,monkeypatch):
    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config
    monkeypatch.setenv('CAPX_SHADOW_EVIDENCE_ROOT',str(tmp_path/'evidence'))
    factory,cfg,_=_load_config(LaunchArgs(config_path='env_configs/opentac/play_piano_shadow_digit_evidence_memory.yaml'))
    assert cfg['shadow_piano_evidence_memory']==str(tmp_path/'evidence')
    assert factory['cfg']['low_level']['evidence_memory_root']==str(tmp_path/'evidence')
    assert factory['cfg']['low_level']['action_repeat']==4
    assert mem.memory_policy(cfg)['arm']=='tactile_evidence_joint'
    assert mem.repair(SimpleNamespace(),cfg) is None
    assert not mem.action_memory_enabled(cfg)
    assert cfg['max_regenerations']==1
    assert cfg['shadow_piano_evidence_trigger']=='repair_only'
    assert cfg['shadow_piano_repair_scope']=='release_regions_v1'


@pytest.mark.parametrize('mode,text,history,updates',[
    ('none',False,False,False),('text',True,False,True),
    ('tactile',False,True,True),('joint',True,True,True)])
def test_four_arm_policy_reaches_runtime(tmp_path,monkeypatch,mode,text,history,updates):
    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config
    monkeypatch.setenv('CAPX_SHADOW_EVIDENCE_ROOT',str(tmp_path/'memory'))
    monkeypatch.setenv('CAPX_SHADOW_EVIDENCE_MODE',mode)
    monkeypatch.setenv('CAPX_SHADOW_KEYBOARD_CACHE',str(tmp_path/'one_seed_keyboard.json'))
    factory,cfg,_=_load_config(LaunchArgs(config_path='env_configs/opentac/play_piano_shadow_digit_evidence_memory.yaml'))
    assert mem.evidence_policy(cfg)=={'mode':mode,'text_memory':text,'tactile_history':history,'update':updates}
    assert factory['cfg']['low_level']['retrieve_tactile_history'] is history
    assert factory['cfg']['low_level']['keyboard_calibration_path']==str(tmp_path/'one_seed_keyboard.json')


def test_no_memory_never_reads_previous_code_or_calls_episode_updater(tmp_path,monkeypatch):
    cfg,rec=run_public_evidence_fixture(tmp_path,'none')
    cfg['shadow_piano_evidence_mode']='none'
    cfg['_piano_public_request']=rec.request
    monkeypatch.setattr(mem,'compatibility',lambda _: {})
    monkeypatch.setattr(mem,'linked_initial_record',lambda *a:pytest.fail('No-memory arm read code cache'))
    calls=[]
    monkeypatch.setattr(mem,'_query',lambda *a,**kw: calls.append(1) or {'content':EVIDENCE_SOURCE})
    for name in ('seed0','seed1'):
        cfg['output_dir']=str(tmp_path/name)
        mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
        result=mem.finish_linked_episode(cfg,SimpleNamespace())
        assert result['status']=='disabled_no_memory'
    assert len(calls)==2
    assert not (tmp_path/'evidence'/'active_bundle.json').exists()


def test_api_queue_wait_is_separate_and_lock_releases(tmp_path,monkeypatch):
    import fcntl,threading,time
    from capx.llm import client
    from contextlib import contextmanager
    lock_path=tmp_path/'api.lock';lock=lock_path.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    monkeypatch.setenv('CAPX_LLM_SERIAL_LOCK',str(lock_path))
    monkeypatch.setenv('CAPX_OUTPUT_DIR',str(tmp_path))
    deadlines=[]
    @contextmanager
    def deadline(seconds,*,phase):
        deadlines.append(seconds);yield
    monkeypatch.setattr(mem,'request_deadline',deadline)
    monkeypatch.setattr(client,'query_model',lambda *args:{'content':'ok'})
    timer=threading.Timer(.1,lambda:fcntl.flock(lock,fcntl.LOCK_UN));timer.start()
    try:
        assert mem._query(SimpleNamespace(),[],deadline_seconds=300)['content']=='ok'
    finally:
        timer.join();lock.close()
    row=json.loads((tmp_path/'piano_api_queue.jsonl').read_text())
    assert row['queue_seconds']>=.1 and deadlines==[300]
    with lock_path.open('a') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('mode', ['text','tactile','joint'])
def test_updater_arm_input_and_output_isolation(tmp_path,monkeypatch,mode):
    run_public_evidence_fixture(tmp_path,'prior')
    cfg,rec=run_public_evidence_fixture(tmp_path,'current')
    cfg['shadow_piano_evidence_mode']=mode
    seen=[]
    def query(args,messages,**kw):
        e=json.loads(messages[1]['content']);seen.append(e)
        if mode=='text':assert not e['historical_clips']
        else:assert e['historical_clips']
        proposal=linked_proposal(e)
        if mode=='tactile':
            proposal['rules']=[];proposal['rule_evidence']=[]
        return {'content':json.dumps(proposal)}
    monkeypatch.setattr(mem,'_query',query)
    result=mem.finish_linked_episode(cfg,SimpleNamespace())
    assert result['accepted'],result
    assert len(seen)==1
    bundle=mem._linked_bundle(Path(cfg['shadow_piano_evidence_memory']),result['candidate_bundle'])
    assert bool(bundle['rules']['skills'])==(mode!='tactile')
    if mode=='tactile':
        with pytest.raises(ValueError,match='text experience'):
            mem.validate_linked_proposal(linked_proposal(seen[0]),EVIDENCE_SOURCE,seen[0])


def test_unexecuted_or_failed_candidate_keeps_previous_active_version(tmp_path,monkeypatch):
    cfg,rec=run_public_evidence_fixture(tmp_path,'first')
    def query(args,messages,**kw):return {'content':json.dumps(linked_proposal(json.loads(messages[1]['content'])))}
    monkeypatch.setattr(mem,'_query',query)
    first=mem.finish_linked_episode(cfg,SimpleNamespace())
    root=Path(cfg['shadow_piano_evidence_memory'])
    before=json.loads((root/'active_bundle.json').read_text())['bundle_id']
    next_cfg={**cfg,'output_dir':str(tmp_path/'next')}
    record,_=mem.linked_initial_record(next_cfg,{'fixed':'signature'})
    assert record['bundle_id']==first['candidate_bundle']
    next_cfg,rec2=run_public_evidence_fixture(tmp_path,'next',record['source'],next_cfg)
    manifest=json.loads(rec2.manifest_path.read_text())
    # A function definition alone is not an actual application of the patch.
    press_ids={f['function_id'] for f in manifest['functions'] if f['name']=='press'}
    manifest['invocations']=[i for i in manifest['invocations'] if i['function_id'] not in press_ids]
    mem.atomic_json(rec2.manifest_path,manifest)
    monkeypatch.setattr(mem,'_query',lambda *a,**kw: (_ for _ in ()).throw(TimeoutError('offline')))
    second=mem.finish_linked_episode(next_cfg,SimpleNamespace())
    assert not second['public_execution_checked'] and not second['accepted']
    assert json.loads((root/'active_bundle.json').read_text())['bundle_id']==before
    fallback,_=mem.linked_initial_record({**next_cfg,'output_dir':str(tmp_path/'fallback')},{'fixed':'signature'})
    assert fallback['bundle_id']==before


def test_tactile_memory_pairs_contact_signature_with_public_hold_and_actual_application(tmp_path):
    path=tmp_path/'tactile.json';events=[]
    emit=lambda kind,**row:events.append({'kind':kind,**row})
    models={'white':{'iqr':[1.]*5},'black':{'iqr':[1.]*5}}
    cold=mem.PublicTactileSkillMemory(path,{},models,'collect',emit)
    def feed(bank):
        for step in range(1,10):
            bank.observe('F4','light',{'quality':'ok','estimated_key_force_N':.35},
                [1.]*5,step,step*4,[0,0,.0102,.5],{'index_flexion_rad':.5})
    feed(cold)
    assert cold.query('F4','light','white',.01,9)['status']=='empty'
    src='def press_and_hold():\n    return 1\n'+SOURCE
    assert cold.record('F4','light','white',.01,src,'press_and_hold',9,'cold')['stored']
    assert cold.query('F4','light','white',.01,9)['status']=='empty'
    warm=mem.PublicTactileSkillMemory(path,{},models,'reuse',emit)
    feed(warm)
    answer=warm.query('F4','light','white',.02,9)
    assert answer['status']=='matched' and answer['z_correction_m']==pytest.approx(.0002)
    warm.observe_action([0,.3,.0202,.5],10)
    assert any(e['kind']=='tactile_skill_applied' and e['nonzero_correction'] for e in events)
    before=path.read_bytes()
    assert not warm.record('F4','light','white',.01,src,'press_and_hold',9,'warm')['stored']
    assert path.read_bytes()==before
    warm.next_attempt();feed(warm)
    warm.samples[0]['features']=[100.]*5;warm.samples[1]['features']=[100.]*5
    assert warm.query('F4','light','white',.02,9)['status']=='miss'


def test_tactile_memory_rejects_incomplete_or_invalid_hold(tmp_path):
    bank=mem.PublicTactileSkillMemory(tmp_path/'memory.json',{},
        {'white':{'iqr':[1.]*5}},'collect',lambda *a,**kw:None)
    for step in range(1,10):
        bank.observe('F4','light',{'quality':'ood' if step==5 else 'ok','estimated_key_force_N':.35},
            [1.]*5,step,step*4,[0,0,0,.5],{'index_flexion_rad':.5})
    assert not bank.record('F4','light','white',0,SOURCE,'run_piano',9,'cold')['stored']
    assert not bank.path.exists()


def test_tactile_pilot_freezes_first_controller_before_execution_and_skips_repairs(tmp_path,monkeypatch):
    asset=tmp_path/'asset';asset.write_text('fixed');calls=[]
    cfg={'shadow_piano_tactile_skill_pilot':True,'shadow_piano_code_memory':str(tmp_path/'cache'),
         'output_dir':str(tmp_path/'run'),'_piano_public_request':{'notes':['F4']},
         '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(asset),'digit_force_sidecar':str(asset)}}
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:calls.append(1) or {'content':SOURCE})
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    active=(tmp_path/'cache'/'active.json').read_bytes()
    assert mem.repair(SimpleNamespace(),cfg) is None
    cfg['output_dir']=str(tmp_path/'warm')
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    assert calls==[1] and (tmp_path/'cache'/'active.json').read_bytes()==active


def test_cached_module_accepts_new_requests_and_resets_progress():
    source=mem.validate_source(SOURCE)
    for notes in [['C4','D4'],['F#4','B4','C4']]:
        scope={};exec(mem.wrapped(source,{'notes':notes,'dynamics':['light']*len(notes),'hold_seconds':[.12]*len(notes)},{}),scope)
        assert scope['piano_state']['completed']==len(notes) and scope['capx_stop']
    scope={'piano_state':{'completed':1,'policy_parameters':{}}}
    exec(mem.wrapped(source,{'notes':['C4','F4']},{}),scope)
    assert scope['piano_state']['completed']==2


def test_initial_and_repair_share_timeout_unless_explicitly_overridden(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from capx.llm import client
    deadlines=[]
    @contextmanager
    def deadline(seconds,*,phase):
        deadlines.append((seconds,phase))
        yield
    monkeypatch.setattr(mem,'request_deadline',deadline)
    http_timeouts=[]
    def query(args,_):
        http_timeouts.append(args.request_timeout_seconds)
        return {'content':SOURCE}
    monkeypatch.setattr(client,'query_model',query)
    monkeypatch.setenv('CAPX_LLM_TIMEOUT_SECONDS','300')
    monkeypatch.delenv('CAPX_SHADOW_REPAIR_TIMEOUT_SECONDS',raising=False)
    asset=tmp_path/'asset';asset.write_text('fixed')
    cfg={'shadow_piano_code_memory':str(tmp_path/'cache'),'output_dir':str(tmp_path/'run'),
        '_piano_public_request':{'notes':['F4'],'dynamics':['light'],'hold_seconds':[.12]},
        '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(asset),'digit_force_sidecar':str(asset)}}
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    monkeypatch.setattr(mem,'compact_repair_evidence',lambda _: {})
    mem.repair(SimpleNamespace(),cfg,{'reason':'test repair'})
    assert deadlines == [(300.,'code_memory_initial'),(300.,'code_memory_repair')]
    assert http_timeouts==[300.,300.]
    monkeypatch.setenv('CAPX_SHADOW_REPAIR_TIMEOUT_SECONDS','450')
    mem._query(SimpleNamespace(),[])
    assert deadlines[-1]==(450.,'code_memory_repair') and http_timeouts[-1]==450.


def test_cache_hit_skips_api_and_invalid_signature_misses(tmp_path,monkeypatch):
    sensor=tmp_path/'sensor';sensor.write_text('sensor');model=tmp_path/'model';model.write_text('model')
    cfg={'shadow_piano_code_memory':str(tmp_path/'cache'),'output_dir':str(tmp_path/'run'),
        '_piano_public_request':{'notes':['F4'],'dynamics':['light'],'hold_seconds':[.12]},
        '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(sensor),'digit_force_sidecar':str(model)}}
    signature=mem.compatibility(cfg)
    version=mem.store_version(Path(cfg['shadow_piano_code_memory']),SOURCE,signature,None,{},'public_execution_checked',{})
    monkeypatch.setattr(mem,'_query',lambda *_:pytest.fail('API called on cache hit'))
    code,_,_=mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    assert 'run_piano' in code
    session=json.loads((Path(cfg['output_dir'])/'piano_code_memory_session.json').read_text())
    assert session['cache_hit'] and session['executed_version']==version
    wrong={**signature,'action_repeat':8};assert mem.read_active(Path(cfg['shadow_piano_code_memory']),wrong)==(None,'incompatible')
    path=Path(cfg['shadow_piano_code_memory'])/'versions'/version/'policy.py';path.write_text('corrupt')
    assert mem.read_active(Path(cfg['shadow_piano_code_memory']),signature)==(None,'hash_mismatch')


@pytest.mark.parametrize('source',[SOURCE.replace("request['notes'][i]","'C4'"),SOURCE+'\nrun_piano({}, {})',SOURCE.replace("print('NOTE', request['notes'][i])","open('/tmp/no')")])
def test_reject_nonreusable_or_unsafe_modules(source):
    with pytest.raises(ValueError):mem.validate_source(source)


def test_marker_exception_is_valid_and_runs_inside_host_wrapper():
    source = '''class PianoTerminated(Exception):
    """Local control-flow marker; not a native scoring signal."""
    pass

DT = 4.0 / 240.0
def run_piano(request, piano_state):
    try:
        raise PianoTerminated()
    except PianoTerminated:
        piano_state['completed'] += 1
'''
    scope = {}
    exec(mem.wrapped(source, {'notes': ['F4']}, {}), scope)
    assert scope['piano_state']['completed'] == 1 and scope['capx_stop'] is True


@pytest.mark.parametrize('declaration', [
    'class Unsafe(Exception):\n    get_hand_state()\n',
    'class Unsafe(Exception):\n    def __init__(self):\n        pass\n',
    '@get_hand_state\nclass Unsafe(Exception):\n    pass\n',
    'class Unsafe(get_hand_state()):\n    pass\n',
])
def test_exception_declaration_cannot_execute_or_hide_class_logic(declaration):
    with pytest.raises(ValueError, match='marker exception'):
        mem.validate_source(declaration + SOURCE)


@pytest.mark.parametrize('call', ['solve()', 'run_piano(request, piano_state)'])
def test_entry_calls_are_rejected_with_precise_location(call):
    source = SOURCE + '\n' + call + '\n'
    with pytest.raises(ValueError, match='Top-level call') as error:
        mem.validate_source(source)
    assert call in str(error.value) and 'line ' in str(error.value)
    assert 'exactly once' in str(error.value)


@pytest.mark.parametrize('extra', [
    'state = get_hand_state()\n',
    'def helper(value=get_hand_state()):\n    pass\n',
    '@get_hand_state\ndef helper():\n    pass\n',
])
def test_observation_and_action_setup_cannot_run_while_module_loads(extra):
    with pytest.raises(ValueError, match='during loading|Module-loading expression'):
        mem.validate_source(extra + SOURCE)


def test_host_module_calls_policy_once_without_automatic_solve(monkeypatch):
    from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
    low = SimpleNamespace(get_observation=lambda: {}, get_action_count=lambda: 0, get_step_count=lambda: 0)
    monkeypatch.setattr(CodeExecutionEnvBase, '_build_low_level', lambda *a: low)
    env = CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low, apis=[], prompt='Reusable policy.',
        generated_code_mode='host_module', stream_user_code_output=False))
    calls = []
    env._exec_globals['finish_piano_note_attempt'] = calls.append
    source = '''def solve():
    finish_piano_note_attempt(99)
def run_piano(request, piano_state):
    finish_piano_note_attempt(0)
'''
    result = env._exec_user_code(mem.wrapped(source, {'notes':['F4']}, {}))
    assert result['ok'], result['stderr']
    assert calls == [0]
    assert 'auto-calling generated solve' not in result['stdout']
    assert 'must also call solve()' not in env._system_prompt
    assert 'exactly once' in env._system_prompt


@pytest.mark.parametrize('arm', ['none', 'text', 'tactile', 'joint'])
def test_memory_generation_prompt_uses_one_consistent_module_contract(tmp_path, monkeypatch, arm):
    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config
    from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
    monkeypatch.setenv('CAPX_SHADOW_EVIDENCE_ROOT', str(tmp_path/'memory'))
    monkeypatch.setenv('CAPX_SHADOW_EVIDENCE_MODE', arm)
    factory, cfg, _ = _load_config(LaunchArgs(
        config_path='env_configs/opentac/play_piano_shadow_digit_evidence_memory.yaml'))
    assert factory['cfg']['generated_code_mode'] == 'host_module'
    low = SimpleNamespace(get_observation=lambda: {})
    monkeypatch.setattr(CodeExecutionEnvBase, '_build_low_level', lambda *a: low)
    env = CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low, apis=[],
        prompt=factory['cfg']['prompt'], generated_code_mode=factory['cfg']['generated_code_mode']))
    cfg['output_dir'] = str(tmp_path/'run')
    cfg['_piano_public_request'] = {'notes':['F4'], 'dynamics':['light'], 'hold_seconds':[.12]}
    monkeypatch.setattr(mem, 'compatibility', lambda _: {'test':'module'})
    captured = []
    monkeypatch.setattr(mem, '_query', lambda args, messages, **kw: captured.append(messages) or {'content':SOURCE})
    code, _, _ = mem.initial_code(SimpleNamespace(), cfg, env._get_observation())
    prompt = json.dumps(captured)
    assert 'must also call solve()' not in prompt
    assert 'perform actions immediately' not in prompt
    assert 'calls the entry point exactly once' in prompt
    assert 'No top-level execution or entry-point call' in prompt
    assert 'run_piano(_PUBLIC_PIANO_REQUEST, piano_state)' in code


def test_code_memory_refuses_explicit_script_execution_mode(monkeypatch):
    from capx.envs.launch import LaunchArgs
    from capx.utils import launch_utils
    monkeypatch.setattr(launch_utils.DictLoader, 'load', lambda _: {
        'env': {'cfg': {'generated_code_mode':'script'}}, 'shadow_piano_code_memory':'cache'})
    with pytest.raises(ValueError, match='requires generated_code_mode=host_module'):
        launch_utils._load_config(LaunchArgs(config_path='unused.yaml'))


def test_release_guard_refuses_early_finalization_without_advancing(tmp_path):
    env=ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.piano_continue_on_note_failure=True;env.visual_key_mode=True;env._policy_end_submitted=False
    env.piano_notes=['C4'];env._public_attempts_finalized=0;env.require_public_release=True;env._native_done=False
    env._release_ready_since=100;env._task=SimpleNamespace(step_count=112)
    env.get_observation=lambda:{'hand_state':{'index_flexion_rad':.37,'wrist_z_m':.016}}
    events=[];env._write_public_event=lambda kind,**data:events.append((kind,data))
    with pytest.raises(ValueError,match='release has not completed'):env.finish_piano_note_attempt(0)
    assert env._task.step_count==112 and env._public_attempts_finalized==0
    assert events[0][0]=='release_rejected'


def terminal_return_fixture(monkeypatch, *, consumed=6, finalized=5, verdict=True, native_done=True):
    from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
    low = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    low._native_done = native_done
    low._policy_end_submitted = False
    low._policy_end_verdict = False
    low.visual_key_mode = True
    low.piano_continue_on_note_failure = True
    low.piano_notes = ['F4', 'A4', 'C#4', 'A4', 'A4', 'E4']
    low._public_attempts_finalized = finalized
    low.require_public_release = True
    low._release_ready_since = None
    low._sim_step_count = 1406
    low._native_info = {'notes_attempted': consumed, 'success': verdict, 'actual_force_N': 98765}
    low._task = SimpleNamespace(step_count=5621,
        step=lambda *a: pytest.fail('Terminal cleanup must not step physics'),
        finish_piano_note_attempt=lambda *a: pytest.fail('Already-consumed attempts must not be scored again'))
    low.get_observation = lambda: {'hand_state': {'index_flexion_rad': .14, 'wrist_z_m': -.001}}
    low._save_tactile_snapshot = lambda *a: None
    events, native_events = [], []
    low._write_public_event = lambda kind, **fields: events.append({'kind': kind, **fields})
    low._write_native_event = native_events.append
    monkeypatch.setattr(CodeExecutionEnvBase, '_build_low_level', lambda *a: low)
    high = CodeExecutionEnvBase(CodeExecEnvConfig(low_level=low, apis=[], prompt='Piano.',
        generated_code_mode='host_module', stream_user_code_output=False))
    high._exec_globals['piano_state'] = {'completed': finalized, 'policy_parameters': {}}
    return high, low, events, native_events


@pytest.mark.parametrize('verdict', [True, False])
def test_terminal_return_submits_end_without_changing_native_verdict(monkeypatch, verdict):
    high, low, events, native_events = terminal_return_fixture(monkeypatch, verdict=verdict)
    # Reproduce the archived policy: release returns False after native_done,
    # so run_piano returns normally before updating completed or submitting END.
    source = '''def release():
    print('PIANO_STOP release_timeout')
    return False
def run_piano(request, piano_state):
    if not release():
        return
'''
    snapshot = dict(low._native_info)
    _, reward, terminated, _, info = high.step(mem.wrapped(source, {'notes': low.piano_notes}, {}))
    assert info['sandbox_rc'] == 0 and terminated
    assert reward == float(verdict) and info['task_completed'] is verdict
    assert low._native_info == snapshot and low._task.step_count == 5621
    assert low._public_attempts_finalized == 6 and low._policy_end_submitted
    assert high._exec_globals['capx_stop'] is True
    # Preserve what the policy actually did; host cleanup is separately labeled.
    assert high._exec_globals['piano_state']['completed'] == 5
    assert 'PIANO_STOP release_timeout' in info['stdout'] and '[capx-host] END' in info['stdout']
    finalizations = [e for e in events if e['kind'] == 'attempt_finalized']
    assert finalizations == [{'kind':'attempt_finalized', 'index':5, 'release_verified':False,
                             'finalization_source':'host_native_termination'}]
    assert native_events.count('END') == 1
    assert low.finalize_native_termination() is False and native_events.count('END') == 1
    assert '98765' not in json.dumps(events) and not any('success' in e for e in events)
    with pytest.raises(RuntimeError, match='ended'):
        low.step([0, 0, .04, 0])


@pytest.mark.parametrize('consumed,finalized,new_indices', [(2, 2, []), (3, 2, [2]), (6, 6, [])])
def test_terminal_cleanup_never_invents_unattempted_notes(monkeypatch, consumed, finalized, new_indices):
    high, low, events, native_events = terminal_return_fixture(
        monkeypatch, consumed=consumed, finalized=finalized, verdict=False)
    _, reward, terminated, _, info = high.step(mem.wrapped(
        'def run_piano(request, piano_state):\n    return\n', {'notes':low.piano_notes}, {}))
    assert terminated and reward == 0 and info['task_completed'] is False
    assert low._public_attempts_finalized == consumed
    assert [e['index'] for e in events if e['kind']=='attempt_finalized'] == new_indices
    assert native_events.count('END') == 1


def test_active_release_timeout_does_not_trigger_terminal_cleanup(monkeypatch):
    high, low, events, native_events = terminal_return_fixture(monkeypatch, consumed=5, native_done=False)
    source = 'def run_piano(request, piano_state):\n    print("PIANO_STOP release_timeout")\n'
    _, reward, terminated, _, info = high.step(mem.wrapped(source, {'notes':low.piano_notes}, {}))
    assert not terminated and reward == 0 and not low._policy_end_submitted
    assert low._public_attempts_finalized == 5 and not native_events
    with pytest.raises(ValueError, match='release has not completed'):
        low.finish_piano_note_attempt(5)


def test_terminal_cleanup_does_not_mask_policy_exceptions(monkeypatch):
    high, low, events, native_events = terminal_return_fixture(monkeypatch)
    source = 'def run_piano(request, piano_state):\n    raise ValueError("policy bug")\n'
    _, reward, terminated, _, info = high.step(mem.wrapped(source, {'notes':low.piano_notes}, {}))
    assert info['sandbox_rc'] == 1 and 'policy bug' in info['stderr']
    assert reward == 0 and not terminated and not low._policy_end_submitted
    assert low._public_attempts_finalized == 5 and not native_events


def test_host_terminal_outcome_is_public_but_not_a_release_check(tmp_path):
    from test_shadow_piano_online_skills import evidence_recorder
    rec = evidence_recorder(tmp_path, 'terminal')
    rec.event('attempt_finalized', index=0, release_verified=False,
        finalization_source='host_native_termination', success=True, actual_force_N=98765)
    rec.event('native_terminal_end', finalized_indices=[0], success=True)
    rec.close()
    assert rec.events[-2]['release_verified'] is False
    assert rec.events[-2]['finalization_source'] == 'host_native_termination'
    assert rec.events[-1]['finalized_indices'] == [0]
    assert 'success' not in json.dumps(rec.events) and '98765' not in json.dumps(rec.events)


def test_public_evidence_includes_actual_release_actions_and_excludes_labels(tmp_path):
    (tmp_path/'shadow_public_episode.json').write_text(json.dumps({'seed':1,'sequence':{},'keyboard_cache_hit':True,'clear_action_steps':0}))
    rows=[{'kind':'action','action_step':i,'target':[0,0,.04,0],'hand_state':{'index_flexion_rad':.1/i},'native_done':False,'private_key_top_force_N':99} for i in range(1,11)]
    rows.append({'kind':'attempt_finalized','action_step':10,'index':0})
    (tmp_path/'shadow_public_observations.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    evidence=public_evidence(tmp_path)
    assert len([x for x in evidence['public_events'] if x['kind']=='action'])==8
    assert 'private_key_top_force' not in json.dumps(evidence)


def test_third_camera_writer_uses_same_frame_count_and_audio(monkeypatch,tmp_path):
    calls=[]
    low=SimpleNamespace(get_third_party_video_frames=lambda **_:[np.zeros((16,16,3),dtype=np.uint8)]*4)
    monkeypatch.setattr(trial,'_write_video',lambda frames,base,**kw:calls.append(('video',len(frames),kw)))
    monkeypatch.setattr(trial,'_add_piano_contact_audio',lambda env,base,suffix,start,end:calls.append(('audio',suffix,start,end)))
    trial._save_third_party_video(SimpleNamespace(low_level_env=low),str(tmp_path),5.)
    assert calls==[('video',4,{'suffix':'combined_third','fps':5.}),('audio','combined_third',0,4)]


def test_fixed_parameter_copies_are_readonly_and_zero_usage_is_known():
    from capx.llm.client import summarize_llm_usage
    values={'white_heavy':{'target_force_N':.5,'tolerance_N':.07,'hold_gate_median_N':.473,'probe_trim_m':-.00286}}
    assert mem.validate_parameters(values)==values
    values['white_heavy']['target_force_N']=.6
    with pytest.raises(ValueError,match='Fixed constraints'):mem.validate_parameters(values)
    assert summarize_llm_usage([])['totals']['total_tokens']==0


def test_readonly_diagnostic_never_queries_on_cache_miss_or_repair(tmp_path,monkeypatch):
    p=tmp_path/'asset';p.write_text('fixed')
    cfg={'shadow_piano_code_memory':str(tmp_path/'empty'), 'shadow_piano_code_memory_read_only':True,
        'output_dir':str(tmp_path/'run'),'_piano_public_request':{'notes':[]},
        '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(p),'digit_force_sidecar':str(p)}}
    monkeypatch.setattr(mem,'_query',lambda *_:pytest.fail('Readonly mode called API'))
    with pytest.raises(ValueError,match='Read-only code-memory diagnostic'):
        mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    assert mem.repair(SimpleNamespace(),cfg) is None


def test_safe_uncertain_completion_does_not_trigger_repair(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();root=tmp_path/'memory'
    (run/'shadow_public_episode.json').write_text(json.dumps({'seed':3302,'sequence':{'notes':['D#4']},'keyboard_cache_hit':True,'clear_action_steps':0}))
    events=[{'kind':'action','action_step':1,'target':[0,0,.04,0],'hand_state':{}},
        {'kind':'attempt_finalized','action_step':1,'index':0,'release_verified':True},
        {'kind':'code_execution','action_step':1,'public_messages':["PIANO_NOTE {'quality': 'ood', 'public_hold_confidence': False, 'abort_reason': 'ood'}"]}]
    (run/'shadow_public_observations.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    (run/'piano_code_memory_session.json').write_text(json.dumps({'request':{'notes':['D#4']},'compatibility':{},
        'cache_hit':True,'cache_reason':'hit','input_version':None,'executed_version':mem.digest(SOURCE),
        'source':SOURCE,'repair_calls':0,'parameters':{}}))
    monkeypatch.setattr(mem,'_query',lambda *_:pytest.fail('Safe uncertainty triggered API repair'))
    result=mem.finish_episode({'output_dir':str(run),'shadow_piano_code_memory':str(root)})
    assert result['repair_calls']==0 and result['post_episode_usage']['query_count']==0
    assert result['public_summary']['failures']==[]
    assert len(result['public_summary']['uncertain_observations'])==1
    # A real code error stays actionable, while read-only diagnostics cannot
    # issue a repair request even in this branch.
    with (run/'shadow_public_observations.jsonl').open('a') as stream:
        stream.write('\n'+json.dumps({'kind':'code_execution','action_step':2,
            'public_messages':['ValueError: release has not completed']}))
    result=mem.finish_episode({'output_dir':str(run),'shadow_piano_code_memory':str(root),
        'shadow_piano_code_memory_read_only':True})
    assert result['repair_calls']==0
    assert result['public_summary']['failures']==['ValueError: release has not completed']


def test_recovered_execution_can_be_admitted_without_private_verdict(tmp_path):
    run=tmp_path/'run';run.mkdir();root=tmp_path/'memory'
    (run/'shadow_public_episode.json').write_text(json.dumps({'seed':1,'sequence':{'notes':['C4']},'keyboard_cache_hit':True,'clear_action_steps':0}))
    values={'white_light':{'probe_trim_m':-.00069}}
    events=[{'kind':'code_execution','action_step':1,'public_messages':['ValueError: release not ready']},
        {'kind':'action','action_step':2,'target':[0,0,.04,0],'hand_state':{}},
        {'kind':'attempt_finalized','action_step':2,'index':0,'release_verified':True},
        {'kind':'code_execution','action_step':2,'public_messages':['PIANO_POLICY_STATE '+json.dumps({'policy_parameters':values})]}]
    (run/'shadow_public_observations.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    session={'request':{'notes':['C4']},'compatibility':{},'cache_hit':False,'cache_reason':'empty',
        'input_version':None,'executed_version':mem.digest(SOURCE),'source':SOURCE,'repair_calls':1,'parameters':{}}
    (run/'piano_code_memory_session.json').write_text(json.dumps(session))
    result=mem.finish_episode({'output_dir':str(run),'shadow_piano_code_memory':str(root)})
    active=json.loads((root/'active.json').read_text())
    assert active['status']=='public_execution_checked' and active['parameters']==values
    assert result['repair_calls']==1 and result['post_episode_usage']['query_count']==0


def test_A_generates_fresh_even_if_given_an_existing_cache(tmp_path,monkeypatch):
    asset=tmp_path/'asset';asset.write_text('fixed')
    cfg={'shadow_piano_code_memory':str(tmp_path/'memory'),
        'shadow_piano_memory_policy':mem.ARM_POLICIES['A'],'output_dir':str(tmp_path/'run'),
        '_piano_public_request':{'notes':['F4']},
        '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(asset),'digit_force_sidecar':str(asset)}}
    root=Path(cfg['shadow_piano_code_memory'])
    mem.store_version(root,SOURCE,mem.compatibility(cfg),None,{},'public_execution_checked',{})
    before=(root/'active.json').read_bytes();calls=[]
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:calls.append(a) or {'content':SOURCE})
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    session=json.loads((Path(cfg['output_dir'])/'piano_code_memory_session.json').read_text())
    assert len(calls)==1 and session['cache_hit'] is False
    assert session['input_version'] is None and session['parameters']=={}
    assert (root/'active.json').read_bytes()==before


@pytest.mark.parametrize('arm',['A','B'])
@pytest.mark.parametrize('failed',[False,True])
def test_A_B_never_promote_or_query_after_episode(tmp_path,monkeypatch,arm,failed):
    run=tmp_path/'run';run.mkdir();root=tmp_path/'memory'
    version=mem.store_version(root,SOURCE,{},None,{},'pending_public_execution',{})
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    session={'compatibility':{},'cache_hit':arm=='B','cache_reason':'hit' if arm=='B' else 'fresh',
        'input_version':version if arm=='B' else None,'executed_version':mem.digest(SOURCE+'# repaired'),
        'source':SOURCE+'# repaired','repair_calls':0,'parameters':{},'memory_policy':mem.ARM_POLICIES[arm]}
    (run/'piano_code_memory_session.json').write_text(json.dumps(session))
    monkeypatch.setattr(mem,'public_summary',lambda _: {
        'public_checks_complete':not failed,'last_code_messages':['ValueError: release incomplete'] if failed else [],
        'failures':['Public attempt chain incomplete'] if failed else [],
        'control_failures':[{'index':0,'reason':'alignment_lost'}],
        'parameters':{'white_light':{'trim_m':-.001}}})
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:pytest.fail('A/B made an episode-end API call'))
    # No explicit config policy: the runtime session must preserve isolation.
    result=mem.finish_episode({'output_dir':str(run),'shadow_piano_code_memory':str(root)})
    assert result['post_episode_usage']['query_count']==0 and result['repair_calls']==0
    assert result['updates_retained'] is False and result['post_episode_repair_enabled'] is False
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('reason',['alignment_lost','press_cap','ood','overforce'])
@pytest.mark.parametrize('as_json',[False,True])
def test_explicit_public_control_issue_is_distinct_from_native_verdict(reason,as_json):
    data={'index':0,'note':'untrusted printed name','abort_reason':reason,
        'quality':'ood' if reason=='ood' else 'ok','public_hold_confidence':False,
        'private_key_top_force_N':99,'native_notes_completed':6}
    text=json.dumps(data) if as_json else repr(data)
    issue=mem._public_note_control_failure('PIANO_NOTE '+text,{'notes':['F4']},75)
    assert issue['reason']==reason and issue['index']==0 and issue['note']=='F4'
    assert issue['action_step']==75 and issue['basis']=='public_report_not_native_verdict'
    assert 'private_key_top_force' not in json.dumps(issue)
    assert 'native_notes_completed' not in json.dumps(issue)


def test_ambiguous_or_malformed_public_reports_do_not_invent_control_failures():
    request={'notes':['F4']}
    messages=[
        "PIANO_NOTE {'index':0,'outcome':'uncertain'}",
        "PIANO_NOTE {'index':0,'public_hold_confidence':False}",
        "PIANO_NOTE {'index':0,'public_hold_confidence':True,'abort_reason':'public_hold_provisional'}",
        "PIANO_NOTE {'index':99,'abort_reason':'ood'}",
        "PIANO_NOTE {'index':True,'abort_reason':'ood'}",
        'PIANO_NOTE not a dictionary',
    ]
    assert all(mem._public_note_control_failure(s,request,1) is None for s in messages)


@pytest.mark.parametrize('used_budget,read_only,reason,decision',[
    (0,False,'alignment_lost','requested'),
    (1,False,'alignment_lost','repair_budget_exhausted'),
    (0,True,'alignment_lost','read_only'),
    (0,False,'public_hold_provisional','no_public_failure'),
])
def test_C_public_contact_problem_requests_next_episode_patch(tmp_path,monkeypatch,
        used_budget,read_only,reason,decision):
    run=tmp_path/'run';run.mkdir();root=tmp_path/'memory';calls=[]
    request={'notes':['F4'],'dynamics':['heavy'],'hold_seconds':[.12]}
    (run/'shadow_public_episode.json').write_text(json.dumps({
        'sequence':request,'clear_action_steps':0}))
    note={'index':0,'note':'F4','abort_reason':reason,'quality':'ok',
        'had_valid_contact':reason=='public_hold_provisional',
        'public_hold_confidence':reason=='public_hold_provisional'}
    events=[{'kind':'action','action_step':1,'target':[0,0,.04,0]},
        {'kind':'attempt_finalized','action_step':1,'index':0,'release_verified':True},
        {'kind':'code_execution','action_step':1,'public_messages':['PIANO_NOTE '+repr(note)]}]
    (run/'shadow_public_observations.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    session={'request':request,'compatibility':{},'cache_hit':True,'cache_reason':'hit',
        'input_version':None,'executed_version':mem.digest(SOURCE),'source':SOURCE,
        'repair_calls':used_budget,'parameters':{},'memory_policy':mem.ARM_POLICIES['C'],
        'control_instructions':'fixed controls'}
    (run/'piano_code_memory_session.json').write_text(json.dumps(session))
    # An unreadable private verdict must not affect public feedback or repairs.
    (run/'shadow_piano_native_audit_summary.json').write_text('NOT PUBLIC JSON')
    monkeypatch.setattr(mem,'compact_repair_evidence',lambda _:{'public_events':[]})
    monkeypatch.setattr(mem,'_query',lambda args,messages,**kw:
        calls.append(messages) or {'content':SOURCE+'\n# proposed public control patch'})
    monkeypatch.setenv('CAPX_API_BASE_URL','https://unused.invalid/v1')
    monkeypatch.setenv('CAPX_API_KEY','unused-test-key')
    cfg={'output_dir':str(run),'shadow_piano_code_memory':str(root),
        'shadow_piano_memory_policy':mem.ARM_POLICIES['C'],
        'shadow_piano_code_memory_read_only':read_only}
    result=mem.finish_episode(cfg)
    assert result['public_summary']['failures']==[]
    assert result['post_episode_repair_decision']==decision
    assert result['feedback_version']==mem.PUBLIC_FEEDBACK_VERSION
    assert len(calls)==int(decision=='requested')
    if calls:
        extra=json.loads(calls[0][1]['content'])['public_failure']
        assert extra['control_failures'][0]['reason']=='alignment_lost'
        assert extra['control_failures'][0]['index']==0
        assert mem.OPTIMIZATION_CONTRACT in calls[0][0]['content']
        active=json.loads((root/'active.json').read_text())
        assert active['status']=='pending_public_execution'
        assert active['version']!=mem.digest(SOURCE)
        assert mem.read_active(root,{})[0]['source']!=SOURCE
    mem.finish_episode(cfg)
    assert len(calls)==int(decision=='requested')


@pytest.mark.parametrize('used_budget',[0,1])
def test_C_repairs_with_public_efficiency_objective_within_shared_budget(tmp_path,monkeypatch,used_budget):
    run=tmp_path/'run';run.mkdir();root=tmp_path/'memory';calls=[]
    session={'compatibility':{},'cache_hit':False,'cache_reason':'empty',
        'input_version':None,'executed_version':mem.digest(SOURCE),'source':SOURCE,
        'repair_calls':used_budget,'parameters':{},'memory_policy':mem.ARM_POLICIES['C'],
        'request':{'notes':['F4']},'control_instructions':'fixed controls'}
    (run/'piano_code_memory_session.json').write_text(json.dumps(session))
    summary={'public_checks_complete':False,'last_code_messages':['ValueError: release incomplete'],
        'failures':['Public attempt chain incomplete'],'parameters':{},'completed':1,
        'notes':[{'index':0,'stage_steps':{'arrival':100,'press':90,'release':60},'release_verified':True}]}
    monkeypatch.setattr(mem,'public_summary',lambda _:summary)
    monkeypatch.setattr(mem,'compact_repair_evidence',lambda _:{'public_events':[]})
    monkeypatch.setattr(mem,'_query',lambda args,messages,**kw:calls.append(messages) or {'content':SOURCE})
    monkeypatch.setenv('CAPX_API_BASE_URL','https://unused.invalid/v1')
    monkeypatch.setenv('CAPX_API_KEY','unused-test-key')
    cfg={'output_dir':str(run),'shadow_piano_code_memory':str(root),
         'shadow_piano_memory_policy':mem.ARM_POLICIES['C']}
    result=mem.finish_episode(cfg)
    assert len(calls)==1-used_budget and result['repair_calls']==1
    if calls:
        assert mem.OPTIMIZATION_CONTRACT in calls[0][0]['content']
        evidence=json.loads(calls[0][1]['content'])['public_evidence']
        assert evidence['public_motion_summary']['notes']==summary['notes']
        record=json.loads((root/'active.json').read_text())
        assert record['status']=='pending_public_execution'
    # Re-finalization cannot spend a second request budget.
    mem.finish_episode(cfg)
    assert len(calls)==1-used_budget


ACTION_SOURCE = '''def report_note(note, state):
    """Record an attempted public note without assuming task success."""
    state.setdefault('seen', []).append(note)

def run_piano(request, piano_state):
    for note in request['notes']:
        report_note(note, piano_state)
'''


def action_fixture(tmp_path, source=ACTION_SOURCE):
    run=tmp_path/'run';run.mkdir()
    asset=tmp_path/'asset';asset.write_text('frozen')
    cfg={'output_dir':str(run),'shadow_piano_code_memory':str(tmp_path/'memory'),
        'shadow_piano_action_memory':True,
        '_piano_public_request':{'notes':['C4','F#4'],'dynamics':['light','heavy'],'hold_seconds':[.12,.12]},
        '_piano_low_level':{'action_repeat':4,'digit_piano_config':str(asset),'digit_force_sidecar':str(asset)}}
    session={'compatibility':mem.compatibility(cfg),'cache_hit':False,'cache_reason':'empty',
        'input_version':None,'executed_version':mem.digest(source),'source':source,
        'repair_calls':1,'parameters':{},'action_memory':True,'request':cfg['_piano_public_request']}
    (run/'piano_code_memory_session.json').write_text(json.dumps(session))
    summary={'request':cfg['_piano_public_request'],'public_checks_complete':False,
        'last_code_messages':['ValueError: release incomplete'],
        'failures':['Public attempt chain incomplete'],'parameters':{},'completed':1,
        'notes':[{'index':0,'release_verified':True}],'uncertain_observations':[]}
    return cfg,session,summary


def test_partial_episode_preserved_without_promoting_failed_full_policy(tmp_path,monkeypatch):
    cfg,session,summary=action_fixture(tmp_path)
    monkeypatch.setattr(mem,'public_summary',lambda _:summary)
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:pytest.fail('Budget exhausted'))
    result=mem.finish_episode(cfg)
    root=Path(cfg['shadow_piano_code_memory'])
    assert not (root/'active.json').exists() and result['output_version'] is None
    record=json.loads((root/'episodes/run/episode.json').read_text())
    assert record['public_summary']['completed']==1 and record['status']=='experimental_partial'
    assert Path(record['source']).read_text()==ACTION_SOURCE
    assert record['public_summary']['failures']==summary['failures']
    # Normal-return evidence for a parent does not manufacture function success.
    function=json.loads((root/'action_skills.json').read_text())['bundles'][0]['functions'][0]
    assert function['name']=='report_note' and function['portable']
    assert 'success_count' not in function
    again=mem.finish_episode(cfg)
    assert again==result


def test_new_melody_reuses_local_helper_body_but_only_signature_is_sent(tmp_path,monkeypatch):
    cfg,session,summary=action_fixture(tmp_path)
    monkeypatch.setattr(mem,'public_summary',lambda _:summary)
    mem.finish_episode(cfg)
    next_cfg={**cfg,'output_dir':str(tmp_path/'next'),
        '_piano_public_request':{'notes':['G4','D4','B4'],'dynamics':['light']*3,'hold_seconds':[.12]*3}}
    generated='''def run_piano(request, piano_state):
    for note in request['notes']:
        report_note(note, piano_state)
'''
    prompts=[]
    def query(args,messages,**kw):
        prompts.append(messages)
        return {'content':generated}
    monkeypatch.setattr(mem,'_query',query)
    code,_,_=mem.initial_code(SimpleNamespace(),next_cfg,{'full_prompt':[]})
    scope={};exec(code,scope)
    assert scope['piano_state']['seen']==['G4','D4','B4']
    prompt=json.dumps(prompts)
    assert 'report_note(note, state)' in prompt
    assert "state.setdefault('seen'" not in prompt  # Actual function body stays local.
    next_session=json.loads((Path(next_cfg['output_dir'])/'piano_code_memory_session.json').read_text())
    assert not next_session['cache_hit']
    assert next_session['action_functions_offered'][0]['status']=='experimental_partial'
    assert 'def report_note' in next_session['source']  # Executed source is reproducible.
    assert len(prompts)==1


def test_closure_lifting_requires_context_and_dependency_closure():
    source='''LIMIT = 1
def run_piano(request, piano_state):
    LIMIT = 2
    def pure(x):
        return abs(x)
    def nested(x):
        return pure(x)
    def captured(x):
        return piano_state['completed'] + LIMIT + x
    def bad_dependency(x):
        return captured(x)
'''
    rows={x['name']:x for x in mem.extract_action_functions(source)}
    assert rows['pure']['portable'] and rows['nested']['portable']
    assert rows['nested']['dependencies']==['pure']
    assert not rows['captured']['portable']
    assert rows['captured']['captured_context']==['LIMIT','piano_state']
    assert not rows['bad_dependency']['portable']
    assert rows['captured']['argument_fields']=={}


def test_action_memory_must_replan_pending_module_and_respects_readonly(tmp_path,monkeypatch):
    cfg,session,summary=action_fixture(tmp_path)
    root=Path(cfg['shadow_piano_code_memory'])
    mem.store_version(root,ACTION_SOURCE,session['compatibility'],None,{},'pending_public_execution',{})
    calls=[]
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:calls.append(1) or {'content':ACTION_SOURCE})
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})
    assert calls==[1]
    # A checked full policy still skips all model work.
    mem.store_version(root,ACTION_SOURCE,session['compatibility'],None,{},'public_execution_checked',{})
    monkeypatch.setattr(mem,'_query',lambda *a,**kw:pytest.fail('API called on checked cache hit'))
    mem.initial_code(SimpleNamespace(),cfg,{'full_prompt':[]})


def test_action_memory_config_reaches_runtime(monkeypatch,tmp_path):
    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config
    monkeypatch.setenv('CAPX_SHADOW_CODE_MEMORY',str(tmp_path/'memory'))
    monkeypatch.delenv('CAPX_SHADOW_ACTION_MEMORY',raising=False)
    _,cfg,_=_load_config(LaunchArgs(config_path='env_configs/opentac/play_piano_shadow_digit_code_memory.yaml'))
    assert mem.action_memory_enabled(cfg)
    monkeypatch.setenv('CAPX_SHADOW_ACTION_MEMORY','false')
    _,cfg,_=_load_config(LaunchArgs(config_path='env_configs/opentac/play_piano_shadow_digit_code_memory.yaml'))
    assert not mem.action_memory_enabled(cfg)
    monkeypatch.setenv('CAPX_TACTILE_SKILL_MEMORY',str(tmp_path/'tactile.json'))
    _,cfg,_=_load_config(LaunchArgs(config_path='env_configs/opentac/play_piano_shadow_digit_tactile_skill_pilot.yaml'))
    assert cfg['shadow_piano_tactile_skill_pilot'] is True
    assert mem.memory_policy(cfg)['mode']=='frozen'


def test_timing_keeps_execution_separate_and_chooses_one_usage_snapshot(tmp_path):
    cfg,session,summary=action_fixture(tmp_path)
    run=Path(cfg['output_dir'])
    for name,queries in [('trial_01_in_progress',[{'phase':'initial_code','latency_seconds':10.}]),
                         ('trial_01_done',[{'phase':'initial_code','latency_seconds':10.},
                            {'phase':'multi_turn','request_wall_seconds':20.}])]:
        folder=run/name;folder.mkdir()
        (folder/'llm_usage.json').write_text(json.dumps({'queries':queries}))
    mem.record_execution_timing(cfg,block=1,seconds=42.,returned=True,action_start=0,action_end=30)
    report=mem.timing_report(run,{'queries':[{'phase':'code_memory_repair','latency_seconds':3.}]})
    assert report['api']['sum_seconds']==33. and report['api']['query_count']==3
    assert report['code_execution_seconds']==42.
    assert report['code_blocks'][0]['source_version']==session['executed_version']


def test_unused_library_bodies_are_not_assembled_and_dependencies_are_kept():
    prefix='''def leaf(x):
    return abs(x)
def used(x):
    return leaf(x)
def unused(x):
    return -x
'''
    cards=[{'name':'leaf','dependencies':[]},{'name':'used','dependencies':['leaf']},
        {'name':'unused','dependencies':[]}]
    source='def run_piano(request, piano_state):\n    used(1)\n'
    assembled,names=mem.assemble_action_source(source,prefix,cards)
    assert names==['leaf','used'] and 'def unused' not in assembled
    scope={};exec(mem.wrapped(assembled,{'notes':[]},{}),scope)
    fresh='def run_piano(request, piano_state):\n    print(request)\n'
    assembled,names=mem.assemble_action_source(fresh,prefix,cards)
    assert names==[] and assembled==mem.validate_source(fresh)


def test_corrupt_function_index_falls_back_and_preserves_original_on_rebuild(tmp_path,monkeypatch):
    cfg,session,summary=action_fixture(tmp_path)
    root=Path(cfg['shadow_piano_code_memory']);root.mkdir()
    (root/'action_skills.json').write_text('{broken')
    assert mem.load_action_functions(root,{})==('',[])
    monkeypatch.setattr(mem,'public_summary',lambda _:summary)
    mem.finish_episode(cfg)
    assert next((root/'rejected_indexes').glob('*.json')).read_text()=='{broken'
    assert json.loads((root/'action_skills.json').read_text())['bundles']


def test_parallel_api_failure_sets_shared_abort_and_prevents_next_request(tmp_path, monkeypatch):
    flag = tmp_path / 'api_abort.json'
    monkeypatch.setenv('CAPX_LLM_ABORT_FILE', str(flag))
    monkeypatch.setenv('CAPX_OUTPUT_DIR', str(tmp_path))
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('request failed secret-test-key')
    monkeypatch.setattr(mem, '_query_impl', failing)
    args = SimpleNamespace(model='test', api_key='secret-test-key')
    with pytest.raises(RuntimeError, match='request failed'):
        mem._query(args, [], phase='initial')
    first = flag.read_text()
    assert 'secret-test-key' not in first and '[REDACTED]' in first
    assert json.loads(first)['phase'] == 'initial'
    with pytest.raises(RuntimeError, match='earlier API failure'):
        mem._query(args, [], phase='update')
    assert calls == [1] and flag.read_text() == first
    events = [json.loads(line) for line in (tmp_path/'piano_api_requests.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'failed']


def test_parallel_success_records_timing_without_abort(tmp_path, monkeypatch):
    flag = tmp_path / 'api_abort.json'
    monkeypatch.setenv('CAPX_LLM_ABORT_FILE', str(flag))
    monkeypatch.setenv('CAPX_OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(mem, '_query_impl', lambda *a, **kw: {'content': 'code'})
    assert mem._query(SimpleNamespace(model='test'), []) == {'content': 'code'}
    assert not flag.exists()
    events = [json.loads(line) for line in (tmp_path/'piano_api_requests.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'completed']
    assert events[0]['request_id'] == events[1]['request_id']


def test_parallel_reply_is_not_executed_after_other_request_failed(tmp_path, monkeypatch):
    flag = tmp_path / 'api_abort.json'
    monkeypatch.setenv('CAPX_LLM_ABORT_FILE', str(flag))
    monkeypatch.setenv('CAPX_OUTPUT_DIR', str(tmp_path))
    def reply(*args, **kwargs):
        flag.write_text('{"error":"other request"}')
        return {'content':'code'}
    monkeypatch.setattr(mem, '_query_impl', reply)
    with pytest.raises(RuntimeError, match='another request failed'):
        mem._query(SimpleNamespace(model='test'), [])
