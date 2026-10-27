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
