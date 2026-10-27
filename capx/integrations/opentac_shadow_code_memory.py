"""Public-only executable policy memory. Never consume native verdict artifacts."""
from __future__ import annotations
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import symtable
import textwrap
import numpy as np

from capx.integrations.opentac_shadow_keyboard import atomic_json
from capx.integrations.opentac_shadow_online_skills import public_evidence
from capx.llm.deadline import request_deadline
from capx.integrations.opentac_shadow_skills import _PRIVATE_MARKERS

SCHEMA = 'shadow_piano_executable_memory.v1'
PUBLIC_FEEDBACK_VERSION = 'public_control_feedback.v2'
ARM_POLICIES = {
    'A': {'arm': 'A', 'mode': 'fresh', 'post_episode_repair': False, 'optimize_actions_on_repair': False},
    'B': {'arm': 'B', 'mode': 'frozen', 'post_episode_repair': False, 'optimize_actions_on_repair': False},
    'C': {'arm': 'C', 'mode': 'adaptive', 'post_episode_repair': True, 'optimize_actions_on_repair': True},
}
OPTIMIZATION_CONTRACT = '''First address the reported public control problems; completing an attempt safely does not
establish that contact or the required hold was achieved. A public problem is not a native scoring verdict.
After resolving the public failure, improve reusable action planning using ONLY the supplied
public motion summary and recent observations. Minimize redundant approach, force-trim, and release/wait
actions per finalized attempt. Prefer actual-pose arrival checks and bounded adaptive motion over excess
fixed waits. Do not reduce required hold/release duration, the fixed hover height, minimum press duration,
force bounds, or any other fixed constraint. Fewer actions caused by aborted/unattempted notes are not
an efficiency improvement. Preserve completed progress; never replay finalized notes. Generalize helpers
and the four color/dynamic parameter groups, not seed-specific notes or coordinates. Only change a planning
choice supported by public evidence; otherwise preserve it. Briefly document any evidence-backed change
as a Python comment in the returned module. This is a proposed optimization, not proof of native success.'''
CONTRACT = '''Return ONLY Python source for a reusable module defining run_piano(request, piano_state).
No top-level execution or entry-point call: the host supplies the request and calls the function.
request contains notes, dynamics and hold_seconds. Iterate those lists, never hardcode the episode melody,
seed, or note-specific wrist coordinates. A0/C8 endpoint names are allowed for first visual calibration.
Use the documented public piano APIs, math, and builtin Python only. Never open files or import task modules.
Read cached mapped_keys or establish the endpoint map once using the initial HEAD image.
piano_state persists ONLY within this episode and contains completed, keys, attempts, policy_parameters.
Store four black/white x light/heavy parameter dictionaries under piano_state['policy_parameters'].
Reuse helper functions for every note; retain completed progress on recovery. All control bounds, force
targets, estimator and timing constraints in the task instructions remain fixed.
After each attempt release fully, then call finish_piano_note_attempt(i), increment completed and print
PIANO_NOTE with public quality, hold confidence and abort reason. The adapter requires measured index
flexion <=0.02 rad and wrist Z within 0.002 m of hover, continuously for >=0.12 simulated seconds before
finalization. Send bounded extension/lift actions and continue checking actual pose; a sent target does
not establish arrival. Do not read key angles or native scoring. If native_done is true no more motion
is allowed; finalization and return remain permitted. At a first early OOD checkpoint print
PIANO_STOP OOD_REPROBE and return without finalizing that attempt. Return after all attempts; the host
submits END. Do not use globals(), eval(), exec(), arbitrary attributes or filesystem access.
When adapting parameters only use public observations and the existing numerical limits.'''

ACTION_CONTRACT = '''Define reusable action helpers at MODULE LEVEL, with explicit arguments for key targets,
controller parameters, duration and episode state. run_piano chooses/composes these helpers for the current
request; never bake the melody into a helper. Keep useful helpers small. A helper must return its public
outcome rather than assuming a normal Python return proves contact or task success. A function depending
on an enclosing function's local variables cannot be independently reused until those become arguments.
If LOCAL ACTION FUNCTIONS are supplied, their implementations are already loaded by the host. You may
call them by the exact documented signature without reprinting/redefining their bodies. Their evidence
status refers to the parent program, not individually verified task success. Experimental functions may
contain the recorded defect: inspect the limitations, choose only applicable functions, and generate a
replacement under a NEW function name when necessary. Reuse is optional; reason about this request.
After native_done do not move, but still call finish_piano_note_attempt for any remaining lawful final
submission before advancing completed. A completed counter alone is not evidence of finalization.'''

PUBLIC_FUNCTIONS = {'get_keyboard_endpoints', 'locate_visual_keys', 'get_hand_state',
    'move_shadow_index', 'wait_steps', 'get_index_tactile_rgb', 'estimate_key_force',
    'finish_piano_note_attempt', 'query_tactile_skill', 'record_tactile_skill'}


def action_memory_enabled(config):
    value = config.get('shadow_piano_action_memory', False)
    if isinstance(value, str) and value.lower() in ('true', 'false', '1', '0'):
        return value.lower() in ('true', '1')
    if type(value) is not bool:
        raise ValueError('shadow_piano_action_memory must be boolean')
    return value


def digest(value):
    return hashlib.sha256(value.encode('utf-8') if isinstance(value,str) else value).hexdigest()


def memory_policy(config):
    """Validate explicit comparison policies; retain legacy suite behavior."""
    if config.get('shadow_piano_tactile_skill_pilot'):
        return {'arm': 'C_tactile_pilot', 'mode': 'frozen',
            'post_episode_repair': False, 'optimize_actions_on_repair': False}
    raw = config.get('shadow_piano_memory_policy')
    if raw is None:
        return {'arm': None, 'mode': 'adaptive', 'post_episode_repair': True, 'optimize_actions_on_repair': False}
    if (not isinstance(raw, dict) or not isinstance(raw.get('arm'), str)
            or raw.get('arm') not in ARM_POLICIES
            or type(raw.get('post_episode_repair')) is not bool
            or type(raw.get('optimize_actions_on_repair')) is not bool
            or raw != ARM_POLICIES[raw['arm']]):
        raise ValueError('Invalid A/B/C Shadow piano memory policy')
    return dict(raw)


def validate_source(source):
    source=source.strip()
    if source.startswith('```'):
        source=re.sub(r'^```(?:python)?\s*|\s*```$', '', source)
    if len(source.encode())>64000 or not source:
        raise ValueError('Policy source is empty or exceeds 64 KB')
    if any(m in source.lower() for m in _PRIVATE_MARKERS):
        raise ValueError('Private marker in policy source')
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_piano']
    if len(functions)!=1 or [a.arg for a in functions[0].args.args]!=['request','piano_state']:
        raise ValueError('Policy must define run_piano(request, piano_state)')
    for n in tree.body:
        if not isinstance(n,(ast.FunctionDef,ast.Import,ast.Assign,ast.AnnAssign)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):
            raise ValueError('Policy module must not execute actions at top level')
    for n in ast.walk(tree):
        if isinstance(n,ast.Import) and any(x.name!='math' for x in n.names):
            raise ValueError('Only math imports are allowed')
        if isinstance(n,ast.ImportFrom):raise ValueError('from imports are not allowed')
        if isinstance(n,ast.Attribute) and n.attr.startswith('_'):
            raise ValueError('Private attributes are not public APIs')
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'open','exec','eval','compile','__import__','globals','getattr','setattr','input','exit','quit'}:
            raise ValueError('Unsupported policy operation: '+n.func.id)
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and re.fullmatch(r'[A-G](?:#|b)?[0-8]',n.value) and n.value not in ('A0','C8'):
            raise ValueError('Hardcoded episode note is not reusable: '+n.value)
    compile(tree,'<piano-code-memory>','exec')
    return source+'\n'


def validate_parameters(parameters):
    if not isinstance(parameters,dict) or len(json.dumps(parameters,allow_nan=False))>8192:
        raise ValueError('Policy parameters must be a bounded public dictionary')
    allowed={'white_light','white_heavy','black_light','black_heavy'}
    if not set(parameters)<=allowed:raise ValueError('Parameters must use the four color/dynamic classes')
    fixed = {'white_light':(.35,.07,.311),'white_heavy':(.5,.07,.473),
             'black_light':(.7,.14,.615),'black_heavy':(1.,.14,1.04)}
    for name,values in parameters.items():
        if not isinstance(values,dict) or len(values)>16:raise ValueError('Invalid class parameter dictionary')
        for k,v in values.items():
            if not isinstance(k,str) or not isinstance(v,(int,float)) or isinstance(v,bool):raise ValueError('Parameters must be finite numeric scalars')
            frozen=dict(zip(('target_force_N','tolerance_N','hold_gate_median_N'),fixed[name]))
            if k in frozen:
                if v!=frozen[k]:raise ValueError('Fixed constraints are not mutable memory parameters')
            elif any(s in k.lower() for s in ('target_force','tolerance','threshold','estimator','score','reward','success')):
                raise ValueError('Fixed constraints are not mutable memory parameters')
    return parameters


def compatibility(config):
    low=config.get('_piano_low_level') or config['env']['cfg']['low_level']
    def sha(path):return digest(Path(path).read_bytes())
    signature = {'schema':SCHEMA,'api':'ShadowPianoVisualForceApi.v1','entry':'run_piano(request,piano_state)',
        'action_repeat':int(low['action_repeat']),'physics_hz':240,
        'sensor':sha(low['digit_piano_config']),'force_model':sha(low['digit_force_sidecar']),
        'frame':'public_endpoint_geometry.v1','release_guard':'measured_pose_0p12s.v1'}
    if action_memory_enabled(config):
        signature['action_memory'] = 'parameterized_functions.v1'
    return signature


def extract_action_functions(source):
    """Extract Python functions without pretending closures are standalone skills.

    A nested function can be lifted only if every external name is a public API,
    builtin, literal module constant or another portable helper. No LLM call,
    simulator access or policy execution is needed for this operation.
    """
    source = validate_source(source)
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = ast.get_source_segment(source, node)
    tables = {}
    def visit_table(table):
        for child in table.get_children():
            tables[(child.get_name(), child.get_lineno())] = child
            visit_table(child)
    visit_table(symtable.symtable(source, '<action-memory>', 'exec'))
    nodes = []
    def visit_nodes(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                nodes.append((child, '.'.join((*parents, child.name))))
                visit_nodes(child, (*parents, child.name))
            else:
                visit_nodes(child, parents)
    visit_nodes(tree)
    helpers = [(n, q) for n, q in nodes if n.name != 'run_piano']
    counts = {n.name: sum(other.name == n.name for other, _ in helpers) for n, _ in helpers}
    names = {n.name for n, _ in helpers}
    result = []
    for node, qualified in helpers:
        table = tables[(node.name, node.lineno)]
        free = set(table.get_frees()) if hasattr(table, 'get_frees') else set()
        external = set(free)
        def globals_used(t):
            out = {s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced()}
            for child in t.get_children():
                out.update(globals_used(child))
            return out
        external.update(globals_used(table))
        dependencies = sorted((external & names) - {node.name})
        captured = sorted((free - names) | (external - names - set(constants) - PUBLIC_FUNCTIONS - set(dir(builtins)) - {'math'}))
        reason = []
        if captured: reason.append('requires_explicit_arguments:' + ','.join(captured))
        if counts[node.name] > 1: reason.append('ambiguous_function_name')
        if node.decorator_list: reason.append('decorated_function')
        for value in [*node.args.defaults, *(x for x in node.args.kw_defaults if x is not None)]:
            try: ast.literal_eval(value)
            except (ValueError, TypeError): reason.append('nonliteral_default')
        arguments = {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
        argument_fields = {name: set() for name in arguments}
        return_fields = set()
        for part in ast.walk(node):
            if isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name) and part.value.id in arguments:
                if isinstance(part.slice, ast.Constant) and isinstance(part.slice.value, str):
                    argument_fields[part.value.id].add(part.slice.value)
            if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and isinstance(part.func.value, ast.Name):
                if part.func.value.id in arguments and part.func.attr in ('get', 'setdefault', 'pop') and part.args:
                    if isinstance(part.args[0], ast.Constant) and isinstance(part.args[0].value, str):
                        argument_fields[part.func.value.id].add(part.args[0].value)
            if isinstance(part, ast.Return) and isinstance(part.value, ast.Dict):
                return_fields.update(k.value for k in part.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
        result.append({'name': node.name, 'qualified_name': qualified,
            'signature': node.name + '(' + ast.unparse(node.args) + ')',
            'description': (ast.get_docstring(node) or '')[:300],
            'code': textwrap.dedent(ast.get_source_segment(source, node)) + '\n',
            'dependencies': dependencies, 'captured_context': captured,
            'argument_fields': {k: sorted(v) for k, v in argument_fields.items() if v},
            'return_fields': sorted(return_fields),
            'line': node.lineno, 'end_line': node.end_lineno,
            'portable': not reason, 'limitations': reason,
            'constants': [constants[k] for k in sorted(external & constants.keys())]})
    # A helper with an unavailable dependency is unavailable too.
    for _ in result:
        available = {x['name'] for x in result if x['portable']}
        for row in result:
            if row['portable'] and not set(row['dependencies']) <= available:
                row['portable'] = False
                row['limitations'].append('nonportable_dependency')
    return result


def archive_proposal(config, source, stage):
    """Persist incomplete/invalid proposals independently of the active policy."""
    root = Path(config['shadow_piano_code_memory'])
    version = digest(source)
    directory = root / 'candidates' / version
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'policy.py'
    if not path.exists(): path.write_text(source, encoding='utf-8')
    error = None
    try: validate_source(source)
    except (ValueError, SyntaxError) as exc: error = str(exc)
    session_path = _session_path(config)
    signature = json.loads(session_path.read_text())['compatibility'] if session_path.exists() else compatibility(config)
    record = {'version': version, 'run_id': Path(config['output_dir']).name,
        'stage': stage, 'status': 'invalid' if error else 'proposed', 'validation_error': error,
        'source': str(path), 'compatibility': signature}
    atomic_json(directory / 'proposals' / (record['run_id'] + '_' + stage + '.json'), record)
    return record


def archive_episode(config, session, summary, checked):
    """Keep public experience even when the whole melody cannot be admitted.

    Function status describes parent-protocol evidence only. It is never a
    per-function execution success claim or a native piano verdict.
    """
    root = Path(config['shadow_piano_code_memory'])
    run = Path(config['output_dir'])
    source = session['source']
    proposal = archive_proposal(config, source, 'episode_end')
    safe = {k: summary[k] for k in ('request', 'completed', 'failures', 'control_failures', 'notes',
        'parameters', 'uncertain_observations', 'public_checks_complete') if k in summary}
    record = {**proposal, 'status': 'parent_protocol_checked' if checked else 'experimental_partial',
        'public_summary': safe, 'public_observations': str(run / 'shadow_public_observations.jsonl'),
        'limitations': 'Public protocol evidence only; individual function effects and native task success are not established.'}
    if proposal['validation_error']: record['status'] = 'invalid'
    atomic_json(root / 'episodes' / run.name / 'episode.json', record)
    atomic_json(run / 'piano_experience_archive.json', record)
    if session.get('action_memory') and not proposal['validation_error']:
        functions = extract_action_functions(source)
        atomic_json(root / 'candidates' / proposal['version'] / 'functions.json', functions)
        index_path = root / 'action_skills.json'
        index = {'schema': 'public_action_functions.v1', 'bundles': []}
        if index_path.exists():
            raw = index_path.read_text()
            try:
                loaded = json.loads(raw)
                if loaded.get('schema') != index['schema'] or not isinstance(loaded.get('bundles'), list):
                    raise ValueError('Invalid function index')
                index = loaded
            except (ValueError, AttributeError):
                backup = root / 'rejected_indexes' / (digest(raw) + '.json')
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists(): backup.write_text(raw, encoding='utf-8')
        bundle = {k: record[k] for k in ('version', 'run_id', 'status', 'source', 'compatibility', 'limitations')}
        bundle['public_completed'] = summary.get('completed')
        bundle['public_failures'] = summary.get('failures', [])[-4:]
        bundle['functions'] = functions
        previous = [x for x in index['bundles'] if x['version'] == bundle['version']]
        # Reuse in a failed episode must not erase a previous checked observation.
        if previous and previous[-1]['status'] == 'parent_protocol_checked' and not checked:
            bundle = previous[-1]
        index['bundles'] = [x for x in index['bundles'] if x['version'] != bundle['version']] + [bundle]
        index['bundles'] = index['bundles'][-128:]  # Full evidence stays in episodes/candidates.
        atomic_json(index_path, index)
        record['portable_functions'] = sum(x['portable'] for x in functions)
    return record


def load_action_functions(root, signature):
    """Offer one compatible bundle; do not merge conflicting helper versions.

    Only declarations are loaded. Calling experimental helpers requires the new
    LLM-generated plan; an incomplete full policy is never directly executed.
    """
    path = root / 'action_skills.json'
    if not path.exists(): return '', []
    try:
        index = json.loads(path.read_text())
        if index.get('schema') != 'public_action_functions.v1' or not isinstance(index.get('bundles'), list):
            return '', []
    except (OSError, ValueError, AttributeError):
        return '', []
    bundles = [b for b in reversed(index['bundles']) if b['compatibility'] == signature and b['status'] in ('parent_protocol_checked', 'experimental_partial')]
    bundles.sort(key=lambda b: b['status'] != 'parent_protocol_checked')
    for bundle in bundles:
        source_path = root / 'candidates' / bundle['version'] / 'policy.py'
        try:
            source = source_path.read_text(encoding='utf-8')
            if digest(source) != bundle['version']: continue
            functions = [x for x in extract_action_functions(source) if x['portable']]
            # An assembled policy may contain offered helpers it never uses.
            # They are archived, but should not crowd out its actual actions.
            entry = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_piano')
            needed = {n.id for n in ast.walk(entry) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for _ in functions:
                needed.update(d for x in functions if x['name'] in needed for d in x['dependencies'])
            functions = [x for x in functions if x['name'] in needed]
        except (ValueError, SyntaxError, OSError): continue
        if not functions or len(functions) > 16: continue
        constants = sorted({c for x in functions for c in x['constants']})
        prefix = 'import math\n' + '\n'.join(constants) + '\n' + '\n'.join(x['code'] for x in functions)
        cards = [{k: x[k] for k in ('name', 'signature', 'description', 'dependencies', 'argument_fields', 'return_fields')} | {
            'source_version': bundle['version'], 'status': bundle['status'],
            'parent_public_completed': bundle.get('public_completed'),
            'parent_public_failures': bundle.get('public_failures', []),
            'limitations': bundle['limitations']} for x in functions]
        return prefix, cards
    return '', []


def assemble_action_source(source, prefix, cards):
    """Keep only referenced local helpers and their dependency closure."""
    source = validate_source(source)
    tree = ast.parse(source)
    offered = {x['name']: x for x in cards}
    definitions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if definitions & offered.keys():
        raise ValueError('Redefined a supplied action helper; use a new name for a replacement')
    needed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} & offered.keys()
    for _ in cards:
        needed.update(d for name in list(needed) for d in offered[name]['dependencies'])
    if not needed:
        return source, []
    declarations = [ast.get_source_segment(prefix, n) for n in ast.parse(prefix).body
        if not isinstance(n, ast.FunctionDef) or n.name in needed]
    return '\n'.join(declarations) + '\n' + source, sorted(needed)


def record_execution_timing(config, **row):
    """Record code-block wall time separately from API wait and action counts."""
    try:
        run = Path(config['output_dir'])
        path = _session_path(config)
        session = json.loads(path.read_text()) if path.exists() else {}
        row.update(schema='piano_execution_timing.v1', source_version=session.get('executed_version'),
            scope='Generated code wall time, including simulation/rendering/tactile estimation; excludes model requests.')
        with (run / 'piano_execution_timing.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    except (OSError, ValueError) as exc:
        print('[piano-code-memory] execution timing unavailable: ' + str(exc), flush=True)


def timing_report(run, post_usage=None):
    """Read usage snapshots only; never infer model compute time from tokens."""
    from capx.llm.client import summarize_llm_latency
    snapshots = []
    for path in run.glob('trial_*/llm_usage.json'):
        usage = json.loads(path.read_text())
        snapshots.append((len(usage.get('queries', [])), path.stat().st_mtime_ns, path, usage))
    latest = max(snapshots, key=lambda x: (x[0], x[1])) if snapshots else None
    queries = list(latest[3].get('queries', [])) if latest else []
    queries += (post_usage or {}).get('queries', [])
    blocks_path = run / 'piano_execution_timing.jsonl'
    blocks = [json.loads(s) for s in blocks_path.read_text().splitlines() if s] if blocks_path.exists() else []
    report = {'api': summarize_llm_latency(queries), 'code_blocks': blocks,
        'code_execution_seconds': sum(x['seconds'] for x in blocks) if blocks_path.exists() else None,
        'usage_artifact': str(latest[2]) if latest else None,
        'limitations': 'API time includes network/queue/compute/retries. Non-streaming requests cannot isolate reasoning or time-to-first-token. Preflight is separate and not measured here.'}
    if latest is None:
        report['api']['sum_seconds'] = None
        report['api']['scope'] += ' Trial usage artifact missing; request absence is not established.'
    atomic_json(run / 'piano_timing_summary.json', report)
    return report


def read_active(root,signature):
    path=root/'active.json'
    if not path.exists():return None,'empty'
    try:
        record=json.loads(path.read_text())
        if record.get('status')=='quarantined':return None,'previous_execution_rejected'
        if record['compatibility']!=signature:return None,'incompatible'
        source=(root/'versions'/record['version']/'policy.py').read_text()
        if digest(source)!=record['version']:return None,'hash_mismatch'
        validate_source(source);validate_parameters(record.get('parameters',{}))
        return {**record,'source':source},'hit'
    except (ValueError,KeyError,OSError,SyntaxError) as exc:return None,'invalid:'+type(exc).__name__


def store_version(root,source,signature,parent,parameters,status,evidence):
    source=validate_source(source);parameters=validate_parameters(parameters)
    version=digest(source);directory=root/'versions'/version;directory.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'active.json').read_text()) if (root/'active.json').exists() else {}
    if parent==version:
        parent=previous.get('parent_version')
        if status=='pending_public_execution' and previous.get('status')=='public_execution_checked':
            status='public_execution_checked'
    p=directory/'policy.py'
    if not p.exists():p.write_text(source,encoding='utf-8')
    record={'schema':SCHEMA,'version':version,'parent_version':parent,'compatibility':signature,
        'parameters':parameters,'status':status,'public_evidence':evidence}
    # Keep each provenance record, including repeat uses of the same source.
    with (directory/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    atomic_json(root/'active.json',record)
    return version


def wrapped(source,request,parameters):
    source=validate_source(source)
    # The dynamic request is host-supplied public data, never string-substituted in code.
    return ('import json\n_PUBLIC_PIANO_REQUEST = '+repr(request)+'\n'
        "piano_state = globals().setdefault('piano_state', {'completed': 0, 'keys': None, 'attempts': [], 'policy_parameters': "+repr(parameters)+"})\n"
        +source+"\ntry:\n    run_piano(_PUBLIC_PIANO_REQUEST, piano_state)\nfinally:\n"
        "    print('PIANO_POLICY_STATE', json.dumps({'policy_parameters': piano_state.get('policy_parameters', {})}, allow_nan=False))\n"
        "if piano_state.get('completed', 0) == len(_PUBLIC_PIANO_REQUEST['notes']):\n    capx_stop = True\n    print('END')\n")


def _session_path(config):return Path(config['output_dir'])/'piano_code_memory_session.json'


def _query(args,messages,*,deadline_seconds=None,phase='code_memory_repair'):
    from capx.llm.client import query_model
    # This mutates only a task-specific argument copy; API keys never enter artifacts.
    args=copy.copy(args);args.max_tokens=8192
    if deadline_seconds is None:
        deadline_seconds=float(os.getenv('CAPX_SHADOW_REPAIR_TIMEOUT_SECONDS',
            os.getenv('CAPX_LLM_TIMEOUT_SECONDS','300')))
    args.request_timeout_seconds=float(deadline_seconds)
    print(f'[piano-code-memory] phase={phase} request_deadline_seconds={deadline_seconds:g} '
          f'http_timeout_seconds={deadline_seconds:g}',flush=True)
    with request_deadline(deadline_seconds,phase=phase):return query_model(args,messages)


def initial_code(args,config,obs):
    root=Path(config['shadow_piano_code_memory']);signature=compatibility(config)
    policy=memory_policy(config)
    action_memory = action_memory_enabled(config)
    request=config['_piano_public_request']
    if policy['mode']=='fresh':
        record,reason=None,'fresh_generation_required'
    else:
        record,reason=read_active(root,signature)
    if action_memory and record and record.get('status') != 'public_execution_checked':
        record,reason=None,'candidate_requires_new_plan'
    if record and not config.get('_piano_key_map_ready',True):
        record=None;reason='keyboard_calibration_required'
    if config.get('shadow_piano_code_memory_read_only') and record is None:
        raise ValueError('Read-only code-memory diagnostic requires a compatible policy and keyboard calibration: '+reason)
    session={'schema':SCHEMA,'request':request,'compatibility':signature,'cache_hit':record is not None,
        'cache_reason':reason,'input_version':record['version'] if record else None,
        'parameters':record.get('parameters',{}) if record else {},'repair_calls':0,
        'memory_policy':policy, 'action_memory': action_memory}
    print(f"[piano-code-memory] arm={policy['arm']} mode={policy['mode']} "
          f"post_episode_repair={policy['post_episode_repair']}",flush=True)
    instructions=[]
    for message in obs['full_prompt']:
        content=message.get('content','')
        if isinstance(content,str):instructions.append(content)
        elif isinstance(content,list):instructions.extend(x['text'] for x in content if x.get('type')=='text')
    session['control_instructions']='\n'.join(instructions)
    if record:
        source=record['source'];reasoning='';session['executed_version']=record['version']
        print('[piano-code-memory] cache hit version='+record['version'],flush=True)
    else:
        messages=copy.deepcopy(obs['full_prompt']);messages.append({'role':'user','content':CONTRACT})
        prefix, cards = ('', [])
        if action_memory:
            if policy['mode'] != 'fresh':
                prefix, cards = load_action_functions(root, signature)
            messages.append({'role': 'user', 'content': ACTION_CONTRACT +
                '\nLOCAL ACTION FUNCTIONS (implementations already loaded, no source retransmission):\n' + json.dumps(cards)})
        session['action_functions_offered'] = cards
        # Initial generation and repairs both default to the launcher's timeout.
        # A caller may explicitly choose a separate repair budget.
        out=_query(args,messages,deadline_seconds=float(os.getenv('CAPX_LLM_TIMEOUT_SECONDS','300')),
            phase='code_memory_initial')
        source=out['content'];reasoning=out.get('reasoning','')
        raw_source = source
        if prefix:
            try:
                source, selected = assemble_action_source(source, prefix, cards)
                session['action_functions_referenced'] = selected
            except (ValueError, SyntaxError) as exc:
                # Retain the raw proposal and send its assembly error to the
                # normal bounded repair path, never execute ambiguous bodies.
                session['assembly_error'] = str(exc)
        session['executed_version']=digest(source)
        atomic_json(Path(config['output_dir'])/'piano_code_initial_response.json',{'content':source,
            'model_content':raw_source, 'action_functions_offered':cards})
    session['source']=source
    atomic_json(_session_path(config),session) # Retain invalid first proposals for one bounded repair.
    try:
        if session.get('assembly_error'): raise ValueError(session['assembly_error'])
        source=validate_source(source)
        session['source']=source;session['executed_version']=digest(source);atomic_json(_session_path(config),session)
        if config.get('shadow_piano_tactile_skill_pilot') and record is None:
            store_version(root,source,signature,None,{},'pending_public_execution',
                {'basis':'frozen_before_execution_for_tactile_memory_pair'})
        if policy['mode'] == 'adaptive' and not config.get('shadow_piano_code_memory_read_only'):
            archive_proposal(config, source, 'initial')
        return wrapped(source,request,session['parameters']),reasoning,None
    except (ValueError,SyntaxError) as exc:
        session['validation_error']=str(exc);atomic_json(_session_path(config),session)
        if policy['mode'] == 'adaptive' and not config.get('shadow_piano_code_memory_read_only'):
            archive_proposal(config, source, 'initial_invalid')
        return "raise ValueError("+repr('CodeMemoryValidationError: '+str(exc))+")",reasoning,None


def repair(args,config,extra=None):
    if config.get('shadow_piano_tactile_skill_pilot'):
        print('[piano-code-memory] tactile pilot freezes source; no repair request', flush=True)
        return None
    if config.get('shadow_piano_code_memory_read_only'):
        return None
    path=_session_path(config);session=json.loads(path.read_text())
    if session['repair_calls']>=1:return None
    session['repair_calls']+=1;atomic_json(path,session)
    evidence=compact_repair_evidence(Path(config['output_dir']))
    policy=session.get('memory_policy') or memory_policy(config)
    optimization=''
    if policy.get('optimize_actions_on_repair'):
        optimization='\n'+OPTIMIZATION_CONTRACT
        # Compact stage counts only, never private scores or the full trajectory.
        summary=public_summary(Path(config['output_dir']))
        evidence['public_motion_summary']={
            'completed':summary['completed'], 'notes':summary['notes'],
            'stage_basis':'public commanded motion; not native contact or scoring stages'}
    # public_evidence's explicit whitelist never reads the native verdict paths.
    action_contract = '\n' + ACTION_CONTRACT if session.get('action_memory') else ''
    prompt=[{'role':'system','content':session['control_instructions']+'\n'+CONTRACT+action_contract+'\nRepair the whole self-contained reusable module, including any helpers it uses. Only public evidence is available; do not infer native success.'+optimization},
        {'role':'user','content':json.dumps({'current_source':session['source'],'request':session['request'],
          'public_evidence':evidence,'public_failure':extra or {}},ensure_ascii=False)}]
    atomic_json(Path(config['output_dir'])/'piano_code_repair_prompt.json',{'messages':prompt})
    out=_query(args,prompt)
    atomic_json(Path(config['output_dir'])/'piano_code_repair_response.json',{'content':out['content']})
    if policy['mode'] == 'adaptive': archive_proposal(config, out['content'], 'repair_proposal')
    try:source=validate_source(out['content'])
    except (ValueError,SyntaxError) as exc:
        session['repair_error']=str(exc);atomic_json(path,session);return None
    session['source']=source;session['executed_version']=digest(source);session.pop('validation_error',None)
    session.pop('assembly_error',None)
    session['repair_validated']=True;atomic_json(path,session)
    return wrapped(source,session['request'],session['parameters'])


def compact_repair_evidence(run):
    """One source copy belongs in the repair prompt; telemetry is a small whitelist.

    Keep the last failure context and release samples without uploading full
    robot joint arrays or the generated wrapper a second time. Full telemetry
    remains on disk for offline inspection, never native verdict artifacts.
    """
    evidence=public_evidence(run)
    events=evidence['public_events']
    hand_keys=('wrist_x_offset_m','wrist_y_m','wrist_z_m',
               'index_flexion_rad','wrist_and_index_velocity')
    selected={e['event_id'] for e in events if e['kind']=='attempt_finalized'}
    for kind,count in (('action',10),('force_estimate',8),('release_rejected',2),('code_execution',2)):
        selected.update(e['event_id'] for e in [x for x in events if x['kind']==kind][-count:])
    compact=[]
    for event in events:
        if event['event_id'] not in selected:continue
        row={k:event[k] for k in ('kind','event_id','action_step','target','note',
             'estimate','index','continuous_release_seconds') if k in event}
        if 'hand_state' in event:
            row['hand_state']={k:event['hand_state'][k] for k in hand_keys if k in event['hand_state']}
        if event['kind']=='code_execution':
            messages=event.get('public_messages',[])
            important=[s for s in messages if s.startswith(('PIANO_STOP','PIANO_NOTE',
                'PIANO_POLICY_STATE','ValueError:','RuntimeError:','StaticCodeError:'))
                or re.match(r'^[A-Za-z]+(?:Error|Exception):',s)]
            row['public_messages']=[s[:1000] for s in (important or messages)[-10:]]
        compact.append(row)
    result={k:evidence[k] for k in ('schema','run_id','request','quality_counts','limitations')}
    result['public_events']=compact
    # Do not let an unusually verbose policy recreate the former huge prompt.
    # The newest error/pose context has priority over older note summaries.
    while len(json.dumps(result,ensure_ascii=False))>14000 and len(compact)>1:
        compact.pop(0)
    result['selection']='last public failure/release context; full code supplied once separately'
    return result


def _public_note_control_failure(line, request, action_step):
    """Classify explicit public reports, without treating uncertainty as a verdict."""
    if not line.startswith('PIANO_NOTE ') or len(line) > 4096:
        return None
    text = line[len('PIANO_NOTE '):]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None
    if not isinstance(data, dict):
        return None
    index = data.get('index')
    notes = request.get('notes', [])
    if type(index) is not int or not 0 <= index < len(notes):
        return None
    reason = data.get('abort_reason')
    reason = reason.lower() if isinstance(reason, str) else ''
    quality = data.get('quality')
    known_reasons = {'alignment_lost', 'no_contact', 'missing_contact', 'press_cap',
        'hold_uncertain', 'hold_timeout', 'release_timeout', 'ood', 'overforce'}
    if reason not in known_reasons:
        if quality == 'ood':
            reason = 'ood'
        elif data.get('had_valid_contact') is False:
            reason = 'no_valid_contact'
        else:
            return None
    # Never copy arbitrary fields from generated text into the repair evidence.
    result = {'index': index, 'note': notes[index], 'reason': reason,
        'action_step': action_step, 'basis': 'public_report_not_native_verdict'}
    if quality in ('ok', 'ood', 'no_contact', 'unavailable', 'stale'):
        result['quality'] = quality
    for key in ('had_valid_contact', 'public_hold_confidence'):
        if type(data.get(key)) is bool:
            result[key] = data[key]
    return result


def public_summary(run):
    episode=json.loads((run/'shadow_public_episode.json').read_text())
    events=[json.loads(s) for s in (run/'shadow_public_observations.jsonl').read_text().splitlines()]
    actions=[e for e in events if e['kind']=='action']
    failures=[];control_failures=[];uncertain_observations=[];messages=[];last_messages=[];params={};previous=episode['clear_action_steps']
    notes=[]
    for e in events:
        if e['kind']=='attempt_finalized':
            aa=[x for x in actions if previous<x['action_step']<=e['action_step']]
            counts={'arrival':0,'press':0,'release':0};pressed=False;releasing=False;last_flex=0.
            for x in aa:
                flex=x['target'][3]
                if flex>=.05:pressed=True
                if pressed and flex < last_flex-1.e-5:releasing=True
                phase=('release' if releasing else ('press' if pressed else 'arrival'))
                counts[phase]+=1
                last_flex=flex
            notes.append({'index':e['index'],'stage_steps':counts,'release_verified':e.get('release_verified',False)})
            previous=e['action_step']
        elif e['kind']=='code_execution':
            last_messages=e['public_messages']
            messages+=e['public_messages']
            for line in e['public_messages']:
                if line.startswith('PIANO_POLICY_STATE '):
                    try:params=validate_parameters(json.loads(line[len('PIANO_POLICY_STATE '):])['policy_parameters'])
                    except (ValueError,KeyError):pass
                if line.startswith(('ValueError:','RuntimeError:','StaticCodeError:','PIANO_STOP')):failures.append(line)
                control_failure = _public_note_control_failure(line, episode['sequence'], e['action_step'])
                if control_failure is not None:
                    control_failures.append(control_failure)
                if line.startswith('PIANO_NOTE') and any(v in line for v in ('hold_uncertain','press_cap',"'hold_confidence': False","'public_hold_confidence': False",'ood','overforce')):
                    # Preserve uncertainty separately from explicit public
                    # control problems. Neither establishes a native verdict.
                    uncertain_observations.append(line)
    completed=sum(e['kind']=='attempt_finalized' for e in events)
    if completed!=len(episode['sequence']['notes']):failures.append('Public attempt chain incomplete')
    return {'request':episode['sequence'],'completed':completed,'failures':failures,
        'control_failures':control_failures,'feedback_version':PUBLIC_FEEDBACK_VERSION,'messages':messages,
        'parameters':params,'notes':notes,'last_code_messages':last_messages,
        'uncertain_observations':uncertain_observations,
        'public_checks_complete':completed==len(episode['sequence']['notes']) and all(x['release_verified'] for x in notes)}


def finish_episode(config):
    """Called before private summary consumption, after execution has stopped."""
    from capx.llm.client import ModelQueryArgs,collect_llm_usage,summarize_llm_usage
    run=Path(config['output_dir']);path=_session_path(config)
    if not path.exists():return {'error':'no_code_memory_session'}
    session=json.loads(path.read_text());summary=public_summary(run)
    saved_policy=session.get('memory_policy')
    policy=memory_policy(config) if config.get('shadow_piano_memory_policy') is not None else (
        saved_policy or memory_policy(config))
    if saved_policy and saved_policy!=policy:
        raise ValueError('Memory policy changed between execution and episode finalization')
    result_path = run / 'piano_code_memory_result.json'
    def evidence_id():
        latest_session = json.loads(path.read_text())
        return digest(json.dumps({'summary': summary, 'source': latest_session.get('source'),
            'repair_calls': latest_session.get('repair_calls'), 'policy': policy,
            'read_only': bool(config.get('shadow_piano_code_memory_read_only'))}, sort_keys=True))
    if result_path.exists():
        previous_result = json.loads(result_path.read_text())
        if previous_result.get('finalized_evidence') == evidence_id():
            return previous_result
    result={'cache_hit':session['cache_hit'],'cache_reason':session['cache_reason'],
        'input_version':session['input_version'],'executed_version':session['executed_version'],
        'public_summary':summary,'repair_calls':session['repair_calls'],
        'memory_policy':policy,'updates_retained':policy['mode']=='adaptive',
        'post_episode_repair_enabled':policy['post_episode_repair'],
        'feedback_version':PUBLIC_FEEDBACK_VERSION}
    root=Path(config['shadow_piano_code_memory']);events=[]
    if policy['mode'] in ('fresh','frozen'):
        # A/B may repair within this episode, but never spend another API call
        # after it ends or promote those changes into the next episode.
        current,_=read_active(root,session['compatibility']) if policy['mode']=='frozen' else (None,None)
        result['output_version']=current['version'] if current else None
        result['post_episode_repair_decision']='disabled_by_policy'
        result['post_episode_usage']=summarize_llm_usage([])
        result['timing'] = timing_report(run, result['post_episode_usage'])
        result['finalized_evidence'] = evidence_id()
        atomic_json(run/'piano_code_memory_result.json',result)
        return result
    # Admit executed code only with a complete, publicly released attempt chain.
    has_error = any(re.match(r'^[A-Za-z]+(?:Error|Exception):',line) for line in summary['last_code_messages'])
    checked = summary['public_checks_complete'] and not session.get('validation_error') and not has_error
    if not config.get('shadow_piano_code_memory_read_only'):
        result['experience_archive'] = archive_episode(config, session, summary, checked)
    if checked:
        result['output_version']=store_version(root,session['source'],session['compatibility'],
            session['input_version'],summary['parameters'],'public_execution_checked',{'run_id':run.name})
    else:
        current,_=read_active(root,session['compatibility'])
        if has_error and current and current['version']==session['executed_version']:
            parent=current.get('parent_version');seen={current['version']};restored=None
            while parent and parent not in seen:
                seen.add(parent);history=root/'versions'/parent/'history.jsonl'
                if not history.exists():break
                record=json.loads(history.read_text().splitlines()[-1])
                if record['status']=='public_execution_checked' and record['compatibility']==session['compatibility']:
                    restored=record;break
                parent=record.get('parent_version')
            if restored:
                atomic_json(root/'active.json',restored);result['rolled_back_to']=restored['version'];current=restored
            else:
                rejected={k:v for k,v in current.items() if k!='source'};rejected['status']='quarantined'
                atomic_json(root/'active.json',rejected);current=None
        result['output_version']=current['version'] if current else None
    control_failures = summary.get('control_failures', []) if policy['arm'] == 'C' else []
    if config.get('shadow_piano_code_memory_read_only'):
        decision = 'read_only'
    elif not policy['post_episode_repair']:
        decision = 'disabled_by_policy'
    elif not (summary['failures'] or control_failures):
        decision = 'no_public_failure'
    elif session['repair_calls'] >= 1:
        decision = 'repair_budget_exhausted'
    else:
        decision = 'requested'
    result['post_episode_repair_decision'] = decision
    if decision == 'requested':
        print('[piano-code-memory] post-episode repair requested: '
              f"code_failures={len(summary['failures'])} control_failures={len(control_failures)}", flush=True)
        args=ModelQueryArgs(model=os.environ.get('MODEL',os.environ.get('CAPX_MODEL','gpt-6-sol')),
            server_url=os.environ['CAPX_API_BASE_URL'].rstrip('/')+'/responses',api_key=os.environ['CAPX_API_KEY'],max_tokens=8192,reasoning_effort='low')
        try:
            with collect_llm_usage(events,'code_memory_repair'):
                repaired=repair(args,config,{'failures':summary['failures'],
                    'control_failures':control_failures,'notes':summary['notes']})
            session=json.loads(path.read_text())
            if repaired:
                if session.get('action_memory'):
                    # An unexecuted repair stays a candidate; it cannot replace
                    # the last checked policy merely because Python parses.
                    result['candidate_version'] = digest(session['source'])
                else:
                    result['output_version']=store_version(root,session['source'],session['compatibility'],
                        result['output_version'],summary['parameters'],'pending_public_execution',{'run_id':run.name,'basis':'public_failure_repair'})
        except Exception as exc:result['repair_error']=f'{type(exc).__name__}: {exc}'
    result['repair_calls']=json.loads(path.read_text())['repair_calls']
    result['post_episode_usage']=summarize_llm_usage(events)
    result['timing'] = timing_report(run, result['post_episode_usage'])
    result['finalized_evidence'] = evidence_id()
    atomic_json(run/'piano_code_memory_result.json',result)
    return result


class PublicTactileSkillMemory:
    """Small public-only contact-signature -> bounded controller-parameter memory.

    A process reads one immutable input snapshot. Collection writes proposals
    for the NEXT episode; no cold-run retrieval can see its own new entries.
    """
    schema = 'shadow_digit_tactile_skill_memory.v1'

    def __init__(self, path, signature, models, mode, emit):
        if mode not in ('collect', 'reuse'):
            raise ValueError('Tactile skill mode must be collect or reuse')
        self.path=Path(path);self.signature=signature;self.models=models
        self.mode=mode;self.emit=emit;self.samples=[];self.pending=None
        self.proposals=[];self.applied=set()
        data=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {
            'schema':self.schema,'signature':signature,'entries':[]}
        if data.get('schema')!=self.schema or data.get('signature')!=signature:
            raise ValueError('Incompatible tactile skill memory')
        self.entries=data['entries']
        if not isinstance(self.entries,list) or len(self.entries)>16:
            raise ValueError('Invalid tactile skill entry list')
        self.emit('tactile_memory_loaded', mode=mode, input_entries=len(self.entries),
                  input_hash=digest(json.dumps(data,sort_keys=True)))

    def next_attempt(self):
        self.samples=[];self.pending=None

    def observe(self, note, dynamics, estimate, features, step, physics_step, action, hand):
        row={'note':note,'dynamics':dynamics,'step':int(step),'physics_step':int(physics_step),
             'quality':estimate.get('quality'),'force':estimate.get('estimated_key_force_N'),
             'features':features,'action':list(action),
             'actual_flexion':hand.get('index_flexion_rad')}
        self.samples.append(row)
        if len(self.samples)>96:self.samples.pop(0)
        if row['quality']!='ok':self.pending=None

    def _signature_vector(self):
        good=[r for r in self.samples if r['quality']=='ok' and r['features'] is not None]
        # The entry phase is the first three valid contact observations, not
        # the later force-optimized hold. Match the same phase on retrieval.
        if len(good)<3:return None
        value=np.asarray([r['features'] for r in good[:3]],float)
        return np.median(value,axis=0) if value.shape==(3,5) and np.isfinite(value).all() else None

    def query(self, note, dynamics, color, probe_z, step):
        self.pending=None
        vector=self._signature_vector()
        if vector is None:return {'status':'collecting','skill_id':None}
        if not self.samples or self.samples[-1]['step']!=step or self.samples[-1]['quality']!='ok':
            return {'status':'unavailable','skill_id':None}
        if self.mode=='collect':
            self.emit('tactile_skill_query', status='empty', note=note, dynamics=dynamics,
                      features=vector.tolist())
            return {'status':'empty','skill_id':None}
        iqr=np.asarray(self.models[color]['iqr'],float)
        candidates=[]
        for entry in self.entries:
            if entry['color']!=color or entry['dynamics']!=dynamics:continue
            delta=np.abs((vector-np.asarray(entry['entry_features'],float))/iqr)
            distance=float(np.sqrt(np.mean(delta**2)))
            if distance<=1.0 and float(delta.max())<=3.0:candidates.append((distance,entry))
        if not candidates:
            self.emit('tactile_skill_query',status='miss',note=note,dynamics=dynamics,features=vector.tolist())
            return {'status':'miss','skill_id':None}
        distance,entry=min(candidates,key=lambda x:x[0])
        correction=float(entry['z_correction_m'])
        if not np.isfinite(correction) or abs(correction)>.001:
            raise ValueError('Unsafe stored tactile skill correction')
        self.pending={'skill_id':entry['skill_id'],'recommended_z':probe_z+correction,
                      'probe_z':probe_z,'queried_step':step}
        self.emit('tactile_skill_query',status='matched',note=note,dynamics=dynamics,
                  skill_id=entry['skill_id'],distance=distance,features=vector.tolist(),
                  z_correction_m=correction,source_version=entry['source_version'])
        return {'status':'matched','skill_id':entry['skill_id'],'distance':distance,
                'z_correction_m':correction,'source_version':entry['source_version'],
                'function_name':entry['function_name']}

    def observe_action(self, action, step):
        if self.pending is None or step<=self.pending['queried_step']:return
        target=float(action[2]);p=self.pending
        if action[3]>=.495 and abs(target-p['recommended_z'])<=2.e-6 and p['skill_id'] not in self.applied:
            self.applied.add(p['skill_id'])
            self.emit('tactile_skill_applied',**p,applied_step=step,target=list(action),
                      nonzero_correction=abs(target-p['probe_z'])>1.e-7)

    def record(self, note, dynamics, color, probe_z, source, function_name, step, run_id):
        if self.mode!='collect':return {'stored':False,'reason':'reuse_snapshot_is_read_only'}
        vector=self._signature_vector();recent=self.samples[-9:]
        if vector is None or len(recent)!=9:return {'stored':False,'reason':'insufficient_public_history'}
        gate,upper={'white_light':(.311,.42),'white_heavy':(.473,.57),
                    'black_light':(.615,.84),'black_heavy':(1.04,1.14)}[color+'_'+dynamics]
        qualifies=all(r['note']==note and r['dynamics']==dynamics and r['quality']=='ok'
            and isinstance(r['force'],(int,float)) and np.isfinite(r['force'])
            and gate<r['force']<upper and r['actual_flexion'] is not None
            and r['actual_flexion']>=.495 for r in recent)
        consecutive=all(b['step']==a['step']+1 for a,b in zip(recent,recent[1:]))
        if not qualifies or not consecutive or recent[-1]['step']!=step or (
                recent[-1]['physics_step']-recent[0]['physics_step'])/240<.12:
            return {'stored':False,'reason':'public_hold_not_observed'}
        functions=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==function_name]
        if len(functions)!=1:return {'stored':False,'reason':'unknown_controller_function'}
        correction=float(np.median([r['action'][2]-probe_z for r in recent]))
        if abs(correction)>.001:return {'stored':False,'reason':'correction_outside_fixed_bounds'}
        entry={'note_source':note,'color':color,'dynamics':dynamics,'entry_features':vector.tolist(),
            'z_correction_m':correction,'function_name':function_name,'source_version':digest(source),
            'function_hash':digest(ast.dump(functions[0])), 'source_run':run_id,
            'hold_start_step':recent[0]['step'],'hold_end_step':step,
            'status':'public_hold_observed_not_native_verdict'}
        entry['skill_id']=digest(json.dumps(entry,sort_keys=True))
        if not any(e['color']==color and e['dynamics']==dynamics for e in self.proposals):
            self.proposals.append(entry)
            atomic_json(self.path,{'schema':self.schema,'signature':self.signature,'entries':self.proposals})
            self.emit('tactile_skill_recorded',**entry)
            return {'stored':True,'skill_id':entry['skill_id']}
        return {'stored':False,'reason':'class_already_recorded'}
