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
Do not add a solve() wrapper, solve(), run_piano(...), or an if __name__ guard at module level.
The module may contain math imports, inert constants, functions and simple marker exception classes.
A marker exception class may inherit Exception, RuntimeError or ValueError and contain only pass
and an optional docstring; no class methods, decorators or executable class-body statements.
Do not call observation/action helpers in module assignments, function defaults or decorators.
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
not establish arrival. Do not read key angles or native scoring. native_done means the episode stopped,
not that release timed out or the task succeeded. Propagate it through helpers and return normally
from run_piano immediately, without more motion or a fabricated release_timeout. The host reconciles
already-consumed attempts and submits END with the unchanged native verdict. Do not mark unattempted
notes completed. At a first early OOD checkpoint print
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
After native_done do not move: propagate the terminal outcome and return normally from run_piano.
The host closes already-consumed attempts and END; do not invent progress for unattempted notes.
A completed counter alone is not evidence of finalization.'''

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
    if config.get('shadow_piano_evidence_memory'):
        policy = evidence_policy(config)
        return {'arm': 'tactile_evidence_' + policy['mode'],
                'mode': 'adaptive' if policy['update'] else 'fresh',
                'post_episode_repair': False, 'optimize_actions_on_repair': policy['update']}
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
        if isinstance(n, ast.ClassDef):
            inert_exception = (
                len(n.bases) == 1 and isinstance(n.bases[0], ast.Name)
                and n.bases[0].id in {'Exception', 'RuntimeError', 'ValueError'}
                and not n.decorator_list and not n.keywords
                and all(isinstance(part, ast.Pass) or (
                    isinstance(part, ast.Expr) and isinstance(part.value, ast.Constant)
                    and isinstance(part.value.value, str)) for part in n.body)
            )
            if not inert_exception:
                raise ValueError(f'Class {n.name!r} at line {n.lineno} must be a marker exception '
                                 'with an allowed exception base and only pass/docstring')
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            raise ValueError(f'Top-level call {ast.unparse(n.value)!r} at line {n.lineno} is not allowed; '
                             'the host calls run_piano(request, piano_state) exactly once. '
                             'Remove entry-point calls; move setup/action calls inside policy functions.')
        if not isinstance(n,(ast.FunctionDef,ast.Import,ast.Assign,ast.AnnAssign)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):
            raise ValueError(f'Unsupported module-level {type(n).__name__} at line {n.lineno}; '
                             'put control flow inside run_piano or its helpers')
        eager_values = []
        if isinstance(n, ast.FunctionDef):
            if n.decorator_list:
                raise ValueError(f'Function decorators execute during loading: {n.name} at line {n.lineno}')
            eager_values = [*n.args.defaults, *n.args.kw_defaults, n.returns,
                *(a.annotation for a in [*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs]),
                n.args.vararg.annotation if n.args.vararg else None,
                n.args.kwarg.annotation if n.args.kwarg else None]
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            eager_values = [n.value]
            if isinstance(n, ast.AnnAssign):eager_values.append(n.annotation)
        if any(isinstance(part, (ast.Call, ast.NamedExpr)) for value in eager_values
               if value is not None for part in ast.walk(value)):
            raise ValueError(f'Module-loading expression at line {n.lineno} must not call functions; '
                             'move observation/action setup inside run_piano')
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
        'execution_contract':'host_policy_module.v2',
        'terminal_finalization':'native_done_host.v1',
        'action_repeat':int(low['action_repeat']),'physics_hz':240,
        'sensor':sha(low['digit_piano_config']),'force_model':sha(low['digit_force_sidecar']),
        'frame':'public_endpoint_geometry.v1','release_guard':'measured_pose_0p12s.v1'}
    if action_memory_enabled(config):
        signature['action_memory'] = 'parameterized_functions.v1'
    if config.get('shadow_piano_evidence_memory'):
        signature['evidence_policy'] = evidence_policy(config)
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
    """Optionally stop an entire parallel batch on its first request failure."""
    abort_file = os.environ.get('CAPX_LLM_ABORT_FILE')
    if not abort_file:
        return _query_impl(args, messages, deadline_seconds=deadline_seconds, phase=phase)
    import time
    import uuid
    abort = Path(abort_file)
    if abort.exists():
        raise RuntimeError('Parallel batch stopped after an earlier API failure')
    abort.parent.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    output = os.environ.get('CAPX_OUTPUT_DIR')
    base = {'request_id': request_id, 'pid': os.getpid(), 'phase': phase,
            'run': output, 'model': getattr(args, 'model', None)}

    def record_event(event, **fields):
        if output:
            with (Path(output) / 'piano_api_requests.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({**base, 'event': event, 'time_ns': time.time_ns(), **fields}) + '\n')

    record_event('started')
    try:
        response = _query_impl(args, messages, deadline_seconds=deadline_seconds, phase=phase)
    except Exception as exc:
        message = str(exc)
        key = getattr(args, 'api_key', None)
        if key:
            message = message.replace(key, '[REDACTED]')
        failure = {**base, 'time_ns': time.time_ns(), 'error_type': type(exc).__name__,
                   'error': message[:2000]}
        # Exclusive creation retains the first failure if several requests fail.
        try:
            with abort.open('x', encoding='utf-8') as stream:
                json.dump(failure, stream, ensure_ascii=False)
                stream.write('\n')
        except FileExistsError:
            pass
        record_event('failed', error_type=type(exc).__name__)
        print(f'[piano-code-memory] API_FAILURE_ABORT file={abort}', flush=True)
        raise
    record_event('completed')
    if abort.exists():
        raise RuntimeError('Parallel batch stopped after another request failed')
    return response


def _query_impl(args,messages,*,deadline_seconds=None,phase='code_memory_repair'):
    from capx.llm.client import query_model
    # This mutates only a task-specific argument copy; API keys never enter artifacts.
    args=copy.copy(args);args.max_tokens=8192
    if deadline_seconds is None:
        deadline_seconds=float(os.getenv('CAPX_SHADOW_REPAIR_TIMEOUT_SECONDS',
            os.getenv('CAPX_LLM_TIMEOUT_SECONDS','300')))
    args.request_timeout_seconds=float(deadline_seconds)
    print(f'[piano-code-memory] phase={phase} request_deadline_seconds={deadline_seconds:g} '
          f'http_timeout_seconds={deadline_seconds:g}',flush=True)
    # Parallel simulators may share a capacity-limited provider. Serialize only
    # model requests, not episodes; queue time is not provider inference latency.
    lock_path = os.environ.get('CAPX_LLM_SERIAL_LOCK')
    if not lock_path:
        with request_deadline(deadline_seconds,phase=phase):return query_model(args,messages)
    import fcntl
    import time
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    limit = float(os.environ.get('CAPX_LLM_QUEUE_TIMEOUT_SECONDS', '1200'))
    with path.open('a') as lock:
        print(f'[piano-code-memory] phase={phase} waiting_for_api_slot', flush=True)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= limit:
                    raise TimeoutError('API queue deadline exceeded before request')
                time.sleep(.2)
        waited = time.monotonic() - started
        output = os.environ.get('CAPX_OUTPUT_DIR')
        if output:
            with (Path(output) / 'piano_api_queue.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'phase': phase, 'queue_seconds': waited}) + '\n')
        print(f'[piano-code-memory] phase={phase} api_slot_acquired queue_seconds={waited:.3f}', flush=True)
        try:
            with request_deadline(deadline_seconds,phase=phase):return query_model(args,messages)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def initial_code(args,config,obs):
    root=Path(config['shadow_piano_code_memory']);signature=compatibility(config)
    policy=memory_policy(config)
    action_memory = action_memory_enabled(config)
    request=config['_piano_public_request']
    if policy['mode']=='fresh':
        record,reason=None,'fresh_generation_required'
    elif config.get('shadow_piano_evidence_memory'):
        record,reason=linked_initial_record(config,signature)
    else:
        record,reason=read_active(root,signature)
    if action_memory and record and record.get('status') != 'public_execution_checked':
        record,reason=None,'candidate_requires_new_plan'
    if record and not config.get('_piano_key_map_ready',True) and not config.get('shadow_piano_evidence_memory'):
        record=None;reason='keyboard_calibration_required'
    if config.get('shadow_piano_code_memory_read_only') and record is None:
        raise ValueError('Read-only code-memory diagnostic requires a compatible policy and keyboard calibration: '+reason)
    session={'schema':SCHEMA,'request':request,'compatibility':signature,'cache_hit':record is not None,
        'cache_reason':reason,'input_version':record['version'] if record else None,
        'parameters':record.get('parameters',{}) if record else {},'repair_calls':0,
        'memory_policy':policy, 'action_memory': action_memory}
    if config.get('shadow_piano_evidence_memory'):
        session['evidence_input_bundle'] = config.get('_linked_input_bundle')
        session['evidence_policy'] = evidence_policy(config)
        if record:
            session['evidence_rules'] = record['rules']
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
        source_contract = CONTRACT
        if config.get('shadow_piano_evidence_memory'):
            source_contract = source_contract.replace(
                'At a first early OOD checkpoint print\nPIANO_STOP OOD_REPROBE and return without finalizing that attempt.',
                'On OOD safely release and finalize that attempt; there is no in-episode regeneration.')
        messages=copy.deepcopy(obs['full_prompt']);messages.append({'role':'user','content':source_contract})
        if config.get('shadow_piano_evidence_memory'):
            messages.append({'role':'user','content':LINKED_CONTROL_CONTRACT})
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
    if config.get('shadow_piano_evidence_memory'):
        # One joint update at the episode boundary, never a second whole-module
        # repair that competes with its rules/functions/parameters snapshot.
        return None
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
            notes.append({'index':e['index'],'stage_steps':counts,'release_verified':e.get('release_verified',False),
                'finalization_source':e.get('finalization_source','policy')})
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
    if config.get('shadow_piano_evidence_memory'):
        return finish_linked_episode(config)
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


# The evidence mode is opt-in and does not migrate or overwrite legacy memories.
LINKED_CONTROL_CONTRACT = '''TACTILE EVIDENCE MODE: use module-level reusable action functions.
The host archives every public contact sample, actual function call and release check.
An episode-boundary updater may revise press rhythm, hold feedback and release/recovery logic.
Fixed descent/press action counts from historical strategies are not task requirements.
Preserve requested hold_seconds, measured-pose release >=0.12s, hover 0.04m, force targets,
force quality handling and all API action bounds. No private verdict is available.
Historical tactile signatures and rules are advisory; validate them with fresh public observations.
Do not modify request, redefine public APIs, or change the frozen estimator. Helpers must be
parameterized by current note geometry, dynamics and duration, not a specific melody.'''


def _linked_root(config):
    return Path(config['shadow_piano_evidence_memory']).expanduser().resolve()


def evidence_policy(config):
    mode = config.get('shadow_piano_evidence_mode', 'joint')
    if mode not in ('none', 'text', 'tactile', 'joint'):
        raise ValueError('Evidence mode must be none, text, tactile or joint')
    return {'mode': mode, 'update': mode != 'none', 'text_memory': mode in ('text', 'joint'),
            'tactile_history': mode in ('tactile', 'joint')}


def _linked_bundle(root, identifier):
    if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{64}', identifier):
        raise ValueError('Invalid evidence bundle ID')
    directory = root / 'bundles' / identifier
    bundle = json.loads((directory / 'bundle.json').read_text(encoding='utf-8'))
    source = (directory / 'policy.py').read_text(encoding='utf-8')
    from capx.integrations.opentac_shadow_online_skills import validate_library, _evidence_hash
    rules = validate_library((directory / 'rules.yaml').read_bytes())
    if digest(source) != bundle['source_version'] or rules != bundle['rules']:
        raise ValueError('Evidence bundle failed content verification')
    if _evidence_hash(bundle) != identifier:
        raise ValueError('Evidence bundle manifest changed')
    return {**bundle, 'source': source, 'bundle_id': identifier}


def _store_linked_bundle(root, source, signature, rules, parameters, parent, evidence, changes):
    import yaml
    from capx.integrations.opentac_shadow_online_skills import validate_library, _evidence_hash, TactileEvidenceIndex
    source = validate_source(source)
    parameters = validate_parameters(parameters)
    raw = yaml.safe_dump(rules, allow_unicode=True).encode('utf-8')
    validate_library(raw)
    bundle = {'schema': 'tactile_evidence_bundle.v1', 'source_version': digest(source),
        'compatibility': signature, 'rules': rules, 'parameters': parameters,
        'parent_bundle': parent, 'evidence': evidence, 'changed_functions': changes}
    identifier = _evidence_hash(bundle)
    directory = root / 'bundles' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'policy.py').write_text(source, encoding='utf-8')
    (directory / 'rules.yaml').write_bytes(raw)
    atomic_json(directory / 'bundle.json', bundle)
    index = TactileEvidenceIndex(root)
    for rule in rules['skills']:
        index.append('rule', {'bundle_id': identifier, **rule}, [identifier, *evidence])
    for group, values in parameters.items():
        index.append('parameter_binding', {'bundle_id': identifier, 'group': group,
            'values': values, 'source_version': digest(source), 'status': 'public_report_not_force_truth'},
            [identifier, *evidence])
    index.append('bundle', {'bundle_id': identifier, **bundle}, [identifier, *evidence, *([parent] if parent else [])])
    return identifier


def linked_initial_record(config, signature):
    root = _linked_root(config)
    config['_linked_input_bundle'] = None
    for name in ('pending_bundle.json', 'active_bundle.json'):
        path = root / name
        if not path.exists():
            continue
        pointer = json.loads(path.read_text(encoding='utf-8'))
        if name == 'pending_bundle.json' and pointer.get('attempted_run'):
            continue
        bundle = _linked_bundle(root, pointer['bundle_id'])
        if bundle['compatibility'] != signature:
            continue
        config['_linked_input_bundle'] = bundle['bundle_id']
        if name == 'pending_bundle.json':
            atomic_json(path, {**pointer, 'attempted_run': Path(config['output_dir']).name})
        return {**bundle, 'version': bundle['source_version'], 'status': 'pending_public_execution'}, name
    return None, 'no_compatible_evidence_bundle'


def _linked_manifest(config):
    root = _linked_root(config)
    pointer = json.loads((Path(config['output_dir']) / 'piano_tactile_evidence.json').read_text(encoding='utf-8'))
    path = Path(pointer['manifest']).resolve()
    if not path.is_relative_to(root / 'episodes') or path.name != 'manifest.json':
        raise ValueError('Evidence manifest must be in the public episode directory')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'public_tactile_evidence.v1' or manifest['episode_id'] != pointer['episode_id']:
        raise ValueError('Invalid public evidence manifest')
    return manifest, path


def _linked_update_input(source, manifest, rules, parameters, index=None):
    """Whitelist and bound the prompt; no private files, scores, or duplicated module."""
    functions = extract_action_functions(source)
    names = {f['name']: f for f in functions if sum(g['name'] == f['name'] for g in functions) == 1}
    executed = {f['function_id']: f['name'] for f in manifest['functions']}
    calls = [i for i in manifest['invocations'] if executed.get(i['function_id']) in names]
    contact_calls = {i for c in manifest['clips'] for i in c['invocation_ids']}
    # Prefer functions with exceptions, then contact/release helpers actually run.
    ranked = sorted(names, key=lambda name: (
        -sum(bool(i['exceptions']) for i in calls if executed[i['function_id']] == name),
        -int(any(s in name.lower() for s in ('press', 'hold', 'release'))), name))
    selected = []
    for name in ranked:
        f = names[name]
        if any(executed[i['function_id']] == name and i['invocation_id'] in contact_calls for i in calls) and not any(
                f['line'] <= g['line'] <= f['end_line'] or g['line'] <= f['line'] <= g['end_line'] for g in selected):
            selected.append(f)
        if len(selected) == 2:
            break
    if not selected:
        raise ValueError('No executed reusable helper has public evidence; retain the experience')
    ids = {i['invocation_id'] for i in calls if executed[i['function_id']] in {f['name'] for f in selected}}
    eligible = [c for c in manifest['clips'] if ids.intersection(c['invocation_ids'])]
    if not eligible:
        raise ValueError('No tactile clip is associated with the executed helpers')
    by_clip = {c['clip_id']: [] for c in eligible}
    for s in manifest['samples']:
        if s['clip_id'] in by_clip:
            by_clip[s['clip_id']].append(s)
    # Include phase diversity and quality failures; never select using native grades.
    chosen = []
    for phase in ('press', 'hold', 'release'):
        options = [c for c in eligible if c['phase'] == phase]
        if options:
            chosen.append(max(options, key=lambda c: sum(s['estimate']['quality'] != 'ok' for s in by_clip[c['clip_id']])))
    if not chosen:
        chosen = eligible[:3]
    clips = []
    for clip in chosen:
        samples = by_clip[clip['clip_id']]
        # Bound prompt sampling only; the on-disk trace remains complete.
        positions = sorted(set(np.linspace(0, len(samples)-1, min(12, len(samples)), dtype=int).tolist()))
        clips.append({k: clip[k] for k in ('clip_id', 'phase', 'attempt', 'note', 'dynamics',
            'start_physics_step', 'end_physics_step', 'phase_basis')} | {'samples': [
                {k: samples[i][k] for k in ('sample_id', 'action_step', 'physics_step', 'timestamp_s',
                    'instantaneous_5d', 'estimate', 'action', 'hand')} for i in positions]})
    result = {'schema': 'public_tactile_update_input.v1', 'episode_id': manifest['episode_id'],
        'request': {k: manifest['request'][k] for k in ('notes', 'dynamics', 'hold_seconds')},
        'rules': rules, 'parameters': parameters,
        'functions': [{k: f[k] for k in ('name', 'qualified_name', 'signature', 'code', 'dependencies', 'captured_context')} for f in selected],
        'dependency_signatures': [{k: f[k] for k in ('name', 'signature')} for f in functions if f not in selected],
        'clips': clips,
        'public_events': [{k: e[k] for k in ('kind', 'action_step', 'physics_step', 'index',
            'release_verified', 'continuous_release_seconds', 'has_exception',
            'finalization_source', 'finalized_indices') if k in e}
            for e in manifest['events'] if e['kind'] in ('attempt_finalized', 'release_rejected',
                'code_execution', 'native_terminal_end')][-12:],
        'invocations': [{k: i[k] for k in ('invocation_id', 'function_id', 'start', 'end', 'exceptions')}
            for i in calls if i['invocation_id'] in ids][-8:]}
    result['historical_clips'] = []
    result['historical_functions'] = {}
    if index is not None:
        # Retrieve actual trajectories, not just an opaque embedding or a label.
        # Resolve actual invoked functions. Vectors select evidence, never become
        # language advice. Store each historical source once, bounded below.
        records = index.records()
        seen = set()
        for retrieval in reversed(manifest.get('retrievals', [])):
            for match in retrieval['matches']:
                if match['clip_id'] in seen or len(seen) >= 2:
                    continue
                candidates = [r['payload'] for r in records if r['kind'] == 'clip' and r['payload']['clip_id'] == match['clip_id']]
                if not candidates:
                    continue
                historical = candidates[-1]
                path = Path(historical['manifest']).resolve()
                if not path.is_relative_to(index.root / 'episodes') or path.name != 'manifest.json':
                    continue
                prior = json.loads(path.read_text(encoding='utf-8'))
                rows = [s for s in prior['samples'] if s['clip_id'] == match['clip_id']]
                if not rows:
                    continue
                linked = index.resolve_skill_evidence(match['clip_id'], records=records)
                if not linked or not linked['functions']:
                    continue
                seen.add(match['clip_id'])
                function_refs = []
                for f in sorted(linked['functions'], key=lambda f: (f['name'] == 'run_piano', len(f['source']))):
                    if f['source_hash'] not in result['historical_functions']:
                        if len(result['historical_functions']) >= 2:
                            continue
                        current = next((g['name'] for g in selected if g['code'].strip() == f['source'].strip()), None)
                        result['historical_functions'][f['source_hash']] = {
                            k: f[k] for k in ('function_id', 'name', 'source_hash', 'ast_hash')}
                        if current:
                            result['historical_functions'][f['source_hash']]['same_as_current_function'] = current
                        else:
                            result['historical_functions'][f['source_hash']]['source'] = f['source']
                    function_refs.append(f['source_hash'])
                positions = sorted(set(np.linspace(0, len(rows)-1, min(6, len(rows)), dtype=int).tolist()))
                result['historical_clips'].append({
                    k: historical[k] for k in ('clip_id', 'phase', 'attempt', 'note', 'dynamics',
                        'start_physics_step', 'end_physics_step', 'phase_basis')} | {
                    'retrieval_type': retrieval['type'], 'distance': match['distance'],
                    'encoder': retrieval.get('encoder'), 'function_refs': function_refs,
                    'invocations': [i for i in linked['invocations'] if i['function_id'] in
                        {f['function_id'] for f in result['historical_functions'].values()}][-4:],
                    'samples': [{k: rows[i][k] for k in ('sample_id', 'action_step', 'physics_step',
                        'timestamp_s', 'instantaneous_5d', 'estimate', 'action', 'hand')} for i in positions],
                    'association_ids': [r['id'] for r in records if match['clip_id'] in r['links']
                                        and r['kind'] in ('application', 'update', 'rule', 'parameter_binding')][-8:]})
    while len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > 24 * 1024:
        longest = max(result['clips'] + result['historical_clips'], key=lambda c: len(c['samples']))
        if len(longest['samples']) <= 2:
            raise ValueError('Evidence cannot fit the 24 KiB update budget without losing its source')
        longest['samples'].pop(1)
    # Defense in depth; actual confidentiality is the field/file whitelist above.
    if any(m in json.dumps(result).lower() for m in _PRIVATE_MARKERS):
        raise ValueError('Private audit marker in public update input')
    return result


def _protected_expressions(source):
    """Freeze named force/hold/hover bindings, including dictionaries and comparisons.

    This is a conservative patch validator, not a proof about arbitrary Python.
    The adapter remains the authority for motion/release guards.
    """
    protected = re.compile(r'(target_force|tolerance|hold_gate|hold_seconds|force_target|force_bounds|hover|calibration|estimator)')
    tree = ast.parse(source)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(protected.search(ast.unparse(t)) for t in targets):
                result.append(ast.dump(node))
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and protected.search(key.value):
                    result.append(ast.dump(key) + ':' + ast.dump(value))
    return sorted(result)


def validate_linked_proposal(proposal, source, evidence):
    import yaml
    from capx.integrations.opentac_shadow_online_skills import validate_library
    if not isinstance(proposal, dict) or set(proposal) != {'functions', 'rules', 'rule_evidence'}:
        raise ValueError('Update requires functions/rules/rule_evidence')
    if not evidence.get('memory_policy', {}).get('text_memory', True) and (proposal['rules'] or proposal['rule_evidence']):
        raise ValueError('This arm cannot retain a text experience library')
    patches = proposal['functions']
    if not isinstance(patches, list) or len(patches) > 2:
        raise ValueError('At most two local functions may change')
    offered = {f['qualified_name']: f for f in evidence['functions']}
    available = {c['clip_id']: {s['physics_step'] for s in c['samples']}
                 for c in evidence['clips'] + evidence.get('historical_clips', [])}
    def check_refs(row):
        if row.get('basis') not in ('observed', 'hypothesis'):
            raise ValueError('Changes need an evidence basis')
        if not all(isinstance(row.get(k), str) and 1 <= len(row[k]) <= 600 for k in ('reason', 'expected_effect')):
            raise ValueError('Changes need reason and expected public effect')
        if not isinstance(row.get('evidence'), list) or not row['evidence']:
            raise ValueError('Changes must cite tactile clips and sampled physical steps')
        for ref in row['evidence']:
            if set(ref) != {'clip_id', 'physics_steps'} or ref['clip_id'] not in available:
                raise ValueError('Unknown tactile evidence')
            if not ref['physics_steps'] or any(type(s) is not int or s not in available[ref['clip_id']] for s in ref['physics_steps']):
                raise ValueError('Unavailable physical evidence step')
    tree = ast.parse(source)
    nodes = {}
    def visit(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                qualified = '.'.join((*parents, child.name))
                nodes[qualified] = child
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)
    visit(tree)
    edits, changes, seen = [], [], set()
    for patch in patches:
        if set(patch) != {'qualified_name', 'source', 'evidence', 'basis', 'reason', 'expected_effect'}:
            raise ValueError('Unsupported function patch fields')
        name = patch['qualified_name']
        if name not in offered or name in seen:
            raise ValueError('Patch must target a distinct offered function')
        seen.add(name)
        check_refs(patch)
        replacement = ast.parse(patch['source'])
        if len(replacement.body) != 1 or not isinstance(replacement.body[0], ast.FunctionDef):
            raise ValueError('Patch must be exactly one function definition')
        old, new = nodes[name], replacement.body[0]
        if (new.name != old.name or ast.dump(new.args) != ast.dump(old.args) or new.decorator_list
                or ast.dump(new.returns or ast.Constant(None)) != ast.dump(old.returns or ast.Constant(None))):
            raise ValueError('Function signature and binding must stay compatible')
        if _protected_expressions(offered[name]['code']) != _protected_expressions(patch['source']):
            raise ValueError('Force/hold/hover constraints are not patchable')
        # Reject API shadowing and request writes even inside nested helpers.
        for part in ast.walk(new):
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store) and part.id in PUBLIC_FUNCTIONS:
                raise ValueError('Public API bindings are immutable')
            if isinstance(part, ast.Subscript) and isinstance(part.ctx, ast.Store) and ast.unparse(part.value).startswith('request'):
                raise ValueError('The public request is immutable')
        indent = ' ' * old.col_offset
        replacement_text = textwrap.indent(patch['source'].strip(), indent) + '\n'
        edits.append((old.lineno - 1, old.end_lineno, replacement_text))
        def structure(node):
            value = copy.deepcopy(node)
            for item in ast.walk(value):
                if isinstance(item, ast.Constant):
                    item.value = type(item.value).__name__
            return ast.dump(value)
        changes.append({'qualified_name': name, 'old_hash': digest(ast.dump(old)),
            'new_hash': digest(ast.dump(new)), 'ast_structure_changed': structure(old) != structure(new),
            'evidence': patch['evidence'], 'basis': patch['basis'], 'reason': patch['reason'],
            'expected_effect': patch['expected_effect']})
    edits.sort(reverse=True)
    for upper, lower in zip(edits, edits[1:]):
        if lower[1] > upper[0]:
            raise ValueError('Overlapping nested function patches')
    lines = source.splitlines(keepends=True)
    for start, end, body in edits:
        lines[start:end] = [body]
    assembled = validate_source(''.join(lines))
    old_functions = {f['qualified_name']: f for f in extract_action_functions(source)}
    for function in extract_action_functions(assembled):
        old_function = old_functions.get(function['qualified_name'])
        allowed = set(old_function['captured_context']) if old_function else set()
        if set(function['captured_context']) - allowed:
            raise ValueError('Patch introduced undefined or nonpublic dependencies')
    rules = {'schema': 'shadow_piano_public_skills.v1', 'skills': proposal['rules']}
    validate_library(yaml.safe_dump(rules, allow_unicode=True).encode('utf-8'))
    if not isinstance(proposal['rule_evidence'], list):
        raise ValueError('Rule evidence must be a list')
    rule_refs = {}
    for ref in proposal['rule_evidence']:
        if set(ref) != {'rule_id', 'evidence', 'basis', 'reason', 'expected_effect'}:
            raise ValueError('Invalid rule evidence')
        check_refs(ref)
        rule_refs[ref['rule_id']] = ref
    old_rules = {r['id']: r for r in evidence['rules']['skills']}
    new_rules = {r['id']: r for r in rules['skills']}
    for identifier in old_rules.keys() | new_rules.keys():
        if old_rules.get(identifier) != new_rules.get(identifier) and identifier not in rule_refs:
            raise ValueError('Every rule change or removal needs evidence')
    return assembled, rules, changes


def finish_linked_episode(config, query_args=None):
    """One public-only joint update; pending code never overwrites checked code."""
    from capx.integrations.opentac_shadow_online_skills import TactileEvidenceIndex, _evidence_hash
    from capx.llm.client import ModelQueryArgs, collect_llm_usage, summarize_llm_usage
    run = Path(config['output_dir'])
    result_path = run / 'piano_evidence_update_result.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    output = run / 'tactile_evidence_update'
    output.mkdir(parents=True, exist_ok=True)
    # Claim once before the request; a crash is archived, never silently replayed.
    claim = output / 'started.json'
    if claim.exists():
        return {'status': 'interrupted_previous_update', 'accepted': False}
    atomic_json(claim, {'schema': 'tactile_evidence_update.v1'})
    result = {'accepted': False, 'status': 'started', 'official_result_used': False}
    policy = evidence_policy(config)
    result['memory_policy'] = policy
    usage = []
    if not policy['update']:
        result.update(status='disabled_no_memory', update_usage=summarize_llm_usage([]))
        atomic_json(result_path, result)
        atomic_json(output / 'validation.json', result)
        return result
    try:
        root = _linked_root(config)
        index = TactileEvidenceIndex(root)
        session = json.loads(_session_path(config).read_text(encoding='utf-8'))
        source = validate_source(session['source'])
        manifest, manifest_path = _linked_manifest(config)
        summary = public_summary(run)
        parent = config.get('_linked_input_bundle')
        # Config may be reconstructed by a caller after the trial.
        if parent is None:
            parent = session.get('evidence_input_bundle')
        bundle = _linked_bundle(root, parent) if parent else None
        rules = bundle['rules'] if bundle and policy['text_memory'] else {'schema': 'shadow_piano_public_skills.v1', 'skills': []}
        parameters = validate_parameters(summary['parameters'])
        executed_hashes = {f['ast_hash'] for f in manifest['functions'] if any(
            i['function_id'] == f['function_id'] and i['end'] is not None for i in manifest['invocations'])}
        changed_called = not bundle or all(c['new_hash'] in executed_hashes for c in bundle['changed_functions'])
        checked = (summary['public_checks_complete'] and not summary['failures'] and
                   not summary['control_failures'] and changed_called and bool(manifest['samples']) and
                   not any(i['exceptions'] for i in manifest['invocations']))
        evidence_ids = [c['clip_id'] for c in manifest['clips']]
        executed_bundle = parent or _store_linked_bundle(root, source, session['compatibility'], rules,
            parameters, None, evidence_ids, [])
        result.update(input_bundle=parent, executed_bundle=executed_bundle,
            executed_version=digest(source), public_execution_checked=bool(checked),
            manifest_sha256=digest(manifest_path.read_bytes()))
        index.append('application', {'bundle_id': executed_bundle, 'episode_id': manifest['episode_id'],
            'source_version': digest(source), 'public_execution_checked': bool(checked),
            'changed_functions_called': bool(changed_called), 'manifest_sha256': result['manifest_sha256']},
            [executed_bundle, *evidence_ids, *(i['invocation_id'] for i in manifest['invocations'])])
        if checked:
            atomic_json(root / 'active_bundle.json', {'bundle_id': executed_bundle,
                'status': 'public_execution_checked', 'evidence_episode': manifest['episode_id']})
        evidence = _linked_update_input(source, manifest, rules, parameters,
            index=index if policy['tactile_history'] else None)
        evidence['memory_policy'] = policy
        if len(json.dumps(evidence, ensure_ascii=False).encode('utf-8')) > 24 * 1024:
            raise ValueError('Evidence including ablation metadata exceeds 24 KiB')
        atomic_json(output / 'public_input.json', evidence)
        instruction = (LINKED_CONTROL_CONTRACT + '\nReturn ONLY JSON with functions, rules, rule_evidence. '
            'functions is a list of at most two replacements: {qualified_name,source,evidence,basis,reason,expected_effect}. '
            'source is one complete function with unchanged name/signature. Evidence is [{clip_id,physics_steps:[...]}]. '
            'basis is observed or hypothesis. Each rule_evidence item has rule_id,evidence,basis,reason,expected_effect. '
            'rules is the COMPLETE next text library, each item id/trigger/practice, max16, YAML<=8KiB; '
            'trigger<=240 characters, practice<=600. All changes/removals require references. '
            'Preserve fixed force/hold/hover bindings; do not change targets, estimator, native scoring or API limits. '
            'Use tactile changes across press/hold/release to propose reusable LOGIC changes, not merely constants. '
            'If evidence is insufficient return unchanged rules and functions=[]. Improvements are hypotheses '
            'until fresh public execution checks. Do not claim official success. Only listed functions may change.')
        if not policy['text_memory']:
            instruction += ' This arm has no persistent text library: return rules=[] and rule_evidence=[]. Keep rationales only in function proposal evidence.'
        if not policy['tactile_history']:
            instruction += ' This arm has no historical tactile retrieval. Current-episode tactile feedback is available equally to all updating arms.'
        messages = [{'role': 'system', 'content': instruction},
                    {'role': 'user', 'content': json.dumps(evidence, ensure_ascii=False)}]
        atomic_json(output / 'prompt.json', {'messages': messages})
        if query_args is None:
            base = os.environ['CAPX_API_BASE_URL'].rstrip('/')
            query_args = ModelQueryArgs(model=os.environ.get('MODEL', os.environ.get('CAPX_MODEL', 'gpt-6.1-sol')),
                server_url=base if base.endswith('/responses') else base + '/responses',
                api_key=os.environ['CAPX_API_KEY'], max_tokens=8192, reasoning_effort='low')
        # Process is serial; restore environment after the isolated API request.
        proxy_keys = ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy']
        saved = {k: os.environ.get(k) for k in proxy_keys + ['CAPX_DISABLE_PROXY', 'CAPX_LLM_TRUST_ENV']}
        try:
            for key in proxy_keys:
                os.environ.pop(key, None)
            os.environ['CAPX_DISABLE_PROXY'] = '1'
            os.environ['CAPX_LLM_TRUST_ENV'] = '0'
            with collect_llm_usage(usage, 'tactile_evidence_update'):
                response = _query(query_args, messages, deadline_seconds=300, phase='tactile_evidence_update')
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        (output / 'response.txt').write_text(response['content'], encoding='utf-8')
        proposal = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', response['content'].strip()))
        atomic_json(output / 'proposal.json', proposal)
        updated, next_rules, changes = validate_linked_proposal(proposal, source, evidence)
        if updated == source and next_rules == rules:
            result.update(status='no_change', accepted=True, output_bundle=executed_bundle)
        else:
            candidate = _store_linked_bundle(root, updated, session['compatibility'], next_rules, parameters,
                executed_bundle, evidence_ids, changes)
            proposal_id = index.append('update', {'parent_bundle': executed_bundle, 'candidate_bundle': candidate,
                'changes': changes, 'rule_evidence': proposal['rule_evidence'], 'status': 'syntax_checked_pending_execution',
                'proposal_sha256': _evidence_hash(proposal)}, [executed_bundle, candidate, *evidence_ids])
            atomic_json(root / 'pending_bundle.json', {'bundle_id': candidate, 'attempted_run': None,
                'status': 'syntax_checked_pending_execution'})
            result.update(status='candidate', accepted=True, candidate_bundle=candidate, update_id=proposal_id,
                logic_change_proposed=any(c['ast_structure_changed'] for c in changes))
    except Exception as exc:
        result.update(status='rejected_or_unavailable', error=f'{type(exc).__name__}: {exc}')
    result['update_usage'] = summarize_llm_usage(usage)
    atomic_json(result_path, result)
    atomic_json(output / 'validation.json', result)
    return result
