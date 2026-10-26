"""The parent Responses wrapper must preserve explicit experiment models."""
import os
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('overrides,expected', [
    ({'MODEL': 'gpt-6.1-sol', 'CAPX_MODEL': 'gpt-6-sol'}, 'gpt-6.1-sol'),
    ({'CAPX_MODEL': 'gpt-6.1-sol'}, 'gpt-6.1-sol'),
    ({}, 'gpt-6-sol'),
])
def test_explicit_model_survives_secrets_loading(tmp_path, overrides, expected):
    wrapper = Path(__file__).resolve().parents[2] / 'run_responses'
    if not wrapper.exists():
        pytest.skip('Parent t-cap wrapper is not present in this checkout')
    shutil.copy2(wrapper, tmp_path / 'run_responses')
    (tmp_path / '.capx_secrets.env').write_text(
        'MODEL=gpt-6-sol\nCAPX_MODEL=gpt-6-sol\nCAPX_API_BASE_URL=https://unused.invalid/v1\n')
    run = tmp_path / 'run'
    run.write_text('#!/usr/bin/env bash\nprintf "%s" "$MODEL"\n')
    run.chmod(0o700)
    result = subprocess.run(['bash', str(tmp_path / 'run_responses')], check=True,
        capture_output=True, text=True, env={'PATH': os.environ['PATH'], **overrides})
    assert result.stdout == expected


@pytest.mark.parametrize('config_name',[
    'play_piano_shadow_digit_code_memory.yaml',
    'play_piano_shadow_digit_abc_a_fresh.yaml',
    'play_piano_shadow_digit_abc_b_frozen.yaml',
    'play_piano_shadow_digit_abc_c_adaptive.yaml'])
def test_complete_shadow_shell_chain_preserves_model_and_disables_proxy(tmp_path,config_name):
    parent = Path(__file__).resolve().parents[2]
    if not (parent / 'run_responses').exists():
        pytest.skip('Parent t-cap is not present')
    for name in ('run_responses', 'run', 'capx_scripts/capx_env.sh',
                 'capx_scripts/run_capx_shadow_piano.sh'):
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    (tmp_path / '.capx_secrets.env').write_text(
        'export CAPX_MODEL=gpt-6-sol\n'
        'export CAPX_API_BASE_URL=https://unused.invalid/v1\n'
        'export HTTP_PROXY=http://unused.invalid:99\n')
    capx = tmp_path / 'cap-x'; cfg = capx / 'env_configs/opentac' / config_name
    cfg.parent.mkdir(parents=True)
    cfg.write_text('simulator: capx.envs.simulators.opentac_shadow.ShadowPianoLowLevelEnv\n')
    native = tmp_path / 'native'; task = native / 'env/tasks/arm_shadow_piano.py'
    task.parent.mkdir(parents=True); task.write_text('')
    renderer = native / '.cache/piano_tactile/digit/venv/bin/python'
    renderer.parent.mkdir(parents=True); renderer.symlink_to('/bin/true')
    prefix = tmp_path / 'conda'; bindir = prefix / 'bin'; bindir.mkdir(parents=True)
    conda_sh = tmp_path / 'conda.sh'; conda_sh.write_text('conda() { export CONDA_PREFIX="$2"; }\n')
    capture = tmp_path / 'kit_arguments.json'
    fake_python = bindir / 'python'
    fake_python.write_text('#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\n'
        'Path(os.environ["MODEL_CAPTURE_PATH"]).write_text(json.dumps({"argv":sys.argv,"model":os.environ.get("CAPX_MODEL"),'
        '"trust_env":os.environ.get("CAPX_LLM_TRUST_ENV"),"proxies":[k for k in os.environ if k.lower() in ("http_proxy","https_proxy","all_proxy","no_proxy")]}))\n'
        'raise SystemExit(7)\n')
    fake_python.chmod(0o700)
    env = {'PATH': os.environ['PATH'], 'MODEL': 'gpt-6.1-sol', 'CAPX_MODEL': 'gpt-6.1-sol',
        'CAPX_DIR': str(capx), 'UNIVTAC_ROOT': str(native), 'CONDA_SH': str(conda_sh),
        'CONDA_ENV_NAME': str(prefix), 'CAPX_DISABLE_PROXY': '1', 'GPU': '0',
        'CAPX_CACHE_ROOT': str(tmp_path / 'cache'), 'CAPX_RUN_ROOT': str(tmp_path / 'runs'),
        'RUN_ROOT': str(tmp_path / 'runs'), 'TMPDIR': str(tmp_path / 'tmp'),
        'UNIVTAC_HOME': str(tmp_path / 'kit-home'), 'UNIVTAC_XDG_CACHE_HOME': str(tmp_path / 'kit-cache'),
        'UNIVTAC_XDG_DATA_HOME': str(tmp_path / 'kit-data'), 'UNIVTAC_XDG_CONFIG_HOME': str(tmp_path / 'kit-config'),
        'CONFIG_PATH': 'env_configs/opentac/'+config_name,
        'LLM_PREFLIGHT': '0', 'PROCESS_TIMEOUT_SECONDS': '10', 'MODEL_CAPTURE_PATH': str(capture)}
    result = subprocess.run(['bash', str(tmp_path / 'run_responses')], env=env,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 7, result.stdout + result.stderr
    observed = json.loads(capture.read_text())
    assert observed['model'] == 'gpt-6.1-sol'
    assert observed['argv'][observed['argv'].index('--model')+1] == 'gpt-6.1-sol'
    assert observed['argv'][observed['argv'].index('--config-path')+1] == str(cfg)
    assert observed['proxies'] == [] and observed['trust_env'] == '0'
