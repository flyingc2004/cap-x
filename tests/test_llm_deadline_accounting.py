"""Local HTTP reproduction and saved provider accounting; no external API/GPU."""
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from capx.llm import client
from capx.llm.deadline import LLMRequestTimeout, request_deadline
from capx.llm.usage_audit import audit_usage, summarize_accounting


@pytest.fixture
def slow_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            time.sleep(.2)
            try:
                self.send_response(200);self.end_headers();self.wfile.write(b'{}')
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args):
            pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1/responses'
    server.shutdown();server.server_close();thread.join(timeout=1)


def test_local_deadline_is_not_relabelled_or_retried_by_requests(slow_server,monkeypatch):
    monkeypatch.setenv('CAPX_DISABLE_PROXY','1')
    monkeypatch.setenv('CAPX_LLM_PROTOCOL','responses')
    # Even a caller enabling retries must not retry an exhausted local budget.
    monkeypatch.setenv('CAPX_LLM_MAX_RETRIES','3')
    args=client.ModelQueryArgs(model='test',server_url=slow_server,
                               request_timeout_seconds=300)
    events=[];previous=signal.getsignal(signal.SIGALRM)
    with pytest.raises(LLMRequestTimeout) as caught:
        with client.collect_llm_usage(events,'multi_turn'):
            with request_deadline(.05,phase='code_memory_repair'):
                client.query_model(args,[{'role':'user','content':'test'}])
    assert caught.value.source=='local_deadline' and caught.value.timeout_seconds==.05
    assert caught.value.elapsed_seconds<1 and 'read timeout=300' not in str(caught.value)
    assert signal.getsignal(signal.SIGALRM)==previous
    assert len(events)==1 and events[0]['error']['source']=='local_deadline'
    assert events[0]['request']['http_timeout_seconds']==300
    assert events[0]['request']['local_deadline_seconds']==.05
    assert events[0]['usage']['total_tokens'] is None


def test_transport_timeout_has_separate_source_and_effective_budget(slow_server,monkeypatch):
    monkeypatch.setenv('CAPX_DISABLE_PROXY','1')
    monkeypatch.setenv('CAPX_LLM_PROTOCOL','responses')
    monkeypatch.setenv('CAPX_LLM_MAX_RETRIES','0')
    monkeypatch.setenv('CAPX_LLM_TIMEOUT_SECONDS','300')
    args=client.ModelQueryArgs(model='test',server_url=slow_server,request_timeout_seconds=.03)
    events=[]
    with pytest.raises(LLMRequestTimeout) as caught:
        with client.collect_llm_usage(events,'initial_code'):
            client.query_model(args,[{'role':'user','content':'test'}])
    assert caught.value.source=='transport_timeout' and caught.value.timeout_seconds==.03
    assert events[0]['request']['http_timeout_seconds']==.03
    assert events[0]['error']['phase']=='initial_code'


def test_deadline_restores_existing_timer_and_handler():
    previous=signal.getsignal(signal.SIGALRM);timer=signal.setitimer(signal.ITIMER_REAL,0)
    def handler(*_):
        pytest.fail('Previous timer should not fire during this test')
    signal.signal(signal.SIGALRM,handler);signal.setitimer(signal.ITIMER_REAL,3)
    try:
        with request_deadline(1,phase='outer'):
            with request_deadline(.5,phase='inner'):
                pass
        assert signal.getsignal(signal.SIGALRM)==handler
        assert 2<signal.getitimer(signal.ITIMER_REAL)[0]<=3
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        if timer[0]:signal.setitimer(signal.ITIMER_REAL,*timer)


def test_accounting_keeps_both_reports_and_never_divides_by_observed_ratio():
    raw={'input_tokens':16853,'output_tokens':4864,'total_tokens':21717,
         'attribution':{'items':{
             'input':{'input_tokens':10533,'output_tokens':0,
                      'content':[{'input_tokens':10533,'output_tokens':0}]},
             'output':{'input_tokens':0,'output_tokens':3040}}}}
    normalized=client.normalize_llm_usage({'usage':raw})
    audit=audit_usage(normalized,protocol='responses',output_limit=8192)
    assert audit['reported_totals']['total_tokens']==21717
    assert audit['attribution_totals']['total_tokens']==13573
    assert audit['status']=='needs_review' and audit['comparable_totals']['total_tokens'] is None
    assert normalized['provider_usage']==raw
    assert audit['independently_verified'] is False


def test_accounting_includes_instruction_fields_without_double_counting_content():
    raw={'input_tokens':12,'output_tokens':4,'total_tokens':16,
         'attribution':{'items':{'msg':{'input_tokens':7,'output_tokens':4,
             'content':[{'input_tokens':7,'output_tokens':4}]}},
             'request_fields':{'instructions':{'input_tokens':5,'output_tokens':0}}}}
    audit=audit_usage(client.normalize_llm_usage({'usage':raw}))
    assert audit['attribution_totals']['total_tokens']==16
    assert audit['status']=='provider_reported_unverified'
    assert audit['comparable_totals']['total_tokens']==16


def test_incomplete_details_are_not_assumed_to_be_zero_or_actual_usage():
    raw={'input_tokens':10,'output_tokens':5,'total_tokens':15,
         'attribution':{'items':{'msg':{'input_tokens':10}}}}
    audit=audit_usage(client.normalize_llm_usage({'usage':raw}))
    assert audit['attribution_totals']['output_tokens'] is None
    assert audit['independently_verified'] is False


def test_failed_request_preserves_known_subtotal_but_total_is_unknown():
    good={'protocol':'responses','usage':client.normalize_llm_usage({'usage':{
        'input_tokens':10,'output_tokens':5,'total_tokens':15}})}
    failed={'protocol':'responses','usage':client.normalize_llm_usage({})}
    summary=client.summarize_llm_usage([good,failed])
    assert summary['query_count']==2 and summary['unreported_query_count']==1
    assert summary['totals']['total_tokens'] is None
    assert summary['known_reported_totals']['total_tokens']==15
    assert summary['accounting']['comparable_totals']['total_tokens'] is None
    assert summarize_accounting([])['comparable_totals']['total_tokens']==0


def test_latency_reports_zero_unknown_failures_and_request_phases_separately():
    assert client.summarize_llm_latency([])['sum_seconds']==0
    rows=[{'phase':'initial_code','latency_seconds':2.},
        {'phase':'multi_turn','request':{'phase':'code_memory_repair'},'request_wall_seconds':3.,
         'error':{'source':'local_deadline'}},
        {'phase':'skill_reflection'}]
    timing=client.summarize_llm_latency(rows)
    assert timing['sum_seconds'] is None and timing['known_sum_seconds']==5.
    assert timing['by_phase']['code_memory_repair']['sum_seconds']==3.
    assert timing['queries'][1]['failed']
    assert client.summarize_llm_usage(rows)['timing']==timing


@pytest.mark.parametrize('source',['local_deadline','transport_timeout'])
def test_trial_runner_saves_model_timeout_without_trial_retry(monkeypatch,source):
    from types import SimpleNamespace
    from capx.envs import runner
    exc=LLMRequestTimeout(source=source,phase='code_memory_repair',timeout_seconds=300,elapsed_seconds=300.1)
    def run(*args,**kwargs):
        raise exc
    captured={};sentinel=object()
    def save(*args,**kwargs):
        captured.update(kwargs)
        assert args[4] is exc
        return sentinel
    monkeypatch.setattr(runner,'_run_single_trial',run)
    monkeypatch.setattr(runner,'_build_aborted_summary',save)
    result=runner._run_single_trial_with_timeout(SimpleNamespace(),1,SimpleNamespace(),{},None,
                                                timeout_s=0,raise_on_timeout=True)
    assert result is sentinel
    assert captured['error_label']=='LLM Request Timeout'
    assert source in captured['prefix'] and '1800' not in captured['prefix']
