"""Local request deadlines that HTTP libraries cannot relabel as socket timeouts."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import signal
import time


class LLMRequestTimeout(RuntimeError):
    """A model-request timeout, distinct from the simulator/trial deadline.

    Do not inherit OSError/TimeoutError: requests/urllib3 wrap those exceptions
    raised by SIGALRM and misleadingly attach their configured read timeout.
    """
    def __init__(self, *, source, phase, timeout_seconds, elapsed_seconds):
        self.source = source
        self.phase = phase
        self.timeout_seconds = float(timeout_seconds)
        self.elapsed_seconds = float(elapsed_seconds)
        super().__init__(f'LLM request timeout source={source} phase={phase} '
            f'limit_seconds={self.timeout_seconds:g} elapsed_seconds={self.elapsed_seconds:.3f}')


_active_deadline = ContextVar('capx_llm_active_deadline', default=None)


def deadline_metadata():
    active = _active_deadline.get()
    return dict(active) if active else {}


@contextmanager
def request_deadline(seconds, *, phase):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('LLM request deadline must be finite and positive')
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    def expired(_signum, _frame):
        raise LLMRequestTimeout(source='local_deadline', phase=phase,
            timeout_seconds=seconds, elapsed_seconds=time.monotonic()-started)
    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    token = _active_deadline.set({'phase':phase, 'local_deadline_seconds':seconds})
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        _active_deadline.reset(token)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL,
                max(.001, previous_timer[0]-(time.monotonic()-started)), previous_timer[1])
