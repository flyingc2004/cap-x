"""Task-level exceptions with non-sandbox semantics."""

from __future__ import annotations

from typing import Any


class RecoverableTaskFailure(RuntimeError):
    """Expected task failure that should not count as a code execution error."""

    def __init__(
        self,
        reason: str,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason)
        self.details = dict(details or {})
        super().__init__(message or self.reason)


class HardStopTrial(TimeoutError):
    """Trial-level stop that must escape generated-code execution."""

    def __init__(
        self,
        reason: str,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason)
        self.details = dict(details or {})
        super().__init__(message or self.reason)


class CodeBlockTimeout(BaseException):
    """Watchdog interrupt that generated ``except Exception`` must not swallow."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        self.details = dict(details)
        super().__init__(message)
