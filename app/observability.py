"""仅记录受控元数据；日志故障不改变业务结果。"""

from asyncio import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
from time import perf_counter


logger = logging.getLogger("rag")


@dataclass
class RequestTrace:
    request_id: str
    status: str = "ok"
    exception_type: str | None = None


request_trace: ContextVar[RequestTrace | None] = ContextVar("rag_request_trace", default=None)


class SafeStreamHandler(logging.StreamHandler):
    def handleError(self, record) -> None:
        # logging 默认会向 stderr 打印 formatter/sink 异常及堆栈；这里禁止该旁路。
        pass


def configure_logging() -> None:
    """只配置 rag logger；不更改 root、Uvicorn 或 SDK 的日志级别。"""
    try:
        if not any(getattr(handler, "_rag_json", False) for handler in logger.handlers):
            handler = SafeStreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler._rag_json = True
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception:
        pass  # 日志初始化失败也不能阻止服务启动。


def log_event(event: str, stage: str, status: str, *, duration_ms: float | None = None,
              error: BaseException | None = None, http_status: int | None = None,
              trace: RequestTrace | None = None) -> None:
    try:
        trace = trace or request_trace.get()
        exception_type = None
        if error is not None:
            name = type(error).__name__
            exception_type = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name) else "Exception"
        if trace is not None:
            if status == "failed" or (status == "cancelled" and trace.status != "failed"):
                trace.status = status
                trace.exception_type = trace.exception_type or exception_type
            elif status == "no_evidence" and trace.status == "ok":
                trace.status = status
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": trace.request_id if trace else None,
            "event": event, "stage": stage, "status": status,
        }
        if duration_ms is not None:
            payload["duration_ms"] = round(duration_ms, 3)
        if exception_type or (stage == "http" and trace and trace.exception_type):
            payload["exception_type"] = exception_type or trace.exception_type
        if http_status is not None:
            payload["http_status"] = http_status
        # 不使用异常消息、exc_info、请求正文、Prompt、答案或外部任意字段。
        logger.log(logging.ERROR if status == "failed" else logging.INFO,
                   json.dumps(payload, ensure_ascii=True, allow_nan=False))
    except Exception:
        pass  # formatter、handler 或日志输出故障均不影响原返回值/异常。


def failure_status(error: BaseException) -> str:
    return "cancelled" if isinstance(error, (GeneratorExit, CancelledError)) or type(error).__name__ == "ClientDisconnect" else "failed"


@dataclass
class StageState:
    status: str = "ok"


@contextmanager
def stage(name: str):
    trace = request_trace.get()
    started = perf_counter()
    state = StageState()
    log_event("stage_started", name, "started", trace=trace)
    try:
        yield state
    except BaseException as error:
        status = failure_status(error)
        log_event("stage_cancelled" if status == "cancelled" else "stage_failed", name, status,
                  duration_ms=(perf_counter() - started) * 1000, error=error, trace=trace)
        raise
    else:
        log_event("stage_completed", name, state.status,
                  duration_ms=(perf_counter() - started) * 1000, trace=trace)
