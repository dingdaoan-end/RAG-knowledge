"""纯 ASGI 请求追踪：上下文持续到流式响应和资源清理结束。"""

from time import perf_counter
from uuid import uuid4

from fastapi.responses import PlainTextResponse

from app.observability import RequestTrace, failure_status, log_event, request_trace


async def unhandled_error_response(request, error):
    # ServerErrorMiddleware 位于用户中间件外层，从 scope 保留的 ID 补回 500 响应头。
    # 保持默认的通用 500 正文；框架随后仍会重新抛出原异常。
    return PlainTextResponse("Internal Server Error", status_code=500,
                             headers={"X-Request-ID": request.state.request_id})


class RequestTraceMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # 一律忽略外部请求 ID 和其他头字段；不是认证凭据。
        trace = RequestTrace(uuid4().hex)
        scope.setdefault("state", {})["request_id"] = trace.request_id
        context_token = request_trace.set(trace)
        started = perf_counter()
        http_status = None
        body_finished = False
        error = None
        log_event("stage_started", "http", "started", trace=trace)

        async def traced_receive():
            message = await receive()
            if message["type"] == "http.disconnect" and not body_finished and trace.status != "failed":
                trace.status = "cancelled"
            return message

        async def traced_send(message):
            nonlocal http_status, body_finished
            if message["type"] == "http.response.start":
                http_status = message["status"]
                message = dict(message)
                message["headers"] = [(key, value) for key, value in message.get("headers", []) if key.lower() != b"x-request-id"]
                message["headers"].append((b"x-request-id", trace.request_id.encode("ascii")))
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                body_finished = True

        try:
            await self.app(scope, traced_receive, traced_send)
        except BaseException as caught:
            error = caught
            status = failure_status(caught)
            if trace.status != "failed":
                trace.status = status
            if http_status is None and status == "failed":
                http_status = 500
            raise
        finally:
            try:
                status = trace.status
                if status == "ok" and http_status is not None and http_status >= 400:
                    status = "failed" if http_status >= 500 else "rejected"
                event = "stage_failed" if status == "failed" else "stage_cancelled" if status == "cancelled" else "stage_completed"
                log_event(event, "http", status, duration_ms=(perf_counter() - started) * 1000,
                          error=error, http_status=http_status, trace=trace)
            finally:
                request_trace.reset(context_token)
