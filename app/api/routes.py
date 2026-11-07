import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.pipeline import RAGPipeline
from app.observability import log_event


router = APIRouter()
GENERATION_ERROR = "生成答案失败，请检查模型服务配置或稍后重试。"


class ChatStreamingResponse(StreamingResponse):
    """即使发送响应头/正文时断开，也关闭持有的上游生成器。"""

    def __init__(self, content, tokens, **kwargs):
        super().__init__(content, **kwargs)
        self.tokens = tokens

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                self.tokens.close()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "no_evidence"]
    sources: list[dict] = Field(default_factory=list)


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="应用尚未就绪，请检查 /ready。")
    return pipeline


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> ChatResponse:
    # 同步端点由 FastAPI 在线程池执行，模型等待不会阻塞事件循环。
    try:
        result = pipeline.ask_with_result(body.question)
        return ChatResponse(answer=result.answer, status=result.status, sources=result.sources)
    except Exception as error:
        log_event("stage_failed", "response", "failed", error=error)
        raise HTTPException(status_code=502, detail=GENERATION_ERROR) from error


def sse_event(event: str, payload: dict) -> str:
    # JSON 转义换行，保证多行文本也不会破坏 SSE 帧。
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    tokens = pipeline.stream_ask_with_result(body.question)
    try:
        # 等待首段非空白文本再发送响应头；保留前导空白，空流返回 502。
        prefix = []
        for result in tokens:
            prefix.append(result.answer)
            if result.answer.strip():
                first = "".join(prefix)
                status = result.status
                sources = result.sources
                break
        else:
            raise RuntimeError("模型未返回有效的文本答案")
    except Exception as error:
        log_event("stage_failed", "response", "failed", error=error)
        tokens.close()
        raise HTTPException(status_code=502, detail=GENERATION_ERROR) from error

    async def events():
        try:
            yield sse_event("sources", {"sources": sources})
            if first is not None:
                yield sse_event("token", {"token": first})
            while True:
                result = await run_in_threadpool(next, tokens, None)
                if result is None:
                    break
                yield sse_event("token", {"token": result.answer})
            # 普通生成保持旧 done payload；空候选额外标明确定性的证据不足状态。
            yield sse_event("done", {"status": status} if status == "no_evidence" else {})
        except Exception as error:
            log_event("stage_failed", "response", "failed", error=error)
            # 发送响应头后不能改变 HTTP 状态码，用 error 事件通知客户端。
            yield sse_event("error", {"message": GENERATION_ERROR})
        finally:
            tokens.close()

    return ChatStreamingResponse(
        events(), tokens, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
