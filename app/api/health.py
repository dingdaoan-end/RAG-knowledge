"""存活与就绪只读取状态，不执行模型调用或检索。"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready", responses={503: {"description": "应用初始化中、失败或正在停止"}})
async def ready(request: Request):
    status = request.app.state.initialization_status
    if status == "ready":
        return {"status": "ready"}
    body = {"status": status}
    if status == "failed":
        body["detail"] = request.app.state.initialization_detail
    return JSONResponse(body, status_code=503)
