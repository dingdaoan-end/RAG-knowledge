"""在 FastAPI 解析 multipart 之前约束上传请求的实际字节数。"""

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette._utils import get_route_path


MULTIPART_OVERHEAD_BYTES = 64 * 1024
UPLOAD_TOO_LARGE = "上传请求过大，请缩小文件或 multipart 附加内容。"


class UploadLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or get_route_path(scope).rstrip("/") != "/documents/upload":
            await self.app(scope, receive, send)
            return
        service = getattr(scope["app"].state, "ingestion_service", None)
        if service is None:
            # 初始化期间尚无文件限额服务，必须在 multipart 暂存之前拒绝。
            await JSONResponse({"detail": "当前应用未配置文档入库服务，请检查 /ready。"}, status_code=503)(scope, receive, send)
            return
        limit = service.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    oversized = int(value) > limit
                except ValueError:
                    oversized = False
                if oversized:
                    await JSONResponse({"detail": UPLOAD_TOO_LARGE}, status_code=413)(scope, receive, send)
                    return
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    # Starlette 在流读取异常时关闭已创建的 multipart 暂存文件。
                    raise HTTPException(status_code=413, detail=UPLOAD_TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)
