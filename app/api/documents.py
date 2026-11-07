from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ingestion.loader import ALLOWED_EXTENSIONS, DocumentLoadError, display_filename
from app.ingestion.service import IngestionService
from app.observability import log_event
from pathlib import Path


router = APIRouter(prefix="/documents", tags=["documents"])


def get_ingestion_service(request: Request) -> IngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="当前应用未配置文档入库服务")
    return service


@router.post("/upload", status_code=201)
async def upload_document(
    response: Response, file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
):
    try:
        filename = display_filename(file.filename)
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise DocumentLoadError("仅支持 TXT、Markdown 和文本型 PDF", status_code=415)
        limit = service.max_upload_bytes
        if file.size is not None and file.size > limit:
            raise DocumentLoadError("文件超过最大上传大小", status_code=413)
        content = bytearray()
        while True:
            # 限制实际读取量，不信任 Content-Type 或仅依赖客户端长度声明。
            block = await file.read(min(65536, limit + 1 - len(content)))
            if not block:
                break
            content.extend(block)
            if len(content) > limit:
                raise DocumentLoadError("文件超过最大上传大小", status_code=413)
        result = await run_in_threadpool(service.upload, filename, bytes(content))
        if result["status"] == "duplicate":
            response.status_code = 200
        return result
    except DocumentLoadError as error:
        log_event("stage_failed", "ingestion_request", "rejected", error=error)
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
    except Exception as error:
        log_event("stage_failed", "ingestion_request", "failed", error=error)
        raise HTTPException(status_code=500, detail="文档入库失败，请稍后重试；可通过请求 ID 排查。") from None
    finally:
        await file.close()
