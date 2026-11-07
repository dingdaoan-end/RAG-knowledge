from io import BytesIO
import asyncio
import json

from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from main import create_app
from test_ingestion_service import make_service


def text_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (PDF unique technical evidence.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def events(text):
    return [(b.splitlines()[0][7:], json.loads(b.splitlines()[1][6:])) for b in text.strip().split("\n\n")]


@pytest.mark.parametrize("name,content", [("alpha.txt", "甲星事实来自上传TXT。".encode()), ("alpha.md", "# 甲星\nMarkdown事实。".encode()), ("alpha.pdf", None)])
def test_upload_then_chat_and_stream_use_same_actual_new_source(tmp_path, name, content):
    service, pipeline = make_service(tmp_path, base=False)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        uploaded = client.post("/documents/upload", files={"file": (name, content or text_pdf(), "application/octet-stream")})
        assert uploaded.status_code == 201
        info = uploaded.json()
        assert info["status"] == "indexed" and info["chunk_count"] > 0
        assert uploaded.headers["X-Request-ID"]
        response = client.post("/chat", json={"question": "甲星"})
        body = response.json()
        assert response.status_code == 200 and body["status"] == "answered"
        source = body["sources"][0]
        assert source["document_id"] == info["document_id"] and source["source"] == name
        assert source["content"] in body["answer"]
        assert source["content"] in pipeline.llm.prompts[-1]
        streamed = client.post("/chat/stream", json={"question": "甲星"})
        frames = events(streamed.text)
        assert frames[0][0] == "sources" and frames[-1] == ("done", {})
        assert frames[0][1]["sources"] == body["sources"]
        assert "".join(p["token"] for event, p in frames if event == "token") == body["answer"]
        if name.endswith(".pdf"):
            assert source["page"] == 1


@pytest.mark.parametrize("name,content,status", [
    ("bad.exe", b"not supported", 415), ("bad.pdf", b"not a pdf", 422),
    ("broken.pdf", b"%PDF-1.7\nnot valid", 422), ("binary.txt", b"\xff\x00", 422),
    ("empty.txt", b"", 422), ("blank.md", b" \n\t", 422),
])
def test_rejected_upload_leaves_no_files_or_documents(tmp_path, name, content, status):
    service, pipeline = make_service(tmp_path, base=False)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        response = client.post("/documents/upload", files={"file": (name, content, "application/pdf")})
        assert response.status_code == status
        assert response.headers["X-Request-ID"] and response.json()["detail"]
        assert list(service.upload_dir.iterdir()) == []
        assert pipeline.ask_with_result("问题").status == "no_evidence"


def test_upload_size_is_actual_bytes_not_client_mime(tmp_path):
    service, pipeline = make_service(tmp_path, base=False, max_upload_bytes=8)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        response = client.post("/documents/upload", files={"file": ("big.txt", b"123456789", "text/plain")})
    assert response.status_code == 413
    assert list(service.upload_dir.iterdir()) == []


def test_pdf_with_no_text_explains_ocr_requirement(tmp_path):
    service, pipeline = make_service(tmp_path, base=False)
    writer, buffer = PdfWriter(), BytesIO()
    writer.add_blank_page(width=300, height=300)
    writer.write(buffer)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        response = client.post("/documents/upload", files={"file": ("scan.pdf", buffer.getvalue())})
    assert response.status_code == 422 and "OCR" in response.json()["detail"]


def test_http_duplicate_returns_existing_id_and_restart_still_answers(tmp_path):
    service, pipeline = make_service(tmp_path)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        first = client.post("/documents/upload", files={"file": ("../../knowledge.txt", "甲星上传数据".encode())})
        second = client.post("/documents/upload", files={"file": ("renamed.md", "甲星上传数据".encode())})
    assert first.status_code == 201 and second.status_code == 200
    assert first.json()["document_id"] == second.json()["document_id"]
    restored, pipeline2 = make_service(tmp_path)
    with TestClient(create_app(pipeline2, ingestion_service=restored)) as client:
        response = client.post("/chat", json={"question": "甲星"})
    assert response.json()["sources"][0]["document_id"] == first.json()["document_id"]
    assert response.json()["sources"][0]["source"] == "knowledge.txt"


@pytest.mark.parametrize("original, renamed, content", [
    ("valid.txt", "disguised.pdf", "甲星文本资料".encode()),
    ("valid.pdf", "disguised.txt", None),
])
def test_duplicate_bytes_still_require_extension_and_content_to_match(tmp_path, original, renamed, content):
    service, pipeline = make_service(tmp_path, base=False)
    content = text_pdf() if content is None else content
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        assert client.post("/documents/upload", files={"file": (original, content)}).status_code == 201
        before = {p.name: p.read_bytes() for p in service.upload_dir.iterdir()}
        response = client.post("/documents/upload", files={"file": (renamed, content)})
    assert response.status_code == 422
    assert {p.name: p.read_bytes() for p in service.upload_dir.iterdir()} == before
    assert len(service.snapshot.documents) == 1


def test_ingestion_failure_is_safe_500_and_keeps_existing_qa(tmp_path, monkeypatch):
    service, pipeline = make_service(tmp_path)
    monkeypatch.setattr(service, "_write_manifest", lambda records: (_ for _ in ()).throw(OSError("private-request-info")))
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        response = client.post("/documents/upload", files={"file": ("new.txt", "甲星新资料".encode())})
        assert response.status_code == 500 and response.headers["X-Request-ID"]
        assert "private-request-info" not in response.text
        assert "原始" in client.post("/chat", json={"question": "原始"}).json()["answer"]
    assert list(service.upload_dir.iterdir()) == []


def test_empty_http_knowledge_base_starts_and_streams_no_evidence_without_llm(tmp_path):
    service, pipeline = make_service(tmp_path, base=False)
    with TestClient(create_app(pipeline, ingestion_service=service)) as client:
        body = client.post("/chat", json={"question": "问题"}).json()
        assert body["status"] == "no_evidence" and body["sources"] == []
        frames = events(client.post("/chat/stream", json={"question": "问题"}).text)
    assert frames[0] == ("sources", {"sources": []})
    assert frames[-1] == ("done", {"status": "no_evidence"})
    assert not pipeline.llm.prompts


@pytest.mark.parametrize("with_length", [False, True])
@pytest.mark.parametrize("prefix", ["", "/rag"])
def test_oversized_multipart_is_stopped_before_entire_body_is_consumed(tmp_path, monkeypatch, with_length, prefix):
    import starlette.formparsers as parsing

    service, pipeline = make_service(tmp_path, base=False, max_upload_bytes=1024)
    app = create_app(pipeline, ingestion_service=service)
    if prefix:
        mounted = FastAPI()
        mounted.mount(prefix, app)
        app = mounted
    opened = []
    real_spool = parsing.SpooledTemporaryFile

    def recording_spool(*args, **kwargs):
        handle = real_spool(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(parsing, "SpooledTemporaryFile", recording_spool)
    body = (b'--test-boundary\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n'
            b'Content-Type: text/plain\r\n\r\n' + b'x' * 200_000 + b'\r\n--test-boundary--\r\n')
    headers = [(b"content-type", b"multipart/form-data; boundary=test-boundary")]
    if with_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": prefix + "/documents/upload", "raw_path": (prefix + "/documents/upload").encode(),
             "query_string": b"", "root_path": "", "headers": headers,
             "client": ("127.0.0.1", 12345), "server": ("test", 80)}
    consumed, messages = 0, []

    async def receive():
        nonlocal consumed
        start = consumed
        consumed = min(consumed + 4096, len(body))
        return {"type": "http.request", "body": body[start:consumed], "more_body": consumed < len(body)}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    assert start["status"] == 413
    assert dict(start["headers"])[b"x-request-id"]
    assert consumed < len(body)  # 不能先读完超大文件再返回 413。
    if with_length:
        assert consumed == 0 and opened == []
    else:
        assert opened and all(handle.closed for handle in opened)
    assert service.snapshot.documents == () and list(service.upload_dir.iterdir()) == []
