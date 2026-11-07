"""只为评估建立稳定 eval_id 映射，不改变全局 Document 数据模型。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from app.ingestion.document import Document
from .retrieval_metrics import validate_grade, validate_identifier, validate_relevance


@dataclass
class EvaluationDataset:
    documents: list[Document]
    queries: list[dict]
    metadata: dict
    sha256: str


def _unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段重复，不能静默覆盖: {key}")
        result[key] = value
    return result


def load_dataset(path: str | Path) -> EvaluationDataset:
    content = Path(path).read_bytes()
    raw = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(raw, dict) or not raw.get("documents") or not raw.get("queries"):
        raise ValueError("数据集需要非空 documents 和 queries 列表")
    if not isinstance(raw["documents"], list) or not isinstance(raw["queries"], list):
        raise ValueError("documents 和 queries 必须是列表")
    documents, queries = [], []
    document_ids, source_chunks, query_ids = set(), set(), set()
    try:
        for item in raw["documents"]:
            document_id, text, source, chunk_id = item["id"], item["text"], item["source"], item["chunk_id"]
            validate_identifier(document_id)
            validate_identifier(source)
            if document_id in document_ids or (source, chunk_id) in source_chunks:
                raise ValueError(f"文档 ID 或 source/chunk_id 重复: {document_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"文档文本不能为空: {document_id}")
            if type(chunk_id) is not int or chunk_id < 0:
                raise ValueError(f"chunk_id 必须是非负整数: {document_id}")
            document_ids.add(document_id)
            source_chunks.add((source, chunk_id))
            documents.append(Document(text, {"eval_id": document_id, "source": source, "chunk_id": chunk_id}))

        for item in raw["queries"]:
            query_id, question, labels = item["id"], item["query"], item["relevant_documents"]
            validate_identifier(query_id)
            if query_id in query_ids or not isinstance(question, str) or not question.strip():
                raise ValueError(f"Query 标识重复或问题为空: {query_id}")
            if not isinstance(labels, list):
                raise ValueError(f"relevant_documents 必须是列表: {query_id}")
            relevance = {}
            for label in labels:
                document_id, grade = label["id"], label["relevance"]
                validate_identifier(document_id)
                validate_grade(grade)
                if document_id not in document_ids or document_id in relevance:
                    raise ValueError(f"标注 ID 不存在或重复: {query_id}/{document_id}")
                relevance[document_id] = grade
            try:
                validate_relevance(relevance)
            except ValueError as error:
                raise ValueError(f"{query_id}: {error}") from error
            query_ids.add(query_id)
            queries.append({"id": query_id, "query": question, "relevant_documents": labels})
    except (KeyError, TypeError) as error:
        raise ValueError(f"评估数据缺失字段或类型不正确: {error}") from error
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata 必须是字典")
    if metadata.get("unlisted_relevance", 0) != 0 or metadata.get("binary_relevance_threshold", 1) != 1:
        raise ValueError("当前评估规则固定为未标注文档=0、正相关阈值=1，metadata 不能与之冲突")
    return EvaluationDataset(documents, queries, metadata, hashlib.sha256(content).hexdigest())
