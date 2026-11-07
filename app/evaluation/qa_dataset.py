"""独立 QA 标注；复用 Day 7 的稳定文档映射，不修改冻结语料。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .dataset import EvaluationDataset, _unique_json_object
from .retrieval_metrics import validate_identifier


@dataclass
class QADataset:
    version: str
    sha256: str
    corpus_sha256: str
    samples: list[dict]
    metadata: dict


def require_text(value, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def validate_sample(item: dict, known_ids: set[str]) -> None:
    try:
        validate_identifier(item["id"])
        require_text(item["question"], "question")
        require_text(item["reference_answer"], "reference_answer")
        facts, document_ids = item["required_facts"], item["relevant_document_ids"]
        if not isinstance(facts, list) or not facts:
            raise ValueError("required_facts 必须是非空字符串列表")
        for fact in facts:
            require_text(fact, "required_fact")
        if len(set(facts)) != len(facts):
            raise ValueError("required_facts 不能重复")
        if not isinstance(document_ids, list):
            raise ValueError("relevant_document_ids 必须是列表")
        for document_id in document_ids:
            validate_identifier(document_id)
            if document_id not in known_ids:
                raise ValueError(f"相关文档 ID 不存在: {document_id}")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("相关文档 ID 重复")
        if type(item["answerable"]) is not bool:
            raise ValueError("answerable 必须是布尔值")
        if item["answerable"] and not document_ids:
            raise ValueError("可回答问题必须关联已有证据文档")
    except (KeyError, TypeError) as error:
        raise ValueError(f"QA 样本缺少字段或类型错误: {error}") from error


def load_qa_dataset(path: str | Path, corpus: EvaluationDataset) -> QADataset:
    content = Path(path).read_bytes()
    raw = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_json_object)
    try:
        require_text(raw["version"], "version")
        if raw["corpus_sha256"] != corpus.sha256:
            raise ValueError("QA 数据集绑定的语料 SHA256 不一致")
        if not isinstance(raw["samples"], list) or not raw["samples"]:
            raise ValueError("QA samples 必须是非空列表")
        known_ids = {document.metadata["eval_id"] for document in corpus.documents}
        seen = set()
        for item in raw["samples"]:
            validate_sample(item, known_ids)
            if item["id"] in seen:
                raise ValueError(f"QA 样本 ID 重复: {item['id']}")
            seen.add(item["id"])
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("QA metadata 必须是字典")
        return QADataset(raw["version"], hashlib.sha256(content).hexdigest(), corpus.sha256, raw["samples"], metadata)
    except (KeyError, TypeError) as error:
        raise ValueError(f"QA 数据集缺少字段或类型错误: {error}") from error
