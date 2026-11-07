"""保存人工评审；不自动打分，不把待评审或 Mock 记录当真实模型质量。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import fmean

from .dataset import _unique_json_object
from .qa_dataset import require_text


DIMENSIONS = ("faithfulness", "answer_relevance", "factual_correctness")


def pending_review(facts: list[str]) -> dict:
    return {
        "status": "pending", "reviewer": None, "reviewed_at": None,
        "faithfulness": None, "answer_relevance": None, "factual_correctness": None,
        "factual_correctness_basis": None,
        "required_facts_covered": [{"fact": fact, "covered": None, "answer_quote": None, "notes": ""} for fact in facts],
        "unsupported_claims": None, "insufficient_evidence_handled": None, "review_notes": "",
    }


def capture_digest(bundle: dict) -> str:
    """人工只编辑 review。指纹用于发现误改采集数据，不是防伪签名。"""
    payload = {key: value for key, value in bundle.items() if key not in {"capture_sha256", "records"}}
    payload["records"] = [{key: value for key, value in record.items() if key != "review"} for record in bundle["records"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _validate_review(record: dict) -> None:
    review = record["review"]
    if review["status"] not in {"pending", "in_progress", "completed"}:
        raise ValueError("review.status 不合法")
    completed = review["status"] == "completed"
    if not isinstance(review["review_notes"], str):
        raise ValueError("review_notes 必须是字符串")
    for key in DIMENSIONS:
        score = review[key]
        if score is not None and (type(score) is not int or score not in {0, 1, 2}):
            raise ValueError(f"{key} 必须为 null 或 0/1/2")
        if completed and score is None:
            raise ValueError(f"完成评审前必须填写 {key}")
    facts = review["required_facts_covered"]
    if not isinstance(facts, list) or [item["fact"] for item in facts] != record["required_facts"]:
        raise ValueError("关键事实评审项必须与 required_facts 逐项对应")
    for fact in facts:
        covered, quote = fact["covered"], fact["answer_quote"]
        if not isinstance(fact["notes"], str):
            raise ValueError("关键事实 notes 必须是字符串")
        if covered is not None and type(covered) is not bool:
            raise ValueError("covered 必须为 bool 或 null")
        if completed and covered is None:
            raise ValueError("完成评审前必须逐项判断关键事实")
        if covered:
            require_text(quote, "关键事实的 answer_quote")
            if quote not in record["answer"]:
                raise ValueError("关键事实引文必须来自实际答案")
    claims = review["unsupported_claims"]
    if claims is not None and not isinstance(claims, list):
        raise ValueError("unsupported_claims 必须是列表或 null")
    if completed and claims is None:
        raise ValueError("无依据断言必须显式审核；没有时填写 []")
    for claim in claims or []:
        for key in ("claim", "answer_quote", "reason"):
            require_text(claim[key], key)
        if claim["answer_quote"] not in record["answer"]:
            raise ValueError("无依据断言的引文必须来自实际答案")
        ids = claim["context_document_ids"]
        if not isinstance(ids, list) or any(document_id not in record["context_document_ids"] for document_id in ids):
            raise ValueError("断言关联的上下文 ID 不存在")
    handled = review["insufficient_evidence_handled"]
    if handled is not None and type(handled) is not bool:
        raise ValueError("insufficient_evidence_handled 必须为 bool 或 null")
    if completed:
        for key in ("reviewer", "reviewed_at", "factual_correctness_basis", "review_notes"):
            require_text(review[key], key)
        if datetime.fromisoformat(review["reviewed_at"]).utcoffset() is None:
            raise ValueError("reviewed_at 必须包含时区")
        if not record["answerable"] and handled is None:
            raise ValueError("证据不足样本必须审核是否正确说明限制")


def validate_qa_results(bundle: dict) -> None:
    try:
        if bundle["schema_version"] != 1 or not isinstance(bundle["records"], list) or not bundle["records"]:
            raise ValueError("QA 结果格式不合法")
        if bundle["capture_sha256"] != capture_digest(bundle):
            raise ValueError("采集指纹不一致：只能编辑 review，不能静默改写答案、上下文或模型信息")
        seen = set()
        for record in bundle["records"]:
            if record["id"] in seen:
                raise ValueError("QA 结果样本 ID 重复")
            seen.add(record["id"])
            if record["generation"] != bundle["run"]["generation"]:
                raise ValueError("记录与运行的生成模型类型不一致")
            if record["context_document_ids"] != [document["id"] for document in record["context_documents"]]:
                raise ValueError("上下文来源 ID 与快照不一致")
            # 旧采集没有这两个字段，继续兼容；新采集明确区分固定回复与模型生成。
            if "pipeline_status" in record or "llm_called" in record:
                status = record["pipeline_status"]
                if status not in {"answered", "no_evidence"} or record["llm_called"] is not (status == "answered"):
                    raise ValueError("Pipeline 状态与 LLM 调用标记不一致")
                if status == "no_evidence" and (record["prompt"] is not None or record["context"] or record["context_document_ids"]):
                    raise ValueError("未调用 LLM 的证据不足记录不能含实际 Prompt 或上下文")
            _validate_review(record)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"QA 评审记录缺失字段或类型不合法: {error}") from error


def save_qa_results(bundle: dict, path: str | Path) -> None:
    validate_qa_results(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 新建文件，避免覆盖历史采集、人工评审和 Day 7 文件。
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_qa_results(path: str | Path) -> dict:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    validate_qa_results(bundle)
    return bundle


def _review_metrics(completed: list[dict]) -> dict:
    metrics = {f"{key}_mean": fmean(record["review"][key] for record in completed) for key in DIMENSIONS}
    facts = [fact for record in completed for fact in record["review"]["required_facts_covered"]]
    negatives = [record for record in completed if not record["answerable"]]
    metrics.update(
        required_facts_coverage=sum(fact["covered"] for fact in facts) / len(facts),
        unsupported_claims_count=sum(len(record["review"]["unsupported_claims"]) for record in completed),
        insufficient_evidence_handled_rate=(sum(record["review"]["insufficient_evidence_handled"] for record in negatives) / len(negatives)) if negatives else None,
    )
    return metrics


def summarize_reviews(bundle: dict) -> dict:
    validate_qa_results(bundle)
    completed = [record for record in bundle["records"] if record["review"]["status"] == "completed"]
    generation = bundle["run"]["generation"]
    report = {
        "capture_sha256": bundle["capture_sha256"], "generation": generation,
        "total_count": len(bundle["records"]), "reviewed_count": len(completed),
        "pending_count": len(bundle["records"]) - len(completed),
        "no_evidence_count": sum(record.get("pipeline_status") == "no_evidence" for record in bundle["records"]),
        "human_review_metrics": None, "real_llm_quality_metrics": None,
        "message": "尚无已审核评分",
    }
    if completed:
        metrics = _review_metrics(completed)
        report["human_review_metrics"] = metrics
        if generation["mode"] == "real_api":
            generated = [record for record in completed if record.get("llm_called", True)]
            if generated:
                report["real_llm_quality_metrics"] = _review_metrics(generated)
        report["message"] = f"已汇总 {len(completed)} 条人工评审；未完成项不计分"
    if generation["mode"] != "real_api":
        report["message"] += "；当前为 Mock 或未验证自定义 LLM，不能视为真实生成模型质量"
    return report
