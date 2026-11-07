"""相关性 > 0 为二元相关；排名从 1 开始；分级增益为 2**rel - 1。"""

import math
from statistics import fmean


def validate_identifier(value: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("评估标识必须是没有首尾空白的非空字符串")


def validate_grade(grade: int) -> None:
    if type(grade) is not int or not 0 <= grade <= 3:
        raise ValueError("relevance 必须是 0、1、2 或 3")


def validate_relevance(relevance: dict[str, int]) -> None:
    if not isinstance(relevance, dict):
        raise ValueError("相关性标注必须为 ID 到等级的字典")
    for document_id, grade in relevance.items():
        validate_identifier(document_id)
        validate_grade(grade)
    if not any(grade > 0 for grade in relevance.values()):
        raise ValueError("每个 Query 至少需要一个正相关标注，不能静默计为 0")


def _validate_ranking(retrieved_ids: list[str], relevance: dict[str, int]) -> None:
    validate_relevance(relevance)
    for document_id in retrieved_ids:
        validate_identifier(document_id)
    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("排名中不能有重复评估标识")


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")


def recall_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    _validate_k(k)
    _validate_ranking(retrieved_ids, relevance)
    relevant = {document_id for document_id, grade in relevance.items() if grade > 0}
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def first_relevant_rank(retrieved_ids: list[str], relevance: dict[str, int]) -> int | None:
    _validate_ranking(retrieved_ids, relevance)
    return next((rank for rank, document_id in enumerate(retrieved_ids, 1) if relevance.get(document_id, 0) > 0), None)


def reciprocal_rank(retrieved_ids: list[str], relevance: dict[str, int]) -> float:
    """在传入列表中找首个命中；列表若被 Top-N 截断，调用方必须标作 RR@N。"""
    rank = first_relevant_rank(retrieved_ids, relevance)
    return 0.0 if rank is None else 1 / rank


def mean_reciprocal_rank(rankings: list[list[str]], labels: list[dict[str, int]]) -> float:
    if not rankings or len(rankings) != len(labels):
        raise ValueError("Query 集不能为空，且排名与标注数量必须一致")
    return fmean(reciprocal_rank(ranking, relevance) for ranking, relevance in zip(rankings, labels))


def dcg_at_k(grades: list[int], k: int) -> float:
    _validate_k(k)
    for grade in grades:
        validate_grade(grade)
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    _validate_ranking(retrieved_ids, relevance)
    ideal = dcg_at_k(sorted(relevance.values(), reverse=True), k)
    actual = dcg_at_k([relevance.get(document_id, 0) for document_id in retrieved_ids], k)
    return actual / ideal  # validate_relevance 已保证 IDCG > 0。
