import math

import pytest


def test_recall_counts_multiple_relevant_chunks():
    from app.evaluation.retrieval_metrics import recall_at_k

    labels = {"a": 3, "b": 1, "c": 0}
    assert recall_at_k(["a", "c", "b"], labels, 1) == 0.5
    assert recall_at_k(["a", "c", "b"], labels, 3) == 1
    assert recall_at_k([], labels, 5) == 0


@pytest.mark.parametrize("ranking, expected_rank, rr", [(["a"], 1, 1), (["x", "a"], 2, 0.5), (["x", "y", "a"], 3, 1 / 3), (["x"], None, 0), ([], None, 0)])
def test_first_relevant_rank_is_one_based(ranking, expected_rank, rr):
    from app.evaluation.retrieval_metrics import first_relevant_rank, reciprocal_rank

    assert first_relevant_rank(ranking, {"a": 3}) == expected_rank
    assert reciprocal_rank(ranking, {"a": 3}) == rr


def test_mrr_is_mean_of_query_reciprocal_ranks():
    from app.evaluation.retrieval_metrics import mean_reciprocal_rank

    assert mean_reciprocal_rank([["x", "a"], ["x", "y", "a"], ["x"]], [{"a": 3}] * 3) == pytest.approx(5 / 18)


def test_graded_dcg_matches_hand_calculation():
    from app.evaluation.retrieval_metrics import dcg_at_k

    # gains=7,3,1；第 1、2、3 位的折扣分母为 1、log2(3)、2。
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(7 + 3 / math.log2(3) + 0.5)
    assert dcg_at_k([], 3) == 0
    assert dcg_at_k([0, 0], 3) == 0


def test_ndcg_uses_grades_and_penalizes_reversed_order():
    from app.evaluation.retrieval_metrics import ndcg_at_k

    labels = {"a": 3, "b": 2, "c": 1}
    ideal = 7 + 3 / math.log2(3) + 0.5
    assert ndcg_at_k(["a", "b", "c"], labels, 3) == pytest.approx(1)
    assert ndcg_at_k(["b", "a", "c"], labels, 3) == pytest.approx((3 + 7 / math.log2(3) + 0.5) / ideal)
    assert ndcg_at_k(["c"], labels, 1) == pytest.approx(1 / 7)
    assert ndcg_at_k([], labels, 3) == 0
    assert ndcg_at_k(["unjudged"], labels, 3) == 0


@pytest.mark.parametrize("labels", [{}, {"a": 0}, {"a": -1}, {"a": 4}, {"a": True}, {"a": 1.5}, {"": 3}])
def test_invalid_or_no_positive_ground_truth_is_rejected(labels):
    from app.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank, ndcg_at_k

    for metric in [lambda: recall_at_k([], labels, 3), lambda: reciprocal_rank([], labels), lambda: ndcg_at_k([], labels, 3)]:
        with pytest.raises(ValueError):
            metric()


@pytest.mark.parametrize("ranking", [["a", "a"], [""], [" a "], [None], [1]])
def test_duplicate_or_invalid_ranked_ids_are_rejected(ranking):
    from app.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank, ndcg_at_k

    for metric in [lambda: recall_at_k(ranking, {"a": 3}, 3), lambda: reciprocal_rank(ranking, {"a": 3}), lambda: ndcg_at_k(ranking, {"a": 3}, 3)]:
        with pytest.raises(ValueError):
            metric()


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k):
    from app.evaluation.retrieval_metrics import recall_at_k, dcg_at_k, ndcg_at_k

    for metric in [lambda: recall_at_k([], {"a": 3}, k), lambda: dcg_at_k([], k), lambda: ndcg_at_k([], {"a": 3}, k)]:
        with pytest.raises(ValueError):
            metric()


def test_mrr_rejects_empty_or_mismatched_query_sets():
    from app.evaluation.retrieval_metrics import mean_reciprocal_rank

    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [])
