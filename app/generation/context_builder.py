def build_context(results: list) -> str:
    """将 SearchResult 或同字段的字典转为保留出处的资料字符串。"""
    if not results:
        return "没有检索到可用的知识库资料。"

    blocks = []
    for result in results:
        if isinstance(result, dict):
            document, rank = result["document"], result["rank"]
            score = result["score"] if "score" in result else result["final_score"]
        else:
            document, score, rank = result.document, result.score, result.rank

        metadata = document.metadata
        lines = [
            f"[Document {rank}]",
            f"source: {metadata.get('source', '未知来源')}",
            f"chunk_id: {metadata.get('chunk_id', '未知')}",
            f"rank: {rank}",
            f"score: {score}",
        ]
        # 除 source/chunk_id 外的 metadata 也保留，不修改原字典。
        lines.extend(f"{key}: {value}" for key, value in metadata.items() if key not in {"source", "chunk_id"})
        lines.extend(["", "content:", document.text])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
