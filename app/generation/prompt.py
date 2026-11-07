SYSTEM_PROMPT = """你是一个专业知识库助手。

要求：
1. 根据提供的资料回答问题
2. 不要使用资料之外的信息猜测
3. 如果资料中没有答案，明确告诉用户"""


def build_prompt(question: str, context: str) -> str:
    """集中管理规则与格式；返回完整提示词字符串。"""
    return f"System:\n\n{SYSTEM_PROMPT}\n\nContext:\n\n{context}\n\nQuestion:\n\n{question}"
