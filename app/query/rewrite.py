class QueryRewriter:
    """将用户问题改成适合检索的查询；真实实现以后可调用 LLM。"""

    def rewrite(self, query: str) -> str:
        raise NotImplementedError("子类必须实现 rewrite(query)")


class MockQueryRewriter(QueryRewriter):
    """只清理首尾空白，不增加事实或改变问题含义。"""

    def rewrite(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        return query
