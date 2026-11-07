from collections.abc import Iterator

from openai import OpenAI

from config import Settings


class BaseLLM:
    """所有 LLM 实现都提供相同的 generate(prompt) 接口。"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("子类必须实现 generate(prompt)")

    def stream_generate(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError("子类必须实现 stream_generate(prompt)")

    def close(self) -> None:
        """有网络资源的实现可以覆盖此方法，在服务停止时释放连接。"""


class MockLLM(BaseLLM):
    """返回固定答案以验证调用链；不执行真实语言模型推理。"""

    def generate(self, prompt: str) -> str:
        return "这是基于知识库资料生成的测试回答"

    def stream_generate(self, prompt: str) -> Iterator[str]:
        # 固定测试答案逐字符输出；真实网络增量由 OpenAILLM 的流式接口处理。
        for token in self.generate(prompt):
            yield token


class OpenAILLM(BaseLLM):
    """用官方 SDK 调用 Responses API；密钥只来自配置中的环境变量。"""

    def __init__(self, model: str | None = None, settings: Settings | None = None) -> None:
        settings = settings or Settings.from_env()
        if not settings.api_key:
            raise ValueError("请设置 API_KEY 或 OPENAI_API_KEY 环境变量")
        self.model = model or settings.model_name
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=60.0)

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        if response.status != "completed" or not response.output_text.strip():
            raise RuntimeError("模型未返回完整的文本答案")
        return response.output_text

    def stream_generate(self, prompt: str) -> Iterator[str]:
        completed = False
        has_text = False
        # with 确保正常结束、异常或调用方关闭生成器时都关闭上游响应。
        with self.client.responses.create(model=self.model, input=prompt, stream=True) as stream:
            for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    # 保留原始空白，但纯空白不能作为成功生成答案的依据。
                    has_text = has_text or bool(event.delta.strip())
                    yield event.delta
                elif event.type == "response.completed":
                    completed = True
                elif event.type in {"error", "response.failed", "response.incomplete"}:
                    raise RuntimeError("模型生成失败或未完成")
        if not completed:
            raise RuntimeError("模型生成流未正常结束")
        if not has_text:
            raise RuntimeError("模型未返回有效的文本答案")

    def close(self) -> None:
        self.client.close()
