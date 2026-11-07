import { describe, expect, it, vi } from "vitest";
import { chat, readiness, streamChat, uploadDocument } from "./api";

const source = {
  document_id: "doc-1",
  source: "知识.md",
  chunk_id: 0,
  content: "真实原文",
  rank: 1,
  score: 0.85,
  page: 2,
};
const encoder = new TextEncoder();
const frame = (name: string, data: unknown) =>
  `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
function response(parts: Uint8Array[]) {
  return new Response(
    new ReadableStream({
      start(controller) {
        parts.forEach((part) => controller.enqueue(part));
        controller.close();
      },
    }),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "X-Request-ID": "trace-123",
      },
    },
  );
}
function serve(text: string, step = 99999) {
  const bytes = encoder.encode(text);
  const parts = [];
  for (let i = 0; i < bytes.length; i += step)
    parts.push(bytes.slice(i, i + step));
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(parts)));
}
const hooks = () => ({ onSources: vi.fn(), onToken: vi.fn(), onDone: vi.fn() });

describe("POST SSE", () => {
  it.each([1, 2, 7, 10000])(
    "parses fragmented UTF-8 and frames with chunk size %s",
    async (step) => {
      serve(
        frame("sources", { sources: [source] }) +
          frame("token", { token: "中文答" }) +
          frame("token", { token: "案" }) +
          frame("done", {}),
        step,
      );
      const callbacks = hooks();
      await streamChat("问题", callbacks);
      expect(callbacks.onSources).toHaveBeenCalledWith([source]);
      expect(callbacks.onToken.mock.calls.flat().join("")).toBe("中文答案");
      expect(callbacks.onDone).toHaveBeenCalledExactlyOnceWith("answered");
      expect(callbacks.onSources.mock.invocationCallOrder[0]).toBeLessThan(
        callbacks.onToken.mock.invocationCallOrder[0],
      );
      expect(fetch).toHaveBeenCalledWith(
        "/api/chat/stream",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ question: "问题" }),
        }),
      );
    },
  );
  it("handles CRLF split between chunks, comments, and multiline data", async () => {
    serve(
      ': keepalive\r\nevent: sources\r\ndata: {"sources":\r\ndata: []}\r\n\r\n' +
        frame("token", { token: "无资料" }).replaceAll("\n", "\r\n") +
        frame("done", { status: "no_evidence" }).replaceAll("\n", "\r\n"),
      1,
    );
    const callbacks = hooks();
    await streamChat("问题", callbacks);
    expect(callbacks.onSources).toHaveBeenCalledWith([]);
    expect(callbacks.onDone).toHaveBeenCalledWith("no_evidence");
  });
  it.each([
    ["", "连接"],
    [
      frame("sources", { sources: [] }) + frame("token", { token: "部分" }),
      "连接",
    ],
    ["event: sources\ndata: broken\n\n", "格式"],
    [frame("token", { token: "先输出" }) + frame("done", {}), "来源"],
    [frame("sources", { sources: [] }) + frame("done", {}), "有效"],
    [frame("sources", { sources: [{ ...source, score: "bad" }] }), "来源"],
  ])("rejects incomplete or invalid protocol", async (text, message) => {
    serve(text);
    await expect(streamChat("问题", hooks())).rejects.toThrow(message);
  });
  it("keeps partial tokens but rejects error without treating it as done", async () => {
    serve(
      frame("sources", { sources: [source] }) +
        frame("token", { token: "部分" }) +
        frame("error", { message: "上游中断" }),
    );
    const callbacks = hooks();
    await expect(streamChat("问题", callbacks)).rejects.toThrow("上游中断");
    expect(callbacks.onToken).toHaveBeenCalledWith("部分");
    expect(callbacks.onDone).not.toHaveBeenCalled();
  });
  it("surfaces safe HTTP errors and request IDs", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "模型服务不可用" }), {
            status: 502,
            headers: { "X-Request-ID": "test-id" },
          }),
        ),
    );
    await expect(streamChat("问题", hooks())).rejects.toMatchObject({
      message: "模型服务不可用",
      requestId: "test-id",
    });
  });
  it("reports non-SSE success as a protocol error", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response("html", { headers: { "Content-Type": "text/html" } }),
        ),
    );
    await expect(streamChat("问题", hooks())).rejects.toThrow("流式");
  });
  it("propagates abort without retrying", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new DOMException("Aborted", "AbortError")),
    );
    await expect(
      streamChat("问题", hooks(), new AbortController().signal),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("turns a broken network stream into a useful connection error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          new ReadableStream({
            start(controller) {
              controller.error(new TypeError("terminated"));
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      ),
    );
    await expect(streamChat("问题", hooks())).rejects.toThrow("连接");
  });
});

describe("ordinary APIs", () => {
  it("uses the actual multipart field without manually assigning Content-Type", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              status: "indexed",
              document_id: "doc-1",
              filename: "a.pdf",
              chunk_count: 1,
            }),
          ),
        ),
    );
    const file = new File(["test"], "a.pdf", { type: "application/pdf" });
    expect((await uploadDocument(file)).status).toBe("indexed");
    const options = vi.mocked(fetch).mock.calls[0][1]!;
    expect((options.body as FormData).get("file")).toBe(file);
    expect(new Headers(options.headers).has("Content-Type")).toBe(false);
  });
  it("accepts a duplicate with HTTP 200", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              status: "duplicate",
              document_id: "doc-1",
              filename: "a.txt",
              chunk_count: 1,
            }),
          ),
        ),
    );
    expect((await uploadDocument(new File(["a"], "a.txt"))).status).toBe(
      "duplicate",
    );
  });
  it("calls ordinary chat and preserves no_evidence and actual sources", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              answer: "无资料",
              status: "no_evidence",
              sources: [],
            }),
          ),
        ),
    );
    expect(await chat("问题")).toEqual({
      answer: "无资料",
      status: "no_evidence",
      sources: [],
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({ method: "POST" }),
    );
  });
  it("shows backend upload validation messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "PDF 没有可提取正文" }), {
            status: 422,
          }),
        ),
    );
    await expect(uploadDocument(new File(["bad"], "a.pdf"))).rejects.toThrow(
      "PDF 没有可提取正文",
    );
  });
  it("handles network failures without automatic retries", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );
    await expect(chat("问题")).rejects.toThrow("无法连接");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("distinguishes initializing from failed readiness by status, not message keywords", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ status: "initializing" }), {
            status: 503,
          }),
        )
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({ status: "failed", detail: "本地配置错误" }),
            { status: 503 },
          ),
        ),
    );
    expect(await readiness()).toBe("initializing");
    await expect(readiness()).rejects.toThrow("本地配置错误");
  });
});
