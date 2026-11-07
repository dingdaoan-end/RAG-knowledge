export type AnswerStatus = "answered" | "no_evidence";
export interface Source {
  document_id: string;
  source: string;
  chunk_id: string | number | null;
  content: string;
  rank: number;
  score: number;
  page?: number;
}
export interface Answer {
  answer: string;
  status: AnswerStatus;
  sources: Source[];
}
export interface UploadResult {
  status: "indexed" | "duplicate";
  document_id: string;
  filename: string;
  chunk_count: number;
}
export class ApiError extends Error {
  constructor(
    message: string,
    public requestId?: string,
    public status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
const object = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;
const protocol = (message: string, requestId?: string) =>
  new ApiError(message, requestId);

function parseSources(value: unknown, requestId?: string): Source[] {
  if (
    !Array.isArray(value) ||
    !value.every(
      (s) =>
        object(s) &&
        typeof s.document_id === "string" &&
        typeof s.source === "string" &&
        typeof s.content === "string" &&
        (s.chunk_id === null ||
          typeof s.chunk_id === "string" ||
          typeof s.chunk_id === "number") &&
        typeof s.rank === "number" &&
        Number.isInteger(s.rank) &&
        s.rank > 0 &&
        typeof s.score === "number" &&
        Number.isFinite(s.score) &&
        (s.page === undefined || (typeof s.page === "number" && s.page > 0)),
    )
  ) {
    throw protocol("服务返回的来源格式无效，请重试。", requestId);
  }
  return value as Source[];
}
async function request(
  path: string,
  init: RequestInit,
  acceptedStatuses: number[] = [],
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new ApiError("无法连接知识库服务，请检查服务状态后重试。");
  }
  if (!response.ok && !acceptedStatuses.includes(response.status)) {
    let message = `服务暂时不可用（HTTP ${response.status}），请稍后重试。`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail))
        message = body.detail
          .map((item: { msg?: string }) => item.msg || "请求参数无效")
          .join("；");
    } catch {
      /* Nginx 的 HTML 错误页不作为正文渲染。 */
    }
    throw new ApiError(
      message,
      response.headers.get("X-Request-ID") || undefined,
      response.status,
    );
  }
  return response;
}
async function readJson(response: Response): Promise<Record<string, unknown>> {
  try {
    const body: unknown = await response.json();
    if (object(body)) return body;
  } catch {
    /* below */
  }
  throw protocol(
    "服务返回的数据格式无效，请重试。",
    response.headers.get("X-Request-ID") || undefined,
  );
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);
  const response = await request("/documents/upload", { method: "POST", body });
  const data = await readJson(response);
  if (
    !["indexed", "duplicate"].includes(String(data.status)) ||
    typeof data.document_id !== "string" ||
    typeof data.filename !== "string" ||
    typeof data.chunk_count !== "number"
  )
    throw protocol("入库结果格式无效，请核实服务状态。");
  return data as unknown as UploadResult;
}
export async function chat(
  question: string,
  signal?: AbortSignal,
): Promise<Answer> {
  const response = await request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  const data = await readJson(response);
  if (
    typeof data.answer !== "string" ||
    !data.answer.trim() ||
    !["answered", "no_evidence"].includes(String(data.status))
  )
    throw protocol("服务没有返回有效的答案。");
  return {
    answer: data.answer,
    status: data.status as AnswerStatus,
    sources: parseSources(data.sources),
  };
}
export async function readiness(
  signal?: AbortSignal,
): Promise<"ready" | "initializing"> {
  const response = await request("/ready", { signal }, [503]);
  const data = await readJson(response);
  if (response.ok && data.status === "ready") return "ready";
  if (response.status === 503 && data.status === "initializing")
    return "initializing";
  throw new ApiError(
    typeof data.detail === "string"
      ? data.detail
      : "服务尚未就绪，请检查后端配置。",
  );
}

interface StreamCallbacks {
  onSources(sources: Source[]): void;
  onToken(token: string): void;
  onDone(status: AnswerStatus): void;
}
export async function streamChat(
  question: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await request("/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ question }),
    signal,
  });
  const requestId = response.headers.get("X-Request-ID") || undefined;
  if (
    !response.headers
      .get("Content-Type")
      ?.toLowerCase()
      .includes("text/event-stream") ||
    !response.body
  )
    throw protocol("服务没有返回流式响应。", requestId);
  const reader = response.body.getReader(),
    decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "",
    event = "",
    data: string[] = [],
    finished = false,
    sawSources = false,
    validText = false,
    sources: Source[] = [];
  function dispatch() {
    if (!data.length) {
      event = "";
      return;
    }
    const name = event || "message",
      payload = data.join("\n");
    event = "";
    data = [];
    if (!["sources", "token", "done", "error"].includes(name)) return;
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      throw protocol("流式数据格式无效，请重试。", requestId);
    }
    if (!object(value)) throw protocol("流式数据格式无效，请重试。", requestId);
    if (name === "error")
      throw new ApiError(
        typeof value.message === "string"
          ? value.message
          : "生成中断，请重试。",
        requestId,
      );
    if (name === "sources") {
      if (sawSources) throw protocol("流式来源顺序异常，请重试。", requestId);
      sources = parseSources(value.sources, requestId);
      sawSources = true;
      callbacks.onSources(sources);
    } else if (name === "token") {
      if (!sawSources)
        throw protocol("答案开始前未收到来源，请重试。", requestId);
      if (typeof value.token !== "string")
        throw protocol("流式数据格式无效，请重试。", requestId);
      validText ||= Boolean(value.token.trim());
      callbacks.onToken(value.token);
    } else {
      if (!sawSources || !validText)
        throw protocol("服务没有返回有效的文本答案。", requestId);
      if (
        value.status !== undefined &&
        !["answered", "no_evidence"].includes(String(value.status))
      )
        throw protocol("答案状态格式无效。", requestId);
      const status =
        value.status === "no_evidence" ? "no_evidence" : "answered";
      if (status === "no_evidence" && sources.length)
        throw protocol("无证据状态与来源不一致。", requestId);
      finished = true;
      callbacks.onDone(status);
    }
  }
  function consume(final = false) {
    while (!finished) {
      const index = buffer.search(/[\r\n]/);
      if (
        index < 0 ||
        (!final && buffer[index] === "\r" && index === buffer.length - 1)
      )
        break;
      const line = buffer.slice(0, index);
      const width =
        buffer[index] === "\r" && buffer[index + 1] === "\n" ? 2 : 1;
      buffer = buffer.slice(index + width);
      if (!line) {
        dispatch();
        continue;
      }
      if (line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      let value = colon < 0 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      if (field === "data") data.push(value);
    }
  }
  try {
    while (!finished) {
      const chunk = await reader.read();
      if (chunk.done) {
        buffer += decoder.decode();
        consume(true);
        break;
      }
      buffer += decoder.decode(chunk.value, { stream: true });
      consume();
      if (buffer.length > 4 * 1024 * 1024)
        throw protocol("流式响应过大，连接已停止。", requestId);
    }
    if (!finished)
      throw protocol("连接提前结束，回答可能不完整，请重试。", requestId);
  } catch (error) {
    if (
      error instanceof ApiError ||
      (error instanceof DOMException && error.name === "AbortError")
    )
      throw error;
    throw new ApiError("连接已中断，回答可能不完整，请重试。", requestId);
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
