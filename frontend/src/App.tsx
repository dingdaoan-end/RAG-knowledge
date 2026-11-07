import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  FileText,
  Layers,
  Loader2,
  PanelLeft,
  Plus,
  RefreshCw,
  Sparkles,
  Square,
  Upload,
  X,
} from "lucide-react";
import Markdown from "react-markdown";
import {
  ApiError,
  chat,
  readiness,
  streamChat,
  uploadDocument,
  type Source,
  type UploadResult,
} from "./api";
import Sources from "./Sources";
import "./styles.css";

interface UploadRecord {
  key: string;
  filename: string;
  status: "uploading" | "indexed" | "duplicate" | "error";
  document_id?: string;
  chunk_count?: number;
  error?: string;
  createdAt: string;
}
interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: Source[];
  state:
    | "loading"
    | "streaming"
    | "answered"
    | "no_evidence"
    | "error"
    | "cancelled";
  error?: string;
  requestId?: string;
}
const HISTORY = "rag-local-upload-history-v1";
let sequence = 0;
const id = () =>
  globalThis.crypto?.randomUUID?.() || `${Date.now()}-${sequence++}`;
function readHistory(): UploadRecord[] {
  try {
    const data: unknown = JSON.parse(localStorage.getItem(HISTORY) || "[]");
    if (Array.isArray(data))
      return data
        .filter(
          (r) =>
            typeof r?.document_id === "string" &&
            typeof r?.filename === "string" &&
            ["indexed", "duplicate"].includes(r?.status) &&
            typeof r?.key === "string",
        )
        .slice(0, 20);
  } catch {
    /* 浏览器存储不可用不阻止上传和聊天。 */
  }
  return [];
}
const suggestions = [
  {
    title: "快速理解文档",
    text: "请概括已上传文档的核心内容。",
    description: "从长文档中抓住重点",
  },
  {
    title: "比较关键概念",
    text: "请比较资料中涉及的关键技术概念。",
    description: "把零散信息联系起来",
  },
  {
    title: "查找具体细节",
    text: "请列出资料中的重要参数，并提供依据。",
    description: "带着证据确认细节",
  },
];

export default function App() {
  const [records, setRecords] = useState<UploadRecord[]>(readHistory);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(true);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [small, setSmall] = useState(
    () => window.matchMedia("(max-width: 760px)").matches,
  );
  const [connection, setConnection] = useState<
    "checking" | "ready" | "initializing" | "offline"
  >("checking");
  const [connectionDetail, setConnectionDetail] = useState("");
  const [showScroll, setShowScroll] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null),
    textarea = useRef<HTMLTextAreaElement>(null);
  const sidebar = useRef<HTMLElement>(null),
    sidebarToggle = useRef<HTMLButtonElement>(null);
  const scroll = useRef<HTMLDivElement>(null),
    end = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null),
    checking = useRef<AbortController | null>(null);
  const busyRef = useRef(false),
    uploadRef = useRef(false),
    follow = useRef(true),
    mounted = useRef(true);

  const checkConnection = useCallback(async () => {
    checking.current?.abort();
    const controller = new AbortController();
    checking.current = controller;
    setConnection("checking");
    const timeout = window.setTimeout(() => controller.abort(), 5000);
    try {
      const status = await readiness(controller.signal);
      if (mounted.current && !controller.signal.aborted) {
        setConnection(status);
        setConnectionDetail("");
      }
    } catch (error) {
      if (mounted.current && checking.current === controller) {
        setConnection("offline");
        setConnectionDetail(
          error instanceof Error ? error.message : "连接失败",
        );
      }
    } finally {
      window.clearTimeout(timeout);
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    void checkConnection();
    const interval = window.setInterval(() => void checkConnection(), 15000);
    const media = window.matchMedia("(max-width: 760px)");
    const resize = () => setSmall(media.matches);
    media.addEventListener("change", resize);
    return () => {
      mounted.current = false;
      window.clearInterval(interval);
      media.removeEventListener("change", resize);
      checking.current?.abort();
      abort.current?.abort();
    };
  }, [checkConnection]);
  useEffect(() => {
    try {
      localStorage.setItem(
        HISTORY,
        JSON.stringify(
          records
            .filter(
              (r) =>
                r.document_id && ["indexed", "duplicate"].includes(r.status),
            )
            .slice(0, 20),
        ),
      );
    } catch {
      /* 只影响本机历史。 */
    }
  }, [records]);
  useEffect(() => {
    if (follow.current) end.current?.scrollIntoView({ block: "end" });
  }, [messages]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  useEffect(() => {
    if (!small || !sidebarOpen) return;
    sidebar.current
      ?.querySelector<HTMLButtonElement>('[aria-label="关闭知识库"]')
      ?.focus();
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const controls = sidebar.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled):not([tabindex="-1"])',
      );
      if (!controls?.length) return;
      const first = controls[0],
        last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", trapFocus);
    return () => {
      window.removeEventListener("keydown", trapFocus);
      sidebarToggle.current?.focus();
    };
  }, [small, sidebarOpen]);

  async function upload(files: File[]) {
    if (uploadRef.current) return;
    uploadRef.current = true;
    setUploading(true);
    setUploadError("");
    try {
      for (const file of files) {
        if (!/\.(pdf|txt|md|markdown)$/i.test(file.name)) {
          setUploadError("仅支持 PDF、TXT、Markdown 文件。");
          continue;
        }
        const key = id();
        setRecords((old) =>
          [
            {
              key,
              filename: file.name,
              status: "uploading" as const,
              createdAt: new Date().toISOString(),
            },
            ...old,
          ].slice(0, 20),
        );
        try {
          const result: UploadResult = await uploadDocument(file);
          setRecords((old) =>
            old.map((record) =>
              record.key === key ? { ...record, ...result } : record,
            ),
          );
        } catch (error) {
          const text =
            error instanceof Error ? error.message : "上传失败，请重试。";
          setUploadError(text);
          setRecords((old) =>
            old.map((record) =>
              record.key === key
                ? { ...record, status: "error", error: text }
                : record,
            ),
          );
        }
      }
    } finally {
      uploadRef.current = false;
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }
  function update(
    messageId: string,
    change: Partial<Message> | ((message: Message) => Partial<Message>),
  ) {
    setMessages((old) =>
      old.map((message) =>
        message.id === messageId
          ? {
              ...message,
              ...(typeof change === "function" ? change(message) : change),
            }
          : message,
      ),
    );
  }
  async function send() {
    const question = input.trim();
    if (!question || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setInput("");
    setSidebarOpen(false);
    follow.current = true;
    const answerId = id(),
      controller = new AbortController();
    abort.current = controller;
    setMessages((old) => [
      ...old,
      {
        id: id(),
        role: "user",
        text: question,
        sources: [],
        state: "answered",
      },
      {
        id: answerId,
        role: "assistant",
        text: "",
        sources: [],
        state: "loading",
      },
    ]);
    try {
      if (streaming)
        await streamChat(
          question,
          {
            onSources: (sources) => update(answerId, { sources }),
            onToken: (token) =>
              update(answerId, (message) => ({
                text: message.text + token,
                state: "streaming",
              })),
            onDone: (status) => update(answerId, { state: status }),
          },
          controller.signal,
        );
      else {
        const result = await chat(question, controller.signal);
        update(answerId, {
          text: result.answer,
          sources: result.sources,
          state: result.status,
        });
      }
    } catch (error) {
      if (controller.signal.aborted) update(answerId, { state: "cancelled" });
      else
        update(answerId, {
          state: "error",
          error: error instanceof Error ? error.message : "生成失败，请重试。",
          requestId: error instanceof ApiError ? error.requestId : undefined,
        });
    } finally {
      busyRef.current = false;
      setBusy(false);
      abort.current = null;
      textarea.current?.focus();
    }
  }
  function newConversation() {
    if (busyRef.current) return;
    setMessages([]);
    setInput("");
    setSidebarOpen(false);
    textarea.current?.focus();
  }
  const connectionText = {
    checking: "正在连接",
    ready: "知识库已就绪",
    initializing: "知识库初始化中",
    offline: "服务暂不可用",
  }[connection];

  return (
    <div
      className="app-shell"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => e.preventDefault()}
    >
      {small && sidebarOpen && (
        <button
          className="sidebar-scrim"
          aria-label="关闭知识库遮罩"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        ref={sidebar}
        id="knowledge-sidebar"
        role={small ? "dialog" : undefined}
        aria-modal={small && sidebarOpen ? true : undefined}
        className={`sidebar ${sidebarOpen ? "is-open" : ""}`}
        aria-label="知识库"
        aria-hidden={small && !sidebarOpen}
      >
        <div className="brand">
          <span className="brand-mark">
            <Layers size={23} />
          </span>
          <div>
            <strong>RAG Knowledge</strong>
            <span>ASSISTANT</span>
          </div>
          {small && (
            <button
              className="icon-button mobile-close"
              aria-label="关闭知识库"
              onClick={() => setSidebarOpen(false)}
            >
              <X size={19} />
            </button>
          )}
        </div>
        <button
          className="new-conversation"
          onClick={newConversation}
          disabled={busy}
        >
          <Plus size={17} />
          新建对话<span aria-hidden="true">↗</span>
        </button>
        <div className="sidebar-section-label">
          <span>知识库</span>
          <BookOpen size={14} />
        </div>
        <div
          className={`upload-zone ${dragging ? "dragging" : ""}`}
          aria-label="拖拽上传区域"
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (!uploading) void upload(Array.from(e.dataTransfer.files));
          }}
        >
          <span className="upload-symbol">
            {uploading ? (
              <Loader2 size={23} className="spin" />
            ) : (
              <Upload size={23} />
            )}
          </span>
          <strong>
            {uploading ? "文档正在加入知识库" : "把你的资料放在这里"}
          </strong>
          <p>拖拽文件到此处，或点击上传</p>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.markdown"
            aria-label="选择知识库文件"
            className="sr-only"
            tabIndex={-1}
            disabled={uploading}
            onChange={(e) => void upload(Array.from(e.target.files || []))}
          />
          <button
            className="upload-button"
            disabled={uploading}
            onClick={() => fileInput.current?.click()}
          >
            {uploading ? "正在入库" : "上传文档"}
          </button>
          <span className="format-hint">PDF · TXT · Markdown</span>
        </div>
        <p className="upload-help">文件大小及正文格式由服务端校验。</p>
        {uploadError && (
          <div className="upload-error" role="alert">
            <AlertCircle size={15} />
            <span>{uploadError}</span>
          </div>
        )}
        <div className="records-heading">
          <span>
            最近上传 <b>{records.length}</b>
          </span>
          <button
            disabled={uploading || !records.length}
            onClick={() => setRecords([])}
            title="仅清除本机记录，不删除知识库中的文档"
          >
            清除记录
          </button>
        </div>
        <div className="upload-records">
          {!records.length && (
            <div className="records-empty">
              <FileText size={21} />
              <p>还没有本机上传记录</p>
              <span>上传后即可开始提问</span>
            </div>
          )}
          {records.map((record) => (
            <div className={`upload-record ${record.status}`} key={record.key}>
              <div className="record-top">
                <FileText size={17} />
                <strong title={record.filename}>{record.filename}</strong>
                {record.status === "uploading" ? (
                  <Loader2 size={14} className="spin" />
                ) : record.status === "error" ? (
                  <AlertCircle size={14} />
                ) : (
                  <Check size={14} />
                )}
              </div>
              <div className="record-status">
                <span>
                  {
                    {
                      uploading: "正在入库",
                      indexed: "已入库",
                      duplicate: "已存在",
                      error: "上传失败",
                    }[record.status]
                  }
                </span>
                {record.chunk_count !== undefined && (
                  <span>{record.chunk_count} 个文本块</span>
                )}
              </div>
              {record.document_id && (
                <code className="record-id" title={record.document_id}>
                  {record.document_id}
                </code>
              )}
              {record.status === "duplicate" && (
                <p>相同内容已在知识库中，无需重复入库。</p>
              )}
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          <span className="tiny-brand">
            <Layers size={14} />
          </span>
          <div>
            从资料中获取可靠答案<small>本机记录不代表完整知识库列表</small>
          </div>
        </div>
      </aside>

      <main className="chat-main" inert={small && sidebarOpen}>
        <header className="chat-header">
          <div className="header-title">
            <button
              ref={sidebarToggle}
              className="icon-button sidebar-toggle"
              aria-label="打开知识库"
              aria-expanded={sidebarOpen}
              aria-controls="knowledge-sidebar"
              onClick={() => setSidebarOpen(true)}
            >
              <PanelLeft size={20} />
            </button>
            <div>
              <h1>知识库问答</h1>
              <p>基于你的文档，回答有据可循</p>
            </div>
          </div>
          <button
            className={`connection ${connection}`}
            aria-label="重试连接"
            aria-describedby="connection-status"
            title={connectionDetail || "点击重新检查服务状态"}
            onClick={() => void checkConnection()}
          >
            {connection === "checking" ? (
              <RefreshCw size={12} className="spin" />
            ) : (
              <i />
            )}
            <span id="connection-status" role="status">
              {connectionText}
            </span>
          </button>
        </header>
        <div
          ref={scroll}
          className="conversation-scroll"
          onScroll={() => {
            const node = scroll.current;
            if (node) {
              follow.current =
                node.scrollHeight - node.scrollTop - node.clientHeight < 90;
              setShowScroll(!follow.current);
            }
          }}
        >
          {!messages.length ? (
            <section className="welcome">
              <div className="welcome-eyebrow">
                <span />
                <span>YOUR KNOWLEDGE, CONNECTED</span>
              </div>
              <div className="welcome-icon">
                <Sparkles size={32} strokeWidth={1.6} />
              </div>
              <h2>让文档里的知识，成为你的答案。</h2>
              <p className="welcome-copy">
                上传资料，用自然语言提问。
                <br />
                每个回答都能回到原文，帮你理解得更快、确认得更清楚。
              </p>
              <div className="suggestions">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion.title}
                    onClick={() => {
                      setInput(suggestion.text);
                      textarea.current?.focus();
                    }}
                  >
                    <span>
                      <FileText size={17} />
                      <ChevronRight size={15} />
                    </span>
                    <strong>{suggestion.title}</strong>
                    <small>{suggestion.description}</small>
                  </button>
                ))}
              </div>
              <div className="welcome-note">
                <BookOpen size={14} />
                先上传一份文档，或直接查询现有知识库
              </div>
            </section>
          ) : (
            <div
              className="conversation"
              role="log"
              aria-live="polite"
              aria-relevant="additions text"
              aria-label="对话消息"
            >
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === "user" ? (
                    <>
                      <span className="message-role">你</span>
                      <div className="user-bubble">{message.text}</div>
                    </>
                  ) : (
                    <>
                      <div className="assistant-label">
                        <span className="assistant-avatar">
                          <Sparkles size={15} />
                        </span>
                        <strong>Knowledge Assistant</strong>
                        <span>基于知识库</span>
                      </div>
                      {message.state === "no_evidence" && (
                        <div className="evidence-notice">
                          <AlertCircle size={15} />
                          <strong>暂无检索证据</strong>
                          <span>可以补充相关文档，或提供更多问题背景。</span>
                        </div>
                      )}
                      <div className="answer-body">
                        {message.text ? (
                          <Markdown
                            skipHtml
                            components={{
                              a: (props) => (
                                <a
                                  {...props}
                                  target="_blank"
                                  rel="noreferrer noopener"
                                />
                              ),
                              img: () => <span>已省略外部图片</span>,
                            }}
                          >
                            {message.text}
                          </Markdown>
                        ) : (
                          message.state === "loading" && (
                            <span className="answer-loading">
                              <Loader2 size={15} className="spin" />
                              正在检索资料并组织回答…
                            </span>
                          )
                        )}
                        {message.state === "streaming" && (
                          <span
                            className="stream-cursor"
                            aria-label="正在生成"
                          />
                        )}
                      </div>
                      {message.state === "error" && (
                        <div className="message-error" role="alert">
                          <AlertCircle size={15} />
                          <div>
                            <strong>{message.error}</strong>
                            <p>已保留收到的内容。可重新输入问题后重试。</p>
                            {message.requestId && (
                              <code>请求 ID：{message.requestId}</code>
                            )}
                          </div>
                        </div>
                      )}
                      {message.state === "cancelled" && (
                        <p className="cancelled-note">
                          已停止生成，现有内容已保留。
                        </p>
                      )}
                      <Sources sources={message.sources} />
                    </>
                  )}
                </article>
              ))}
              <div ref={end} />
            </div>
          )}
        </div>
        <div className="composer-region">
          {showScroll && (
            <button
              className="scroll-latest"
              onClick={() => {
                follow.current = true;
                setShowScroll(false);
                end.current?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <ArrowDown size={14} />
              回到最新消息
            </button>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={textarea}
              aria-label="输入问题"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={2}
              placeholder="针对你的文档，问一个问题…"
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing &&
                  e.keyCode !== 229
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <div className="composer-bottom">
              <div className="composer-options">
                <label>
                  <input
                    type="checkbox"
                    checked={streaming}
                    onChange={(e) => setStreaming(e.target.checked)}
                    disabled={busy}
                  />
                  流式回答
                </label>
                <span className="enter-hint">
                  Enter 发送 · Shift + Enter 换行
                </span>
              </div>
              {busy ? (
                <button
                  type="button"
                  className="send-button stop"
                  aria-label="停止生成"
                  title="停止生成"
                  onClick={() => abort.current?.abort()}
                >
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  type="submit"
                  className="send-button"
                  aria-label="发送问题"
                  disabled={!input.trim()}
                >
                  <ArrowUp size={20} />
                </button>
              )}
            </div>
          </form>
          <p className="composer-disclaimer">
            每次提问独立检索知识库 · 请结合原文核对重要信息
          </p>
        </div>
      </main>
    </div>
  );
}
