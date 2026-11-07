// Browser-only test fixture. Never proxies to the real API or loads configuration.
// Run: node tests/mock-backend.mjs (127.0.0.1:18001).
import { createServer } from "node:http";

const source = {
  document_id: "browser-fixture-only",
  source: "前端测试夹具.md",
  chunk_id: 0,
  rank: 1,
  score: 0.875,
  content: "这是前端浏览器测试的模拟资料，不是真实知识库检索结果。",
};
const frame = (event, data) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const server = createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/ready") {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ status: "ready" }));
  }
  if (req.method !== "POST" || !["/chat", "/chat/stream"].includes(req.url)) {
    res.writeHead(404);
    return res.end();
  }
  let body = "";
  for await (const chunk of req) body += chunk;
  let question;
  try {
    question = JSON.parse(body).question;
  } catch {
    res.writeHead(400);
    return res.end();
  }
  const empty = question.includes("无证据");
  const text = empty
    ? "这是模拟拒答：现有资料不足，请补充相关文档。"
    : "这是前端模拟回答。\n\n**流式展示与引用展开**已通过测试夹具连接，不代表真实 LLM 联调。";
  if (req.url === "/chat") {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(
      JSON.stringify({
        answer: text,
        status: empty ? "no_evidence" : "answered",
        sources: empty ? [] : [source],
      }),
    );
  }
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
  });
  res.write(frame("sources", { sources: empty ? [] : [source] }));
  const tokens = text.match(/.{1,6}|\n/g) || [];
  let index = 0;
  const timer = setInterval(() => {
    if (res.destroyed) {
      clearInterval(timer);
      return;
    }
    if (question.includes("模拟中断") && index === 2) {
      res.end(frame("error", { message: "测试夹具：模拟流式中途失败。" }));
      clearInterval(timer);
      return;
    }
    if (index < tokens.length)
      res.write(frame("token", { token: tokens[index++] }));
    else {
      res.end(frame("done", empty ? { status: "no_evidence" } : {}));
      clearInterval(timer);
    }
  }, 120);
  res.on("close", () => clearInterval(timer));
});
server.listen(18001, "127.0.0.1", () =>
  console.log(
    "Mock-only browser fixture: http://127.0.0.1:18001 (no LLM calls)",
  ),
);
