import { useState } from "react";
import { ChevronRight, Copy, FileText, Quote } from "lucide-react";
import type { Source } from "./api";

export default function Sources({ sources }: { sources: Source[] }) {
  const [copied, setCopied] = useState("");
  if (!sources.length) return null;
  async function copy(id: string) {
    try {
      await navigator.clipboard.writeText(id);
      setCopied("已复制来源文档 ID");
    } catch {
      setCopied("无法自动复制，请选择下方文档 ID 手动复制。");
    }
  }
  return (
    <section className="sources" aria-label="本次回答的来源">
      <div className="sources-heading">
        <Quote size={13} /> 参考来源 <span>{sources.length}</span>
      </div>
      {sources.map((source) => (
        <details
          className="source-card"
          key={`${source.document_id}:${source.chunk_id}:${source.rank}`}
        >
          <summary>
            <FileText size={16} />
            <strong title={source.source}>{source.source}</strong>
            <span className="source-page">
              {source.page
                ? `第 ${source.page} 页`
                : `Chunk ${source.chunk_id ?? "未知"}`}
            </span>
            <ChevronRight size={14} className="detail-chevron" />
          </summary>
          <div className="source-expanded">
            <div className="source-metrics">
              <span>排名 #{source.rank}</span>
              <span>评分 {source.score.toFixed(3)}</span>
              <span>Chunk {source.chunk_id ?? "未知"}</span>
            </div>
            <div className="source-id">
              <code>{source.document_id}</code>
              <button
                type="button"
                aria-label="复制来源文档 ID"
                title="复制来源文档 ID"
                onClick={() => void copy(source.document_id)}
              >
                <Copy size={13} />
              </button>
            </div>
            <pre className="source-content">{source.content}</pre>
            <p className="source-note">
              以上为本次实际使用的原文；评分用于排序，不代表答案正确率。
            </p>
          </div>
        </details>
      ))}
      {copied && (
        <span className="sr-only" role="status">
          {copied}
        </span>
      )}
    </section>
  );
}
