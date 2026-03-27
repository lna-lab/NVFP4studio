"use client";

import type { BenchmarkRecord, ChatMessage } from "@/types";

function metric(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

function previewContent(content: string): string {
  if (!content) {
    return "Streaming...";
  }
  if (content.includes("</think>")) {
    return content.split("</think>").pop()?.trim() || "Streaming...";
  }
  if (content.includes("<think>") || content.startsWith("Thinking Process:")) {
    return "思考中...";
  }
  return content.replace(/^<think>\s*/i, "").trim() || "Streaming...";
}

interface Props {
  messages: ChatMessage[];
  recentBenchmarks: BenchmarkRecord[];
  onExport: (format: "json" | "csv") => void;
  onClearMessages: () => void;
  onSelectBenchmark: (benchmark: BenchmarkRecord) => void;
}

export function HistoryPanel({
  messages,
  recentBenchmarks,
  onExport,
  onClearMessages,
  onSelectBenchmark
}: Props) {
  const assistantMessages = messages.filter((message) => message.role === "assistant");

  return (
    <aside className="sidebar">
      <section className="panel">
        <div className="panel-header">
          <h2>Sessions</h2>
          <div className="button-row">
            <button type="button" onClick={() => onExport("json")}>
              JSON
            </button>
            <button type="button" onClick={() => onExport("csv")}>
              CSV
            </button>
            <button type="button" onClick={onClearMessages} disabled={messages.length === 0}>
              Clear
            </button>
          </div>
        </div>

        <div className="history-list">
          {assistantMessages.length === 0 ? (
            <p className="muted">まだ会話がありません。</p>
          ) : (
            assistantMessages.map((message) => (
              <button
                key={message.id}
                type="button"
                className="history-item history-button"
                onClick={() => {
                  if (message.benchmark) {
                    onSelectBenchmark(message.benchmark);
                  }
                }}
              >
                <div className="history-title">
                  {previewContent(message.content).slice(0, 64)}
                </div>
                <div className="history-meta">
                  <span>{metric(message.benchmark?.ttft_ms, "ms")}</span>
                  <span>{metric(message.benchmark?.completion_tokens_per_sec, " tok/s")}</span>
                </div>
              </button>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Recent Benchmarks</h2>
        </div>
        <div className="history-list">
          {recentBenchmarks.length === 0 ? (
            <p className="muted">履歴はまだありません。</p>
          ) : (
            recentBenchmarks.map((item) => (
              <button
                key={item.id}
                type="button"
                className="history-item history-button"
                onClick={() => onSelectBenchmark(item)}
              >
                <div className="history-title">{item.model_name}</div>
                <div className="history-meta">
                  <span>TTFT {metric(item.ttft_ms, "ms")}</span>
                  <span>LAT {metric(item.e2e_latency_ms, "ms")}</span>
                </div>
              </button>
            ))
          )}
        </div>
      </section>
    </aside>
  );
}
