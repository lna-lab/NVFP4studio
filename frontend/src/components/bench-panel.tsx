"use client";

import type { BatchResultSummary, BenchmarkRecord } from "@/types";

function average(values: Array<number | null | undefined>): number | null {
  const filtered = values.filter((value): value is number => typeof value === "number");
  if (filtered.length === 0) {
    return null;
  }
  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function metric(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

interface Props {
  activeBenchmark: BenchmarkRecord | null;
  recentBenchmarks: BenchmarkRecord[];
  batchSummary: BatchResultSummary | null;
  onRunPreset: (prompt: string, runs: number) => void;
}

const PRESETS = [
  { label: "短文", prompt: "2文で自己紹介してください。" },
  { label: "日本語長文", prompt: "日本の地方創生について600字程度で要点を整理してください。" },
  { label: "英語長文", prompt: "Explain the tradeoffs of local LLM serving in about 300 words." },
  { label: "コード生成", prompt: "Write a Python function that merges overlapping intervals." }
];

export function BenchPanel({
  activeBenchmark,
  recentBenchmarks,
  batchSummary,
  onRunPreset
}: Props) {
  const avgTtft = average(recentBenchmarks.map((item) => item.ttft_ms));
  const avgTok = average(recentBenchmarks.map((item) => item.completion_tokens_per_sec));
  const avgLat = average(recentBenchmarks.map((item) => item.e2e_latency_ms));

  return (
    <aside className="sidebar">
      <section className="panel">
        <div className="panel-header">
          <h2>Current Benchmark</h2>
        </div>
        {!activeBenchmark ? (
          <p className="muted">応答後に TTFT / token/s / latency を表示します。</p>
        ) : (
          <dl className="kv-grid compact">
            <div>
              <dt>TTFT</dt>
              <dd>{metric(activeBenchmark.ttft_ms, "ms")}</dd>
            </div>
            <div>
              <dt>Completion tok/s</dt>
              <dd>{metric(activeBenchmark.completion_tokens_per_sec)}</dd>
            </div>
            <div>
              <dt>Total latency</dt>
              <dd>{metric(activeBenchmark.e2e_latency_ms, "ms")}</dd>
            </div>
            <div>
              <dt>Usage</dt>
              <dd>
                {activeBenchmark.prompt_tokens ?? 0} / {activeBenchmark.completion_tokens ?? 0}
              </dd>
            </div>
            <div>
              <dt>Finish</dt>
              <dd>{activeBenchmark.finish_reason ?? "N/A"}</dd>
            </div>
            <div>
              <dt>Request ID</dt>
              <dd className="mono tight">{activeBenchmark.request_id}</dd>
            </div>
          </dl>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Recent Avg</h2>
        </div>
        <div className="metrics-strip vertical">
          <div>
            <span>Avg TTFT</span>
            <strong>{metric(avgTtft, "ms")}</strong>
          </div>
          <div>
            <span>Avg Completion tok/s</span>
            <strong>{metric(avgTok)}</strong>
          </div>
          <div>
            <span>Avg Latency</span>
            <strong>{metric(avgLat, "ms")}</strong>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Preset Bench</h2>
        </div>
        <div className="preset-list">
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              className="preset-button"
              onClick={() => onRunPreset(preset.prompt, 3)}
            >
              <strong>{preset.label}</strong>
              <span>{preset.prompt}</span>
            </button>
          ))}
        </div>
        {batchSummary ? (
          <div className="batch-summary">
            <div>{batchSummary.completed} / {batchSummary.runs} runs completed</div>
            <div>TTFT avg: {metric(batchSummary.averageTtftMs, "ms")}</div>
            <div>tok/s avg: {metric(batchSummary.averageCompletionTokensPerSec)}</div>
            <div>latency avg: {metric(batchSummary.averageLatencyMs, "ms")}</div>
            {batchSummary.lastError ? <div className="error-text">{batchSummary.lastError}</div> : null}
          </div>
        ) : null}
      </section>
    </aside>
  );
}

