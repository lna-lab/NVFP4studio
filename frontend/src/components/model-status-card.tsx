"use client";

import type { SystemConfig, SystemStatus } from "@/types";

function metricLabel(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value.toFixed(2)}${suffix}`;
}

function compactTokens(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(0)}K`;
  }
  return `${value}`;
}

interface Props {
  status: SystemStatus | null;
  config: SystemConfig | null;
  lastUpdatedAt: string | null;
  refreshing: boolean;
  onRefresh: () => void;
}

export function ModelStatusCard({ status, config, lastUpdatedAt, refreshing, onRefresh }: Props) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Model Status</h2>
        <div className="button-row">
          <button type="button" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
          <span className={status?.vllm_healthy ? "badge ok" : "badge warn"}>
            {status?.vllm_healthy ? "vLLM Ready" : "Waiting"}
          </span>
        </div>
      </div>

      <dl className="kv-grid">
        <div>
          <dt>Model</dt>
          <dd>{status?.served_model_name ?? config?.served_model_name ?? "..."}</dd>
        </div>
        <div>
          <dt>Model Path</dt>
          <dd className="mono tight">{status?.model_path ?? config?.model_path ?? "..."}</dd>
        </div>
        <div>
          <dt>Gateway</dt>
          <dd>{status?.gateway_ok ? "OK" : "Unknown"}</dd>
        </div>
        <div>
          <dt>Model Path Exists</dt>
          <dd>{status?.model_path_exists ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Metrics</dt>
          <dd>{status?.metrics_available ? "Enabled" : "Unavailable"}</dd>
        </div>
        <div>
          <dt>DB</dt>
          <dd className="mono tight">{status?.database_path ?? config?.database_url ?? "..."}</dd>
        </div>
        <div>
          <dt>Recent Records</dt>
          <dd>{status?.recent_benchmark_count ?? 0}</dd>
        </div>
        <div>
          <dt>Runtime Profile</dt>
          <dd>{status?.advisory.runtime_profile ?? "..."}</dd>
        </div>
        <div>
          <dt>API Key</dt>
          <dd>{config?.openai_api_key_hint ?? "..."}</dd>
        </div>
      </dl>

      <p className="muted small-copy">
        最終更新: {lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleTimeString() : "..."}
      </p>

      <div className="metrics-strip">
        <div>
          <span>Avg TTFT</span>
          <strong>{metricLabel(status?.metrics.values["vllm:time_to_first_token_seconds"], "s")}</strong>
        </div>
        <div>
          <span>Inter-token</span>
          <strong>
            {metricLabel(status?.metrics.values["vllm:inter_token_latency_seconds"], "s")}
          </strong>
        </div>
        <div>
          <span>E2E</span>
          <strong>{metricLabel(status?.metrics.values["vllm:e2e_request_latency_seconds"], "s")}</strong>
        </div>
      </div>

      {status?.advisory ? (
        <div className={`runtime-note ${status.advisory.risk_level}`}>
          <strong>VRAM Decision</strong>
          <span>{status.advisory.message}</span>
          <span>
            推奨 Context {status.advisory.recommended_context ?? "N/A"} / 実行上限{" "}
            {status.advisory.hard_context_limit ?? "N/A"}
          </span>
          <span>モデル上限 {status.advisory.model_native_context ?? "N/A"}</span>
          <span>
            KV budget {status.advisory.kv_cache_memory_bytes ?? "auto"} / safe cache{" "}
            {compactTokens(status.advisory.reserved_kv_capacity_tokens)} tokens
          </span>
          <span>
            GPU mem {metricLabel(status.advisory.gpu_memory_utilization)} / seqs{" "}
            {status.advisory.max_num_seqs ?? "N/A"} / batch{" "}
            {status.advisory.max_num_batched_tokens ?? "N/A"}
          </span>
          <span>
            KV usage {status.advisory.kv_cache_usage_percent ?? "N/A"}% / dtype{" "}
            {status.advisory.kv_cache_dtype ?? "N/A"}
          </span>
          <span>
            CPU offload {status.advisory.cpu_offload_detected ? "検出" : "なし"} / budget{" "}
            {metricLabel(status.advisory.cpu_offload_gb, " GiB")} / swap{" "}
            {metricLabel(status.advisory.swap_space_gb, " GiB")}
          </span>
        </div>
      ) : null}

      <div className="gpu-list">
        {(status?.gpu ?? []).length === 0 ? (
          <p className="muted">GPU 情報は取得できませんでした。</p>
        ) : (
          status?.gpu.map((gpu, index) => (
            <div key={`${gpu.name}-${index}`} className="gpu-card">
              <div className="gpu-title">{gpu.name}</div>
              <div className="gpu-stats">
                <span>{gpu.memory_used_mb ?? "?"} / {gpu.memory_total_mb ?? "?"} MB</span>
                <span>{gpu.utilization_gpu_percent ?? "?"}% util</span>
              </div>
              <div className="gpu-stats">
                <span>free {gpu.memory_free_mb ?? "?"} MB</span>
                <span>
                  {metricLabel(gpu.power_draw_watts, "W")} / {metricLabel(gpu.power_limit_watts, "W")}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
