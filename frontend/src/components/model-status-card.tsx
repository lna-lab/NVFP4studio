"use client";

import type { SystemConfig, SystemStatus } from "@/types";

function metricLabel(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value.toFixed(2)}${suffix}`;
}

interface Props {
  status: SystemStatus | null;
  config: SystemConfig | null;
}

export function ModelStatusCard({ status, config }: Props) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Model Status</h2>
        <span className={status?.vllm_healthy ? "badge ok" : "badge warn"}>
          {status?.vllm_healthy ? "vLLM Ready" : "Waiting"}
        </span>
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
          <dt>Metrics</dt>
          <dd>{status?.metrics_available ? "Enabled" : "Unavailable"}</dd>
        </div>
        <div>
          <dt>DB</dt>
          <dd className="mono tight">{status?.database_path ?? config?.database_url ?? "..."}</dd>
        </div>
        <div>
          <dt>API Key</dt>
          <dd>{config?.openai_api_key_hint ?? "..."}</dd>
        </div>
      </dl>

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
            </div>
          ))
        )}
      </div>
    </section>
  );
}

