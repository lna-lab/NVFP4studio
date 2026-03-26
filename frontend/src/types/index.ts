export type Role = "system" | "user" | "assistant";

export interface BenchmarkRecord {
  id: number;
  request_id: string;
  upstream_request_id?: string | null;
  model_name: string;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  started_at: string;
  first_token_at?: string | null;
  finished_at: string;
  ttft_ms?: number | null;
  e2e_latency_ms?: number | null;
  completion_tokens_per_sec?: number | null;
  total_tokens_per_sec?: number | null;
  streaming: boolean;
  temperature?: number | null;
  top_p?: number | null;
  max_tokens?: number | null;
  finish_reason?: string | null;
  error_message?: string | null;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  requestId?: string;
  benchmark?: BenchmarkRecord | null;
  error?: string | null;
}

export interface SystemStatus {
  gateway_ok: boolean;
  vllm_healthy: boolean;
  model_path_exists: boolean;
  model_path: string;
  served_model_name: string;
  database_path: string;
  metrics_available: boolean;
  recent_benchmark_count: number;
  metrics: {
    values: Record<string, number>;
  };
  gpu: Array<{
    name: string;
    memory_total_mb?: number | null;
    memory_used_mb?: number | null;
    utilization_gpu_percent?: number | null;
  }>;
}

export interface SystemConfig {
  model_path: string;
  served_model_name: string;
  vllm_base_url: string;
  gateway_port: number;
  web_port: number;
  database_url: string;
  default_max_tokens: number;
  default_temperature: number;
  default_top_p: number;
  enable_vllm_metrics: boolean;
  bind_localhost_only: boolean;
  web_origin: string[];
  openai_api_key_hint: string;
}

export interface ChatSettings {
  systemPrompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
}

export interface ChatCompletionPayload {
  model: string;
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  messages: Array<{ role: Role; content: string }>;
}

export interface BatchResultSummary {
  runs: number;
  completed: number;
  averageTtftMs: number | null;
  averageCompletionTokensPerSec: number | null;
  averageLatencyMs: number | null;
  lastError?: string;
}

