export type Role = "system" | "user" | "assistant";
export type VllmRuntimeProfile = "speed" | "balanced" | "memory";

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
  peak_power_watts?: number | null;
  peak_vram_used_mb?: number | null;
  power_limit_watts?: number | null;
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
  advisory: {
    runtime_max_context?: number | null;
    model_native_context?: number | null;
    runtime_profile: VllmRuntimeProfile;
    gpu_memory_utilization?: number | null;
    max_num_seqs?: number | null;
    max_num_batched_tokens?: number | null;
    kv_cache_dtype?: string | null;
    kv_cache_memory_bytes?: string | null;
    cpu_offload_gb?: number | null;
    swap_space_gb?: number | null;
    recommended_context?: number | null;
    hard_context_limit?: number | null;
    reserved_kv_capacity_tokens?: number | null;
    kv_cache_usage_percent?: number | null;
    cpu_offload_detected: boolean;
    fits_in_vram?: boolean | null;
    risk_level: string;
    message: string;
  };
  gpu: Array<{
    name: string;
    memory_total_mb?: number | null;
    memory_used_mb?: number | null;
    memory_free_mb?: number | null;
    utilization_gpu_percent?: number | null;
    power_draw_watts?: number | null;
    power_limit_watts?: number | null;
  }>;
}

export interface RuntimeConfigApplyResponse {
  accepted: boolean;
  restarted: boolean;
  previous_runtime_context?: number | null;
  applied_runtime_context?: number | null;
  previous_runtime_profile?: VllmRuntimeProfile | null;
  applied_runtime_profile?: VllmRuntimeProfile | null;
  validation: {
    requested_context: number;
    current_runtime_context?: number | null;
    model_native_context?: number | null;
    current_vram_used_mb?: number | null;
    total_vram_mb?: number | null;
    estimated_required_vram_mb?: number | null;
    fits_in_vram?: boolean | null;
    risk_level: string;
    message: string;
  };
  message: string;
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
  contextLength: number;
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
  enableThinking: boolean;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
}

export interface RuntimeConfigRequest {
  max_model_len: number;
  runtime_profile: VllmRuntimeProfile;
}

export interface ChatCompletionPayload {
  model: string;
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  messages: Array<{ role: Role; content: string }>;
  chat_template_kwargs?: {
    enable_thinking: boolean;
  };
}

export interface ModelDescriptor {
  id: string;
  max_model_len?: number;
  root?: string;
}

export interface BatchResultSummary {
  runs: number;
  completed: number;
  averageTtftMs: number | null;
  averageCompletionTokensPerSec: number | null;
  averageLatencyMs: number | null;
  lastError?: string;
}
