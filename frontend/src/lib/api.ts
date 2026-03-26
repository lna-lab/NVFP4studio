import type {
  BenchmarkRecord,
  ChatCompletionPayload,
  SystemConfig,
  SystemStatus
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store"
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function getApiBase(): string {
  return API_BASE;
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  return fetchJson<SystemStatus>("/api/system/status");
}

export async function fetchSystemConfig(): Promise<SystemConfig> {
  return fetchJson<SystemConfig>("/api/system/config");
}

export async function fetchRecentBenchmarks(limit = 20): Promise<BenchmarkRecord[]> {
  const payload = await fetchJson<{ items: BenchmarkRecord[] }>(
    `/api/benchmarks/recent?limit=${limit}`
  );
  return payload.items;
}

export async function fetchBenchmarkByRequestId(requestId: string): Promise<BenchmarkRecord> {
  return fetchJson<BenchmarkRecord>(`/api/benchmarks/request/${requestId}`);
}

export async function fetchModels(): Promise<{ data: Array<{ id: string }> }> {
  return fetchJson<{ data: Array<{ id: string }> }>("/v1/models");
}

export async function createChatCompletion(
  payload: ChatCompletionPayload,
  signal?: AbortSignal
): Promise<{ body: any; requestId: string | null }> {
  const response = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return {
    body: await response.json(),
    requestId: response.headers.get("x-nvfp4studio-request-id")
  };
}

interface StreamCallbacks {
  onRequestId?: (requestId: string | null) => void;
  onChunk?: (chunk: any) => void;
}

export async function streamChatCompletion(
  payload: ChatCompletionPayload,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<string | null> {
  const response = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    signal
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  const requestId = response.headers.get("x-nvfp4studio-request-id");
  callbacks.onRequestId?.(requestId);

  const reader = response.body?.getReader();
  if (!reader) {
    return requestId;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n");
    buffer = parts.pop() ?? "";

    for (const line of parts) {
      if (!line.startsWith("data: ")) {
        continue;
      }
      const data = line.slice(6).trim();
      if (!data || data === "[DONE]") {
        continue;
      }
      callbacks.onChunk?.(JSON.parse(data));
    }
  }

  if (buffer.startsWith("data: ")) {
    const data = buffer.slice(6).trim();
    if (data && data !== "[DONE]") {
      callbacks.onChunk?.(JSON.parse(data));
    }
  }

  return requestId;
}

export function getExportUrl(format: "json" | "csv"): string {
  return `${API_BASE}/api/benchmarks/export?format=${format}`;
}

