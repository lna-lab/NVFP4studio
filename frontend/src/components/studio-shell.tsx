"use client";

import { useEffect, useRef, useState } from "react";

import {
  createChatCompletion,
  fetchBenchmarkByRequestId,
  fetchModels,
  fetchRecentBenchmarks,
  fetchSystemConfig,
  fetchSystemStatus,
  getApiBase,
  getExportUrl,
  streamChatCompletion
} from "@/lib/api";
import { BenchPanel } from "@/components/bench-panel";
import { ChatPanel } from "@/components/chat-panel";
import { HistoryPanel } from "@/components/history-panel";
import { ModelStatusCard } from "@/components/model-status-card";
import type {
  BatchResultSummary,
  BenchmarkRecord,
  ChatCompletionPayload,
  ChatMessage,
  ChatSettings,
  SystemConfig,
  SystemStatus
} from "@/types";

function makeId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function assistantContentFromChunk(chunk: any): string {
  const choice = chunk?.choices?.[0];
  const content = choice?.delta?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => (part?.type === "text" ? part.text ?? "" : ""))
      .join("");
  }
  if (typeof choice?.delta?.reasoning_content === "string") {
    return choice.delta.reasoning_content;
  }
  return "";
}

function assistantContentFromResponse(body: any): string {
  const content = body?.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part: any) => (part?.type === "text" ? part.text ?? "" : ""))
      .join("");
  }
  return "";
}

function buildPayload(
  messages: ChatMessage[],
  input: string,
  settings: ChatSettings,
  modelName: string
): ChatCompletionPayload {
  const apiMessages = messages
    .filter((message) => message.role !== "assistant" || message.content.trim())
    .map((message) => ({
      role: message.role,
      content: message.content
    }));

  const finalMessages = [
    ...(settings.systemPrompt.trim()
      ? [{ role: "system" as const, content: settings.systemPrompt.trim() }]
      : []),
    ...apiMessages,
    { role: "user" as const, content: input }
  ];

  return {
    model: modelName,
    stream: settings.stream,
    temperature: settings.temperature,
    top_p: settings.topP,
    max_tokens: settings.maxTokens,
    messages: finalMessages
  };
}

export function StudioShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [modelName, setModelName] = useState("Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recentBenchmarks, setRecentBenchmarks] = useState<BenchmarkRecord[]>([]);
  const [activeBenchmark, setActiveBenchmark] = useState<BenchmarkRecord | null>(null);
  const [batchSummary, setBatchSummary] = useState<BatchResultSummary | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    systemPrompt:
      "You are a local NVFP4 assistant. Respond clearly, concisely, and preserve benchmark friendliness.",
    temperature: 0.7,
    topP: 0.95,
    maxTokens: 2048,
    stream: true
  });
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function hydrate() {
      try {
        const [systemStatus, systemConfig, models, benchmarks] = await Promise.all([
          fetchSystemStatus(),
          fetchSystemConfig(),
          fetchModels().catch(() => ({ data: [] })),
          fetchRecentBenchmarks()
        ]);
        if (cancelled) {
          return;
        }
        setStatus(systemStatus);
        setConfig(systemConfig);
        setRecentBenchmarks(benchmarks);
        setActiveBenchmark(benchmarks[0] ?? null);
        setSettings((current) => ({
          ...current,
          temperature: systemConfig.default_temperature,
          topP: systemConfig.default_top_p,
          maxTokens: systemConfig.default_max_tokens
        }));
        setModelName(models.data[0]?.id ?? systemConfig.served_model_name);
      } catch (error) {
        console.error(error);
      }
    }

    hydrate();
    const timer = window.setInterval(() => {
      void fetchSystemStatus().then(setStatus).catch(() => undefined);
      void fetchRecentBenchmarks().then(setRecentBenchmarks).catch(() => undefined);
    }, 10000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  async function refreshBenchmarks(requestId?: string | null): Promise<BenchmarkRecord | null> {
    const [benchmarks, benchmark] = await Promise.all([
      fetchRecentBenchmarks(),
      requestId ? fetchBenchmarkByRequestId(requestId).catch(() => null) : Promise.resolve(null)
    ]);
    setRecentBenchmarks(benchmarks);
    if (benchmark) {
      setActiveBenchmark(benchmark);
    } else if (benchmarks[0]) {
      setActiveBenchmark(benchmarks[0]);
    }
    return benchmark;
  }

async function submitPrompt(prompt: string, options?: { silent?: boolean }): Promise<BenchmarkRecord | null> {
    const normalized = prompt.trim();
    if (!normalized || loading) {
      return null;
    }

    const userMessage: ChatMessage = {
      id: makeId("user"),
      role: "user",
      content: normalized,
      createdAt: new Date().toISOString()
    };
    const assistantMessageId = makeId("assistant");
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString()
    };

    const currentMessages = options?.silent ? messages : [...messages, userMessage];
    if (!options?.silent) {
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
    }

    const effectiveSettings = {
      ...settings,
      stream: options?.silent ? true : settings.stream
    };
    const payload = buildPayload(currentMessages, normalized, effectiveSettings, modelName);
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);

    try {
      if (effectiveSettings.stream) {
        let streamRequestId: string | null = null;
        await streamChatCompletion(
          payload,
          {
            onRequestId: (requestId) => {
              streamRequestId = requestId;
              if (!options?.silent) {
                setMessages((prev) =>
                  prev.map((message) =>
                    message.id === assistantMessageId
                      ? { ...message, requestId: requestId ?? undefined }
                      : message
                  )
                );
              }
            },
            onChunk: (chunk) => {
              const delta = assistantContentFromChunk(chunk);
              if (!delta || options?.silent) {
                return;
              }
              setMessages((prev) =>
                prev.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        content: `${message.content}${delta}`
                      }
                    : message
                )
              );
            }
          },
          controller.signal
        );
        const benchmark = await refreshBenchmarks(streamRequestId);
        if (!options?.silent) {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessageId ? { ...message, benchmark } : message
            )
          );
        }
        return benchmark;
      }

      const result = await createChatCompletion(payload, controller.signal);
      const content = assistantContentFromResponse(result.body);
      const benchmark = await refreshBenchmarks(result.requestId);
      if (!options?.silent) {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content,
                  requestId: result.requestId ?? undefined,
                  benchmark
                }
              : message
          )
        );
      }
      return benchmark;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      if (!options?.silent) {
        setMessages((prev) =>
          prev.map((item) =>
            item.id === assistantMessageId
              ? {
                  ...item,
                  error: message,
                  content: item.content || "リクエストが失敗しました。"
                }
              : item
          )
        );
      }
      throw error;
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  async function handleSend() {
    const prompt = input.trim();
    if (!prompt) {
      return;
    }
    setInput("");
    try {
      await submitPrompt(prompt);
    } catch (error) {
      console.error(error);
    }
  }

  async function handleRunPreset(prompt: string, runs: number) {
    setBatchSummary({
      runs,
      completed: 0,
      averageTtftMs: null,
      averageCompletionTokensPerSec: null,
      averageLatencyMs: null
    });

    const ttftValues: number[] = [];
    const tokValues: number[] = [];
    const latValues: number[] = [];

    for (let index = 0; index < runs; index += 1) {
      try {
        const benchmark = await submitPrompt(prompt, { silent: true });
        if (benchmark?.ttft_ms) {
          ttftValues.push(benchmark.ttft_ms);
        }
        if (benchmark?.completion_tokens_per_sec) {
          tokValues.push(benchmark.completion_tokens_per_sec);
        }
        if (benchmark?.e2e_latency_ms) {
          latValues.push(benchmark.e2e_latency_ms);
        }
        setBatchSummary({
          runs,
          completed: index + 1,
          averageTtftMs: ttftValues.length ? ttftValues.reduce((a, b) => a + b, 0) / ttftValues.length : null,
          averageCompletionTokensPerSec: tokValues.length
            ? tokValues.reduce((a, b) => a + b, 0) / tokValues.length
            : null,
          averageLatencyMs: latValues.length ? latValues.reduce((a, b) => a + b, 0) / latValues.length : null
        });
      } catch (error) {
        setBatchSummary((current) => ({
          runs,
          completed: current?.completed ?? index,
          averageTtftMs: current?.averageTtftMs ?? null,
          averageCompletionTokensPerSec: current?.averageCompletionTokensPerSec ?? null,
          averageLatencyMs: current?.averageLatencyMs ?? null,
          lastError: error instanceof Error ? error.message : "benchmark failed"
        }));
        break;
      }
    }
  }

  function handleStop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  function handleExport(format: "json" | "csv") {
    window.open(getExportUrl(format), "_blank", "noopener,noreferrer");
  }

  return (
    <main className="app-shell">
      <HistoryPanel
        messages={messages}
        recentBenchmarks={recentBenchmarks}
        onExport={handleExport}
      />

      <div className="center-column">
        <ChatPanel
          settings={settings}
          setSettings={setSettings}
          messages={messages}
          input={input}
          setInput={setInput}
          loading={loading}
          onSend={handleSend}
          onStop={handleStop}
        />
      </div>

      <div className="right-column">
        <ModelStatusCard status={status} config={config} />
        <BenchPanel
          activeBenchmark={activeBenchmark}
          recentBenchmarks={recentBenchmarks}
          batchSummary={batchSummary}
          onRunPreset={handleRunPreset}
        />
      </div>
    </main>
  );
}
