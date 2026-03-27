"use client";

import { useEffect, useRef, useState } from "react";

import {
  applyRuntimeConfig,
  createChatCompletion,
  fetchBenchmarkByRequestId,
  fetchModels,
  fetchRecentBenchmarks,
  fetchSystemConfig,
  fetchSystemStatus,
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
  ModelDescriptor,
  SystemConfig,
  SystemStatus,
  VllmRuntimeProfile
} from "@/types";

const STORAGE_KEY = "nvfp4studio-state";

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

interface PersistedState {
  settings: ChatSettings;
  draftSettings?: ChatSettings;
  runtimeProfile?: VllmRuntimeProfile;
  draftRuntimeProfile?: VllmRuntimeProfile;
  messages: ChatMessage[];
  input: string;
}

function defaultSystemPrompt(): string {
  return "You are a concise local assistant. Reply in Japanese unless the user asks otherwise. Do not reveal chain-of-thought, hidden reasoning, or 'Thinking Process'. Provide only the final answer.";
}

function validateSettings(
  settings: ChatSettings,
  status: SystemStatus | null,
  modelMaxContext: number | null
): string | null {
  const hardLimit = modelMaxContext ?? status?.advisory?.model_native_context ?? null;
  const recommended = status?.advisory?.recommended_context ?? null;

  if (hardLimit !== null && settings.contextLength > hardLimit) {
    return `Context Length はモデル上限 ${hardLimit} tokens を超えています。`;
  }
  if (settings.contextLength < 2048) {
    return "Context Length は 2048 tokens 以上にしてください。";
  }
  if (settings.maxTokens > settings.contextLength) {
    return "Max Tokens は Context Length 以下にしてください。";
  }
  if (recommended !== null && settings.contextLength > recommended && status?.advisory?.risk_level !== "ok") {
    return `現在の状態では ${recommended} tokens までを推奨します。設定を見直してから再試行してください。`;
  }
  if (status?.advisory?.cpu_offload_detected) {
    return "CPU offload が検出されています。まずコンテキスト長か vLLM 設定を見直してください。";
  }
  return null;
}

function estimateTokenBudget(messages: ChatMessage[], input: string, settings: ChatSettings): number {
  const combined = [
    settings.systemPrompt,
    ...messages.map((message) => message.content),
    input
  ]
    .join("\n")
    .trim();

  if (!combined) {
    return 0;
  }

  return Math.ceil(combined.length / 2.5);
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
    top_k: settings.topK > 0 ? settings.topK : undefined,
    min_p: settings.minP > 0 ? settings.minP : undefined,
    repetition_penalty: settings.repetitionPenalty !== 1 ? settings.repetitionPenalty : undefined,
    presence_penalty: settings.presencePenalty !== 0 ? settings.presencePenalty : undefined,
    frequency_penalty: settings.frequencyPenalty !== 0 ? settings.frequencyPenalty : undefined,
    messages: finalMessages,
    chat_template_kwargs: {
      enable_thinking: settings.enableThinking
    }
  };
}

export function StudioShell() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelDescriptor | null>(null);
  const [modelName, setModelName] = useState("your-nvfp4-model");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recentBenchmarks, setRecentBenchmarks] = useState<BenchmarkRecord[]>([]);
  const [activeBenchmark, setActiveBenchmark] = useState<BenchmarkRecord | null>(null);
  const [batchSummary, setBatchSummary] = useState<BatchResultSummary | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [applyingRuntimeConfig, setApplyingRuntimeConfig] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChatSettings>({
    systemPrompt: defaultSystemPrompt(),
    contextLength: 8192,
    temperature: 0.7,
    topP: 0.95,
    maxTokens: 2048,
    stream: true,
    enableThinking: false,
    topK: 40,
    minP: 0,
    repetitionPenalty: 1.1,
    presencePenalty: 0,
    frequencyPenalty: 0
  });
  const [draftSettings, setDraftSettings] = useState<ChatSettings>({
    systemPrompt: defaultSystemPrompt(),
    contextLength: 8192,
    temperature: 0.7,
    topP: 0.95,
    maxTokens: 2048,
    stream: true,
    enableThinking: false,
    topK: 40,
    minP: 0,
    repetitionPenalty: 1.1,
    presencePenalty: 0,
    frequencyPenalty: 0
  });
  const [configError, setConfigError] = useState<string | null>(null);
  const [runtimeProfile, setRuntimeProfile] = useState<VllmRuntimeProfile>("speed");
  const [draftRuntimeProfile, setDraftRuntimeProfile] = useState<VllmRuntimeProfile>("speed");
  const abortRef = useRef<AbortController | null>(null);
  const loadedFromStorageRef = useRef(false);
  const configRef = useRef<SystemConfig | null>(null);
  const runtimeProfileRef = useRef<VllmRuntimeProfile>("speed");

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        loadedFromStorageRef.current = true;
        return;
      }
      const parsed = JSON.parse(raw) as Partial<PersistedState>;
      if (parsed.settings) {
        const mergedSettings = {
          ...settings,
          ...parsed.settings
        };
        setSettings(mergedSettings);
        setDraftSettings(parsed.draftSettings ? { ...mergedSettings, ...parsed.draftSettings } : mergedSettings);
      }
      if (Array.isArray(parsed.messages)) {
        setMessages(parsed.messages);
      }
      if (parsed.runtimeProfile) {
        setRuntimeProfile(parsed.runtimeProfile);
      }
      if (parsed.draftRuntimeProfile) {
        setDraftRuntimeProfile(parsed.draftRuntimeProfile);
      } else if (parsed.runtimeProfile) {
        setDraftRuntimeProfile(parsed.runtimeProfile);
      }
      if (typeof parsed.input === "string") {
        setInput(parsed.input);
      }
    } catch (error) {
      console.error(error);
    } finally {
      loadedFromStorageRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (!loadedFromStorageRef.current) {
      return;
    }
    const payload: PersistedState = {
      settings,
      draftSettings,
      runtimeProfile,
      draftRuntimeProfile,
      messages,
      input
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [draftRuntimeProfile, draftSettings, input, messages, runtimeProfile, settings]);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useEffect(() => {
    runtimeProfileRef.current = runtimeProfile;
  }, [runtimeProfile]);

  useEffect(() => {
    if (!status?.advisory || !modelInfo?.max_model_len) {
      return;
    }
    setDraftSettings((current) => {
      const maxContext = modelInfo.max_model_len ?? current.contextLength;
      const nextContext = Math.min(Math.max(current.contextLength, 2048), maxContext);
      const nextMaxTokens = Math.min(current.maxTokens, nextContext);
      if (nextContext === current.contextLength && nextMaxTokens === current.maxTokens) {
        return current;
      }
      return {
        ...current,
        contextLength: nextContext,
        maxTokens: nextMaxTokens
      };
    });
  }, [modelInfo?.max_model_len, status?.advisory]);

  async function refreshRuntimeData(options?: { full?: boolean }) {
    if (options?.full) {
      setRefreshing(true);
    }
    try {
      const [systemStatus, benchmarks] = await Promise.all([
        fetchSystemStatus(),
        fetchRecentBenchmarks()
      ]);
      setStatus(systemStatus);
      setRuntimeProfile(systemStatus.advisory.runtime_profile);
      setDraftRuntimeProfile((current) =>
        current === runtimeProfileRef.current ? systemStatus.advisory.runtime_profile : current
      );
      setRecentBenchmarks(benchmarks);
      setActiveBenchmark((current) => current ?? benchmarks[0] ?? null);

      if (options?.full || configRef.current === null) {
        const [systemConfig, models] = await Promise.all([
          fetchSystemConfig(),
          fetchModels().catch(() => ({ data: [] }))
        ]);
        setConfig(systemConfig);
        const detectedModel = models.data[0] ?? null;
        setModelInfo(detectedModel);
        setModelName(detectedModel?.id ?? systemConfig.served_model_name);
        if (detectedModel?.max_model_len) {
          const detectedMaxModelLen = detectedModel.max_model_len;
          setSettings((current) => ({
            ...current,
            contextLength: Math.min(current.contextLength, detectedMaxModelLen),
            maxTokens: Math.min(current.maxTokens, detectedMaxModelLen)
          }));
          setDraftSettings((current) => ({
            ...current,
            contextLength: Math.min(current.contextLength, detectedMaxModelLen),
            maxTokens: Math.min(current.maxTokens, detectedMaxModelLen)
          }));
        }
      }

      setLastUpdatedAt(new Date().toISOString());
    } catch (error) {
      console.error(error);
    } finally {
      if (options?.full) {
        setRefreshing(false);
      }
    }
  }

  async function waitForRuntimeSync(
    expectedContext: number,
    expectedProfile: VllmRuntimeProfile,
    timeoutMs = 180000
  ): Promise<SystemStatus> {
    const startedAt = Date.now();
    let lastStatus: SystemStatus | null = null;

    while (Date.now() - startedAt < timeoutMs) {
      const systemStatus = await fetchSystemStatus();
      lastStatus = systemStatus;
      if (
        systemStatus.vllm_healthy &&
        systemStatus.advisory.runtime_profile === expectedProfile &&
        systemStatus.advisory.runtime_max_context === expectedContext
      ) {
        return systemStatus;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }

    const statusDetail =
      lastStatus === null
        ? "status を取得できませんでした。"
        : `現在値 profile=${lastStatus.advisory.runtime_profile}, context=${lastStatus.advisory.runtime_max_context ?? "N/A"}, healthy=${lastStatus.vllm_healthy}`;
    throw new Error(`vLLM の再構成後同期がタイムアウトしました。${statusDetail}`);
  }

  useEffect(() => {
    let cancelled = false;
    void refreshRuntimeData({ full: true }).then(() => {
      if (cancelled) {
        return;
      }
    });
    const timer = window.setInterval(() => {
      void refreshRuntimeData();
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

    const validationMessage = validateSettings(
      settings,
      status,
      status?.advisory?.model_native_context ?? modelInfo?.max_model_len ?? null
    );
    if (validationMessage) {
      setConfigError(validationMessage);
      if (!options?.silent) {
        const assistantMessage: ChatMessage = {
          id: makeId("assistant"),
          role: "assistant",
          content: "設定が危険なためリクエストを送信しませんでした。",
          createdAt: new Date().toISOString(),
          error: validationMessage
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
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
    const estimatedPromptTokens = estimateTokenBudget(currentMessages, normalized, settings);
    const runtimeRequestBudget = Math.min(
      settings.contextLength,
      status?.advisory?.runtime_max_context ?? settings.contextLength
    );
    if (estimatedPromptTokens + settings.maxTokens > runtimeRequestBudget) {
      const budgetMessage =
        settings.contextLength > runtimeRequestBudget
          ? `現在の vLLM 実行上限は ${runtimeRequestBudget} tokens です。会話量推定 ${estimatedPromptTokens} prompt tokens と Max Tokens ${settings.maxTokens} の合計が収まりません。長い context を使うには vLLM の MAX_MODEL_LEN を上げて再起動してください。`
          : `現在の会話量だと推定 ${estimatedPromptTokens} prompt tokens です。Context Length ${settings.contextLength} に対して Max Tokens ${settings.maxTokens} が大きすぎます。設定を見直してください。`;
      setConfigError(budgetMessage);
      if (!options?.silent) {
        setMessages((prev) => [
          ...prev,
          {
            id: makeId("assistant"),
            role: "assistant",
            content: "推定コンテキスト予算を超えるため送信を止めました。",
            createdAt: new Date().toISOString(),
            error: budgetMessage
          }
        ]);
      }
      return null;
    }
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
    setConfigError(null);

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
    if (applyingRuntimeConfig) {
      return;
    }
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

  function handleClearConversation() {
    setMessages([]);
    setInput("");
    setActiveBenchmark(recentBenchmarks[0] ?? null);
    window.localStorage.removeItem(STORAGE_KEY);
  }

  function handleExport(format: "json" | "csv") {
    window.open(getExportUrl(format), "_blank", "noopener,noreferrer");
  }

  function handleApplySettings() {
    const validationMessage = validateSettings(
      draftSettings,
      status,
      status?.advisory?.model_native_context ?? modelInfo?.max_model_len ?? null
    );
    if (validationMessage) {
      setConfigError(validationMessage);
      return;
    }
    void (async () => {
      try {
        setApplyingRuntimeConfig(true);
        const currentRuntimeContext = status?.advisory?.runtime_max_context ?? null;
        const currentRuntimeProfile = status?.advisory?.runtime_profile ?? runtimeProfile;
        if (
          currentRuntimeContext !== draftSettings.contextLength ||
          currentRuntimeProfile !== draftRuntimeProfile
        ) {
          await applyRuntimeConfig({
            max_model_len: draftSettings.contextLength,
            runtime_profile: draftRuntimeProfile
          });
          await waitForRuntimeSync(draftSettings.contextLength, draftRuntimeProfile);
        }

        setSettings(draftSettings);
        setRuntimeProfile(draftRuntimeProfile);
        setConfigError(null);
        await refreshRuntimeData({ full: true });
      } catch (error) {
        setConfigError(error instanceof Error ? error.message : "runtime config apply failed");
      } finally {
        setApplyingRuntimeConfig(false);
      }
    })();
  }

  const pendingChanges =
    JSON.stringify(settings) !== JSON.stringify(draftSettings) || runtimeProfile !== draftRuntimeProfile;
  const runtimeMaxContext = status?.advisory?.runtime_max_context ?? modelInfo?.max_model_len ?? 8192;
  const sliderMaxContext = status?.advisory?.model_native_context ?? runtimeMaxContext;

  return (
    <main className="app-shell">
      <HistoryPanel
        messages={messages}
        recentBenchmarks={recentBenchmarks}
        onExport={handleExport}
        onClearMessages={handleClearConversation}
        onSelectBenchmark={setActiveBenchmark}
      />

      <div className="center-column">
        <ChatPanel
          modelName={modelName}
          runtimeMaxContext={runtimeMaxContext}
          sliderMaxContext={sliderMaxContext}
          nativeMaxContext={status?.advisory?.model_native_context ?? null}
          settings={draftSettings}
          appliedSettings={settings}
          advisory={status?.advisory ?? null}
          runtimeProfile={runtimeProfile}
          draftRuntimeProfile={draftRuntimeProfile}
          pendingChanges={pendingChanges}
          applyingRuntimeConfig={applyingRuntimeConfig}
          configError={configError}
          setSettings={setDraftSettings}
          setRuntimeProfile={setDraftRuntimeProfile}
          onApplySettings={handleApplySettings}
          messages={messages}
          input={input}
          setInput={setInput}
          loading={loading}
          onSend={handleSend}
          onStop={handleStop}
          onClearConversation={handleClearConversation}
        />
      </div>

      <div className="right-column">
        <ModelStatusCard
          status={status}
          config={config}
          lastUpdatedAt={lastUpdatedAt}
          refreshing={refreshing}
          onRefresh={() => void refreshRuntimeData({ full: true })}
        />
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
