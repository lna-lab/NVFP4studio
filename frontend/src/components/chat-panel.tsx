"use client";

import { useEffect, useRef, useState } from "react";

import type { ChatMessage, ChatSettings, SystemStatus, VllmRuntimeProfile } from "@/types";

function metric(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

function hasReasoningTrace(content: string): boolean {
  return content.includes("<think>") || content.includes("</think>") || content.startsWith("Thinking Process:");
}

function parseAssistantContent(content: string): { answer: string; reasoning: string | null } {
  if (!content) {
    return { answer: "", reasoning: null };
  }
  if (content.includes("</think>")) {
    const [before, after] = content.split("</think>");
    const reasoning = before.includes("<think>")
      ? before.split("<think>").pop()?.trim() ?? ""
      : before.trim();
    return {
      answer: after.trim(),
      reasoning: reasoning || null
    };
  }
  if (content.includes("<think>")) {
    return {
      answer: "",
      reasoning: content.split("<think>").pop()?.trim() ?? content.trim()
    };
  }
  if (content.startsWith("Thinking Process:")) {
    return {
      answer: "",
      reasoning: content.trim()
    };
  }
  return {
    answer: content.trim(),
    reasoning: null
  };
}

function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
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

function RangeControl({
  label,
  min,
  max,
  step,
  value,
  displayValue,
  onChange
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  displayValue?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-control">
      <div className="range-header">
        <span>{label}</span>
        <strong>{displayValue ?? value}</strong>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="range-legend">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </label>
  );
}

interface Props {
  modelName: string;
  runtimeMaxContext: number;
  sliderMaxContext: number;
  nativeMaxContext: number | null;
  settings: ChatSettings;
  appliedSettings: ChatSettings;
  advisory: SystemStatus["advisory"] | null;
  runtimeProfile: VllmRuntimeProfile;
  draftRuntimeProfile: VllmRuntimeProfile;
  pendingChanges: boolean;
  applyingRuntimeConfig: boolean;
  configError: string | null;
  setSettings: (settings: ChatSettings) => void;
  setRuntimeProfile: (profile: VllmRuntimeProfile) => void;
  onApplySettings: () => void;
  messages: ChatMessage[];
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  onSend: () => void;
  onStop: () => void;
  onClearConversation: () => void;
}

const PROMPT_PRESETS = [
  {
    label: "会話",
    prompt:
      "You are a concise local assistant. Reply in Japanese unless the user asks otherwise. Give clear final answers."
  },
  {
    label: "推論抑制",
    prompt:
      "You are a concise local assistant. Reply in Japanese unless the user asks otherwise. Do not reveal chain-of-thought, hidden reasoning, or 'Thinking Process'. Provide only the final answer."
  },
  {
    label: "ベンチ向け",
    prompt:
      "You are a benchmark-friendly assistant. Reply in Japanese unless the user asks otherwise. Keep answers compact, direct, and free of hidden reasoning."
  }
];

const RUNTIME_PROFILES: Array<{
  id: VllmRuntimeProfile;
  label: string;
  description: string;
}> = [
  {
    id: "speed",
    label: "高速",
    description: "速度優先。KV budget は auto で、予約 VRAM は厚めです。"
  },
  {
    id: "balanced",
    label: "バランス",
    description: "単ユーザー向けの標準。12G の明示 KV budget で VRAM と速度の中庸を狙います。"
  },
  {
    id: "memory",
    label: "省VRAM",
    description: "VRAM を節約。8G の明示 KV budget と fp8 KV cache で単ユーザー運用に寄せます。"
  }
];

export function ChatPanel({
  modelName,
  runtimeMaxContext,
  sliderMaxContext,
  nativeMaxContext,
  settings,
  appliedSettings,
  advisory,
  runtimeProfile,
  draftRuntimeProfile,
  pendingChanges,
  applyingRuntimeConfig,
  configError,
  setSettings,
  setRuntimeProfile,
  onApplySettings,
  messages,
  input,
  setInput,
  loading,
  onSend,
  onStop,
  onClearConversation
}: Props) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const maxTokensSliderMax = Math.max(64, Math.min(settings.contextLength, runtimeMaxContext));
  const maxTokensValue = Math.min(settings.maxTokens, maxTokensSliderMax);
  const recommendation =
    advisory?.recommended_context && advisory.recommended_context < runtimeMaxContext
      ? `推奨 ${advisory.recommended_context} tokens`
      : `実行上限 ${runtimeMaxContext} tokens`;

  return (
    <section className="chat-shell">
      <div className="panel hero-panel">
        <div className="panel-header">
          <div>
            <h1>NVFP4studio</h1>
            <span className="hero-copy">ローカル NVFP4 推論のためのランタイム兼ベンチコンソール</span>
          </div>
          <div className="hero-meta">
            <span className="badge ok">Model {modelName}</span>
            <span className="badge neutral">{appliedSettings.stream ? "Streaming ON" : "Streaming OFF"}</span>
            <span className={`badge ${pendingChanges ? "warn" : "ok"}`}>
              {pendingChanges ? "未適用の変更あり" : "設定適用済み"}
            </span>
          </div>
        </div>

        <div className="config-summary">
          <span>Context {appliedSettings.contextLength}</span>
          <span>Max Tokens {appliedSettings.maxTokens}</span>
          <span>Repeat {appliedSettings.repetitionPenalty.toFixed(2)}</span>
          <span>Thinking {appliedSettings.enableThinking ? "ON" : "OFF"}</span>
          <span>Profile {runtimeProfile}</span>
          <span>{recommendation}</span>
          {nativeMaxContext ? <span>モデル上限 {nativeMaxContext}</span> : null}
        </div>

        <div className="preset-row">
          {RUNTIME_PROFILES.map((profile) => (
            <button
              key={profile.id}
              type="button"
              className={`chip-button ${draftRuntimeProfile === profile.id ? "active" : ""}`}
              onClick={() => setRuntimeProfile(profile.id)}
            >
              {profile.label}
            </button>
          ))}
        </div>
        <div className="muted small-copy">
          {
            RUNTIME_PROFILES.find((profile) => profile.id === draftRuntimeProfile)?.description ??
            "vLLM runtime profile を選択してください。"
          }
        </div>

        <div className="controls-grid">
          <label>
            <span>System Prompt</span>
            <textarea
              value={settings.systemPrompt}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  systemPrompt: event.target.value
                })
              }
              rows={4}
            />
          </label>
          <div className="slider-stack">
            <RangeControl
              label="Context Length"
              min={2048}
              max={sliderMaxContext}
              step={256}
              value={settings.contextLength}
              displayValue={`${settings.contextLength} tokens`}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  contextLength: value,
                  maxTokens: Math.min(settings.maxTokens, value)
                })
              }
            />
            <RangeControl
              label="Max Tokens"
              min={64}
              max={maxTokensSliderMax}
              step={64}
              value={maxTokensValue}
              displayValue={`${maxTokensValue} tokens`}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  maxTokens: value
                })
              }
            />
            <RangeControl
              label="Temperature"
              min={0}
              max={2}
              step={0.05}
              value={settings.temperature}
              displayValue={formatNumber(settings.temperature)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  temperature: value
                })
              }
            />
            <RangeControl
              label="Top P"
              min={0.05}
              max={1}
              step={0.01}
              value={settings.topP}
              displayValue={formatNumber(settings.topP)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  topP: value
                })
              }
            />
            <RangeControl
              label="Repetition Penalty"
              min={1}
              max={1.5}
              step={0.01}
              value={settings.repetitionPenalty}
              displayValue={formatNumber(settings.repetitionPenalty)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  repetitionPenalty: value
                })
              }
            />
            <RangeControl
              label="Top K"
              min={0}
              max={200}
              step={1}
              value={settings.topK}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  topK: value
                })
              }
            />
          </div>
        </div>

        <div className="toggle-row">
          <label className="toggle-card">
            <span>Streaming</span>
            <input
              type="checkbox"
              checked={settings.stream}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  stream: event.target.checked
                })
              }
            />
          </label>
          <label className="toggle-card">
            <span>Thinking</span>
            <input
              type="checkbox"
              checked={settings.enableThinking}
              onChange={(event) =>
                setSettings({
                  ...settings,
                  enableThinking: event.target.checked
                })
              }
            />
          </label>
          <button type="button" className="chip-button" onClick={() => setAdvancedOpen((current) => !current)}>
            {advancedOpen ? "Advanced を閉じる" : "Advanced を開く"}
          </button>
          <button type="button" className="primary" onClick={onApplySettings} disabled={applyingRuntimeConfig}>
            {applyingRuntimeConfig ? "vLLM を再構成中..." : "設定を確定"}
          </button>
        </div>

        {advancedOpen ? (
          <div className="advanced-panel">
            <RangeControl
              label="Min P"
              min={0}
              max={0.5}
              step={0.01}
              value={settings.minP}
              displayValue={formatNumber(settings.minP)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  minP: value
                })
              }
            />
            <RangeControl
              label="Presence Penalty"
              min={-2}
              max={2}
              step={0.1}
              value={settings.presencePenalty}
              displayValue={formatNumber(settings.presencePenalty, 1)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  presencePenalty: value
                })
              }
            />
            <RangeControl
              label="Frequency Penalty"
              min={-2}
              max={2}
              step={0.1}
              value={settings.frequencyPenalty}
              displayValue={formatNumber(settings.frequencyPenalty, 1)}
              onChange={(value) =>
                setSettings({
                  ...settings,
                  frequencyPenalty: value
                })
              }
            />
          </div>
        ) : null}

        <div className="preset-row">
          {PROMPT_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              className="chip-button"
              onClick={() =>
                setSettings({
                  ...settings,
                  systemPrompt: preset.prompt
                })
              }
            >
              {preset.label}
            </button>
          ))}
        </div>

        {configError ? <div className="warning-note">{configError}</div> : null}
        {advisory ? (
          <div className={`runtime-note ${advisory.risk_level}`}>
            <strong>Runtime Advisory</strong>
            <span>{advisory.message}</span>
            <span>
              実行上限 {advisory.runtime_max_context ?? "N/A"} / モデル上限{" "}
              {advisory.model_native_context ?? "N/A"}
            </span>
            <span>
              Profile {advisory.runtime_profile} / GPU mem {advisory.gpu_memory_utilization ?? "N/A"} / seqs{" "}
              {advisory.max_num_seqs ?? "N/A"}
            </span>
            <span>
              KV budget {advisory.kv_cache_memory_bytes ?? "auto"} / safe cache{" "}
              {compactTokens(advisory.reserved_kv_capacity_tokens)} tokens
            </span>
            <span>
              KV usage {advisory.kv_cache_usage_percent ?? "N/A"}% / KV dtype {advisory.kv_cache_dtype ?? "N/A"}
            </span>
            <span>
              CPU offload {advisory.cpu_offload_detected ? "検出" : "なし"} / budget{" "}
              {advisory.cpu_offload_gb ?? "N/A"} GiB
            </span>
            {advisory.model_native_context &&
            advisory.runtime_max_context &&
            advisory.model_native_context > advisory.runtime_max_context ? (
              <span>
                もっと長い context を使うには Context Length を上げて「設定を確定」を押してください。
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="panel chat-panel">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Local NVFP4 Chat</h2>
              <p>設定を確定してから、そのまま会話かベンチを回せます。</p>
            </div>
          ) : (
            messages.map((message) => {
              const parsed =
                message.role === "assistant"
                  ? parseAssistantContent(message.content)
                  : { answer: message.content, reasoning: null };

              return (
                <article key={message.id} className={`bubble ${message.role}`}>
                  <header>
                    <span>{message.role}</span>
                    <time>{new Date(message.createdAt).toLocaleTimeString()}</time>
                  </header>
                  <div className="bubble-body">
                    {parsed.answer ||
                      (message.role === "assistant" && parsed.reasoning
                        ? "最終回答を生成中です。必要なら下の折りたたみから思考文を確認できます。"
                        : loading && message.role === "assistant"
                          ? "..."
                          : message.content)}
                  </div>
                  {message.role === "assistant" && hasReasoningTrace(message.content) ? (
                    <details className="reasoning-details">
                      <summary>思考の中身を表示</summary>
                      <div className="reasoning-body">{parsed.reasoning ?? message.content}</div>
                    </details>
                  ) : null}
                  {message.benchmark ? (
                    <footer className="bubble-metrics">
                      <span>TTFT {metric(message.benchmark.ttft_ms, "ms")}</span>
                      <span>tok/s {metric(message.benchmark.completion_tokens_per_sec)}</span>
                      <span>latency {metric(message.benchmark.e2e_latency_ms, "ms")}</span>
                      <span>
                        usage {message.benchmark.prompt_tokens ?? 0}/{message.benchmark.completion_tokens ?? 0}
                      </span>
                    </footer>
                  ) : null}
                  {message.error ? <div className="warning-note">{message.error}</div> : null}
                </article>
              );
            })
          )}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <textarea
            rows={4}
            value={input}
            placeholder="ここにプロンプトを入力。Enter で送信、Shift+Enter で改行。"
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSend();
              }
            }}
          />
          <div className="button-row">
            <button type="button" className="primary" onClick={onSend} disabled={loading || applyingRuntimeConfig}>
              {applyingRuntimeConfig ? "vLLM Reconfiguring" : "Send"}
            </button>
            <button type="button" onClick={onStop} disabled={!loading || applyingRuntimeConfig}>
              Stop
            </button>
            <button type="button" onClick={onClearConversation} disabled={applyingRuntimeConfig}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
