"use client";

import type { ChatMessage, ChatSettings } from "@/types";

function metric(value?: number | null, suffix = ""): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

interface Props {
  settings: ChatSettings;
  setSettings: (settings: ChatSettings) => void;
  messages: ChatMessage[];
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  onSend: () => void;
  onStop: () => void;
}

export function ChatPanel({
  settings,
  setSettings,
  messages,
  input,
  setInput,
  loading,
  onSend,
  onStop
}: Props) {
  return (
    <section className="chat-shell">
      <div className="panel hero-panel">
        <div className="panel-header">
          <h1>NVFP4studio</h1>
          <span className="hero-copy">LM Studio ライクなローカル NVFP4 コンソール</span>
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
          <div className="slider-grid">
            <label>
              <span>Temperature</span>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={settings.temperature}
                onChange={(event) =>
                  setSettings({
                    ...settings,
                    temperature: Number(event.target.value)
                  })
                }
              />
            </label>
            <label>
              <span>Top P</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={settings.topP}
                onChange={(event) =>
                  setSettings({
                    ...settings,
                    topP: Number(event.target.value)
                  })
                }
              />
            </label>
            <label>
              <span>Max Tokens</span>
              <input
                type="number"
                min={1}
                max={8192}
                step={1}
                value={settings.maxTokens}
                onChange={(event) =>
                  setSettings({
                    ...settings,
                    maxTokens: Number(event.target.value)
                  })
                }
              />
            </label>
            <label className="toggle">
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
          </div>
        </div>
      </div>

      <div className="panel chat-panel">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Local NVFP4 Chat</h2>
              <p>system prompt を整えてから、日本語でも英語でもそのまま投げられます。</p>
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`bubble ${message.role}`}>
                <header>
                  <span>{message.role}</span>
                  <time>{new Date(message.createdAt).toLocaleTimeString()}</time>
                </header>
                <div className="bubble-body">{message.content || (loading && message.role === "assistant" ? "..." : "")}</div>
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
                {message.error ? <footer className="error-text">{message.error}</footer> : null}
              </article>
            ))
          )}
        </div>

        <div className="composer">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={5}
            placeholder="ここにプロンプトを入力。Enter で送信、Shift+Enter で改行。"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSend();
              }
            }}
          />
          <div className="button-row">
            <button type="button" className="primary" onClick={onSend} disabled={loading || !input.trim()}>
              Send
            </button>
            <button type="button" onClick={onStop} disabled={!loading}>
              Stop
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

