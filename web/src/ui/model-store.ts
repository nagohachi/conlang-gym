/** The model's loading state, shared by every component that runs it. */
import { useEffect, useState } from "preact/hooks";

import { LlmClient, type LoadProgress } from "../llm/client";

/** Where the exported model (web/scripts/export_onnx.py) is served from; see vite.config.ts. */
export const MODEL_URL: string =
  import.meta.env.VITE_MODEL_URL ||
  (import.meta.env.DEV
    ? `${import.meta.env.BASE_URL}models/conlang-gym-2b-v2/`
    : "https://huggingface.co/nagohachi/Qwen3.5-2B-conlang-gym-v2-plural-past-with-dictionary-ONNX/resolve/main/");

export const MODEL_REPO = "nagohachi/Qwen3.5-2B-conlang-gym-v2-plural-past-with-dictionary";

export type ModelStatus = "unchecked" | "unsupported" | "idle" | "loading" | "ready" | "error";

export interface ModelState {
  status: ModelStatus;
  progress: LoadProgress | null;
  /** Why WebGPU cannot run the model, or the load error. */
  message: string | null;
}

let state: ModelState = { status: "unchecked", progress: null, message: null };
const listeners = new Set<() => void>();
let client: LlmClient | null = null;

function update(patch: Partial<ModelState>): void {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
}

export function useModelState(): ModelState {
  const [, setTick] = useState(0);
  useEffect(() => {
    const rerender = () => setTick((tick) => tick + 1);
    listeners.add(rerender);
    if (state.status === "unchecked") void checkSupport();
    return () => void listeners.delete(rerender);
  }, []);
  return state;
}

/** Find out whether this browser can run the q4f16 model on WebGPU. */
async function checkSupport(): Promise<void> {
  if (!("gpu" in navigator)) {
    update({ status: "unsupported", message: "This browser has no WebGPU. Try a recent Chrome or Edge." });
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      update({ status: "unsupported", message: "WebGPU is present but no GPU adapter is available." });
    } else if (!adapter.features.has("shader-f16")) {
      update({
        status: "unsupported",
        message: "This GPU lacks the WebGPU shader-f16 feature that the 16-bit model needs.",
      });
    } else {
      update({ status: "idle" });
    }
  } catch (error) {
    update({ status: "unsupported", message: String(error) });
  }
}

export async function loadModel(): Promise<void> {
  if (state.status === "loading" || state.status === "ready") return;
  client ??= new LlmClient();
  update({ status: "loading", progress: null, message: null });
  try {
    // Absolute, since the worker would resolve a relative URL against its own script.
    const url = new URL(MODEL_URL, document.baseURI).href;
    await client.load(url, (progress) => update({ progress }));
    update({ status: "ready" });
  } catch (error) {
    update({ status: "error", message: error instanceof Error ? error.message : String(error) });
  }
}

export function getClient(): LlmClient {
  if (!client || state.status !== "ready") throw new Error("model not loaded");
  return client;
}
