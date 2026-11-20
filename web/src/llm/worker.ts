/// <reference lib="webworker" />
/**
 * Web Worker that owns the model: downloads it (cached with the Cache API), runs it on WebGPU and
 * streams generated text back. See client.ts for the message protocol.
 */
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";

import { DirectBackend } from "./direct-backend";
import { Engine, type Ort } from "./engine";
import type { Manifest, WorkerRequest, WorkerResponse } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1; // the work runs on the GPU; threads would need cross-origin isolation
ort.env.logLevel = "error";

const CACHE_NAME = "conlang-gym-model";
const MODEL_FILES = [
  "onnx/embed_tokens_q4f16.onnx",
  "onnx/embed_tokens_q4f16.onnx_data",
  "onnx/decoder_model_merged_q4f16.onnx",
  "onnx/decoder_model_merged_q4f16.onnx_data",
  "tokenizer.json",
  "tokenizer_config.json",
];

let backend: DirectBackend | null = null;

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // e.g. an opaque origin; download every time
  }
}

/** Download url (or read it from the cache), reporting received bytes. */
async function fetchBytes(
  url: string,
  expectedSize: number,
  cache: Cache | null,
  onBytes: (received: number) => void,
): Promise<Uint8Array> {
  const cached = await cache?.match(url);
  const response = cached ?? (await fetch(url));
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const size = Number(response.headers.get("content-length")) || expectedSize;
  let buffer = new Uint8Array(size);
  let offset = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > buffer.length) {
      const grown = new Uint8Array(Math.max(buffer.length * 2, offset + value.length));
      grown.set(buffer);
      buffer = grown;
    }
    buffer.set(value, offset);
    offset += value.length;
    onBytes(value.length);
  }
  const bytes = offset === buffer.length ? buffer : buffer.slice(0, offset);
  if (!cached && cache) {
    try {
      const headers = { "content-length": String(bytes.length) };
      await cache.put(url, new Response(bytes as Uint8Array<ArrayBuffer>, { headers }));
    } catch (error) {
      console.warn(`could not cache ${url}`, error);
    }
  }
  return bytes;
}

async function load(id: number, modelUrl: string): Promise<void> {
  if (!("gpu" in navigator) || !(await navigator.gpu.requestAdapter())) {
    throw new Error("WebGPU is not available in this browser.");
  }
  const base = modelUrl.endsWith("/") ? modelUrl : modelUrl + "/";
  const manifest: Manifest = await (await fetch(base + "manifest.json", { cache: "no-cache" })).json();
  const cache = await openCache();
  const total = MODEL_FILES.reduce((sum, file) => sum + (manifest.files[file] ?? 0), 0);
  let received = 0;
  let lastReport = 0;
  const report = (bytes: number) => {
    received += bytes;
    const now = performance.now();
    if (now - lastReport > 100 || received >= total) {
      lastReport = now;
      post({ type: "progress", id, phase: "download", loaded: received, total });
    }
  };
  // The version query keeps a cached file from outliving a re-export of the model.
  const version = manifest.version ? `?v=${manifest.version}` : "";
  const files: Record<string, Uint8Array> = {};
  for (const file of MODEL_FILES) {
    files[file] = await fetchBytes(base + file + version, manifest.files[file] ?? 0, cache, report);
  }
  post({ type: "progress", id, phase: "init", loaded: total, total });
  const decoder = new TextDecoder();
  const tokenizer = new Tokenizer(
    JSON.parse(decoder.decode(files["tokenizer.json"])),
    JSON.parse(decoder.decode(files["tokenizer_config.json"])),
  );
  const engine = new Engine(ort as unknown as Ort, { device: "webgpu" });
  await engine.init({
    embed: {
      model: files["onnx/embed_tokens_q4f16.onnx"],
      data: files["onnx/embed_tokens_q4f16.onnx_data"],
      dataPath: "embed_tokens_q4f16.onnx_data",
    },
    decoder: {
      model: files["onnx/decoder_model_merged_q4f16.onnx"],
      data: files["onnx/decoder_model_merged_q4f16.onnx_data"],
      dataPath: "decoder_model_merged_q4f16.onnx_data",
    },
  });
  backend = new DirectBackend(engine, tokenizer);
}

function requireBackend(): DirectBackend {
  if (!backend) throw new Error("model not loaded");
  return backend;
}

async function handle(request: WorkerRequest): Promise<unknown> {
  switch (request.type) {
    case "load":
      return load(request.id, request.modelUrl);
    case "reset":
      return requireBackend().reset();
    case "encode":
      return requireBackend().encode(request.text);
    case "decode":
      return requireBackend().decode(request.ids);
    case "generate":
      return requireBackend().generate(request.append, request.maxTokens, (text, numTokens) =>
        post({ type: "text", id: request.id, text, numTokens }),
      );
    case "abort":
      backend?.abort();
      return undefined;
  }
}

async function respond(request: WorkerRequest): Promise<void> {
  try {
    const value = await handle(request);
    post({ type: "result", id: request.id, value });
  } catch (error) {
    post({ type: "error", id: request.id, message: error instanceof Error ? error.message : String(error) });
  }
}

// One conversation lives in the engine, so requests run one at a time, in order. An abort skips the
// queue, so that it reaches the generation it is meant to stop.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "abort") void respond(request);
  else queue = queue.then(() => respond(request));
};
