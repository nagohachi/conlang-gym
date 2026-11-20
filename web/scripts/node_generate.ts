/**
 * Greedy generation with the exported model on CPU (onnxruntime-node), for checking the web runtime
 * against Python. Prints the prompt's token ids and the generated ids as JSON.
 *
 *   npx vite-node scripts/node_generate.ts -- <model dir> <prompt file> <max new tokens>
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Tokenizer } from "@huggingface/tokenizers";

import { userTurn } from "../src/llm/agent";
import { DirectBackend } from "../src/llm/direct-backend";
import { Engine, type Ort } from "../src/llm/engine";

// onnxruntime-node 1.31 rejects empty Float16Array inputs (the empty KV cache); without
// Float16Array, ORT keeps float16 data as Uint16Array.
delete (globalThis as { Float16Array?: unknown }).Float16Array;
const ortNode = await import("onnxruntime-node");

const [modelDir, promptFile, maxNew = "32"] = process.argv.slice(2).filter((a) => a !== "--");
const read = (name: string) => new Uint8Array(readFileSync(join(modelDir, name)));
const tokenizer = new Tokenizer(
  JSON.parse(readFileSync(join(modelDir, "tokenizer.json"), "utf8")),
  JSON.parse(readFileSync(join(modelDir, "tokenizer_config.json"), "utf8")),
);
const engine = new Engine(ortNode as unknown as Ort, { device: "cpu" });
await engine.init({
  embed: {
    model: read("onnx/embed_tokens_q4f16.onnx"),
    data: read("onnx/embed_tokens_q4f16.onnx_data"),
    dataPath: "embed_tokens_q4f16.onnx_data",
  },
  decoder: {
    model: read("onnx/decoder_model_merged_q4f16.onnx"),
    data: read("onnx/decoder_model_merged_q4f16.onnx_data"),
    dataPath: "decoder_model_merged_q4f16.onnx_data",
  },
});
const backend = new DirectBackend(engine, tokenizer);
const promptIds = await backend.encode(userTurn(readFileSync(promptFile, "utf8")));
const started = performance.now();
const { ids, reason } = await backend.generate(promptIds, Number(maxNew));
const seconds = (performance.now() - started) / 1000;
console.log(JSON.stringify({ promptIds, ids, reason, text: await backend.decode(ids), seconds }));
