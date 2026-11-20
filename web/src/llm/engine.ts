/**
 * Greedy decoding of the exported Qwen3.5-2B (embed_tokens + decoder_model_merged, q4f16).
 *
 * The decoder is a hybrid: 18 Gated DeltaNet layers carry a convolution state and a recurrent
 * state, 6 full-attention layers carry a KV cache. The engine keeps all of them between calls, so a
 * conversation is fed incrementally: append() queues tokens and generate() feeds the queue in one
 * prefill and then decodes token by token. The last sampled token is queued too, since the model has
 * not seen it yet. The ORT module is injected so the same code runs on onnxruntime-web (WebGPU) and
 * onnxruntime-node (CPU, for tests).
 */
import type * as OrtTypes from "onnxruntime-web";

export type Ort = typeof OrtTypes;
type Tensor = OrtTypes.Tensor;

export interface ModelBytes {
  embed: { model: Uint8Array; data: Uint8Array; dataPath: string };
  decoder: { model: Uint8Array; data: Uint8Array; dataPath: string };
}

export interface EngineOptions {
  /** "webgpu" in the browser, "cpu" in Node. */
  device: "webgpu" | "cpu";
}

export type StopReason = "eos" | "length" | "aborted";

export interface GenerateResult {
  ids: number[];
  reason: StopReason;
}

// Text config of Qwen3.5-2B: the cache shapes of the decoder's inputs.
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const CONV_CHANNELS = 6144; // key + key + value dims of the Gated DeltaNet
const CONV_STATE = 3; // kernel size 4 minus 1
const LINEAR_HEADS = 16;
const LINEAR_HEAD_DIM = 128;
const NUM_LAYERS = 24;
const FULL_ATTENTION_INTERVAL = 4; // every 4th layer is full attention, the rest Gated DeltaNet

function decoderOutputNames(): string[] {
  const names = ["logits"];
  for (let layer = 0; layer < NUM_LAYERS; layer++) {
    if (layer % FULL_ATTENTION_INTERVAL === FULL_ATTENTION_INTERVAL - 1) {
      names.push(`present.${layer}.key`, `present.${layer}.value`);
    } else {
      names.push(`present_conv.${layer}`, `present_recurrent.${layer}`);
    }
  }
  return names;
}

const FLOAT16_TABLE = buildFloat16Table();

function buildFloat16Table(): Float32Array {
  const table = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits++) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    if (exponent === 0) table[bits] = sign * 2 ** -14 * (fraction / 1024);
    else if (exponent === 31) table[bits] = fraction ? NaN : sign * Infinity;
    else table[bits] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return table;
}

/** Return the index of the largest of the last `size` logits (float16 or float32 data). */
export function argmaxLast(data: ArrayLike<number>, size: number): number {
  const offset = data.length - size;
  const isFloat16Bits = data instanceof Uint16Array;
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < size; i++) {
    const raw = data[offset + i];
    const value = isFloat16Bits ? FLOAT16_TABLE[raw] : raw;
    if (value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  return best;
}

export class Engine {
  private embed: OrtTypes.InferenceSession | null = null;
  private decoder: OrtTypes.InferenceSession | null = null;
  private cache = new Map<string, Tensor>();
  private pastLength = 0;
  private queue: number[] = [];
  private vocabSize = 0;

  constructor(
    private readonly ort: Ort,
    private readonly options: EngineOptions,
  ) {}

  /** Create the two inference sessions. */
  async init(bytes: ModelBytes): Promise<void> {
    const { ort, options } = this;
    const gpu = options.device === "webgpu";
    const common: OrtTypes.InferenceSession.SessionOptions = {
      executionProviders: [options.device],
      graphOptimizationLevel: "all",
      freeDimensionOverrides: { batch_size: 1 },
    };
    this.embed = await ort.InferenceSession.create(bytes.embed.model, {
      ...common,
      externalData: [{ path: bytes.embed.dataPath, data: bytes.embed.data }],
      ...(gpu ? { preferredOutputLocation: "gpu-buffer" } : {}),
    });
    const decoderOutputLocation: Record<string, OrtTypes.Tensor.DataLocation> = {};
    if (gpu) {
      // Caches stay on the GPU between steps; only the logits come back.
      for (const name of decoderOutputNames()) {
        decoderOutputLocation[name] = name === "logits" ? "cpu" : "gpu-buffer";
      }
    }
    this.decoder = await ort.InferenceSession.create(bytes.decoder.model, {
      ...common,
      externalData: [{ path: bytes.decoder.dataPath, data: bytes.decoder.data }],
      ...(gpu ? { preferredOutputLocation: decoderOutputLocation } : {}),
    });
    this.reset();
  }

  /** Forget the conversation: empty caches, nothing queued. */
  reset(): void {
    this.disposeCache();
    const { ort } = this;
    const decoder = this.requireDecoder();
    for (const name of decoder.inputNames) {
      let dims: number[] | null = null;
      if (name.startsWith("past_conv.")) dims = [1, CONV_CHANNELS, CONV_STATE];
      else if (name.startsWith("past_recurrent.")) dims = [1, LINEAR_HEADS, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM];
      else if (name.startsWith("past_key_values.")) dims = [1, NUM_KV_HEADS, 0, HEAD_DIM];
      if (dims) {
        const size = dims.reduce((a, b) => a * b, 1);
        this.cache.set(name, new ort.Tensor("float16", new Uint16Array(size), dims));
      }
    }
    this.pastLength = 0;
    this.queue = [];
  }

  /** Queue tokens to be fed before the next sampled token. */
  append(ids: readonly number[]): void {
    this.queue.push(...ids);
  }

  /** Number of tokens the model has consumed so far. */
  get length(): number {
    return this.pastLength;
  }

  /**
   * Feed the queue, then greedily decode up to maxTokens tokens or until a stop token.
   *
   * The returned ids include the stop token. The last returned token is left queued.
   */
  async generate(
    maxTokens: number,
    stopIds: ReadonlySet<number>,
    onToken?: (id: number) => void,
    shouldAbort?: () => boolean,
  ): Promise<GenerateResult> {
    const ids: number[] = [];
    while (ids.length < maxTokens) {
      if (shouldAbort?.()) return { ids, reason: "aborted" };
      const next = await this.step();
      ids.push(next);
      this.queue = [next];
      onToken?.(next);
      if (stopIds.has(next)) return { ids, reason: "eos" };
    }
    return { ids, reason: "length" };
  }

  /** Feed every queued token and return the greedy next token. */
  private async step(): Promise<number> {
    const { ort } = this;
    const embed = this.embed!;
    const decoder = this.requireDecoder();
    const ids = this.queue;
    if (ids.length === 0) throw new Error("nothing to feed");
    this.queue = [];
    const length = ids.length;
    const total = this.pastLength + length;

    const inputIds = new ort.Tensor("int64", BigInt64Array.from(ids, (id) => BigInt(id)), [1, length]);
    const { inputs_embeds: inputsEmbeds } = await embed.run({ input_ids: inputIds });

    const positions = new BigInt64Array(3 * length);
    for (let axis = 0; axis < 3; axis++) {
      for (let i = 0; i < length; i++) positions[axis * length + i] = BigInt(this.pastLength + i);
    }
    const feeds: Record<string, Tensor> = {
      inputs_embeds: inputsEmbeds,
      attention_mask: new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]),
      position_ids: new ort.Tensor("int64", positions, [3, 1, length]),
      num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), []),
    };
    for (const [name, tensor] of this.cache) feeds[name] = tensor;
    const outputs = await decoder.run(feeds);
    disposeIfGpu(inputsEmbeds);

    const nextCache = new Map<string, Tensor>();
    for (const name of this.cache.keys()) {
      const present = name.startsWith("past_key_values.")
        ? name.replace("past_key_values.", "present.")
        : name.replace("past_", "present_");
      nextCache.set(name, outputs[present]);
    }
    this.disposeCache();
    this.cache = nextCache;
    this.pastLength = total;

    const logits = outputs.logits;
    this.vocabSize = logits.dims[logits.dims.length - 1];
    const next = argmaxLast(logits.data as ArrayLike<number>, this.vocabSize);
    logits.dispose();
    return next;
  }

  private disposeCache(): void {
    for (const tensor of this.cache.values()) disposeIfGpu(tensor);
    this.cache.clear();
  }

  private requireDecoder(): OrtTypes.InferenceSession {
    if (!this.decoder) throw new Error("engine not initialized");
    return this.decoder;
  }

  async release(): Promise<void> {
    this.disposeCache();
    await this.embed?.release();
    await this.decoder?.release();
    this.embed = this.decoder = null;
  }
}

function disposeIfGpu(tensor: Tensor): void {
  if (tensor.location === "gpu-buffer") tensor.dispose();
}
