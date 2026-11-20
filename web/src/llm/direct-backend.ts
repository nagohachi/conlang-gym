/** LlmBackend over an Engine and a tokenizer in the same thread (the worker, or Node tests). */
import type { Tokenizer } from "@huggingface/tokenizers";

import { STOP_IDS, type LlmBackend } from "./agent";
import type { Engine, StopReason } from "./engine";

const TEXT_UPDATE_INTERVAL_MS = 50;

export class DirectBackend implements LlmBackend {
  private abortRequested = false;

  constructor(
    private readonly engine: Engine,
    private readonly tokenizer: Tokenizer,
  ) {}

  async reset(): Promise<void> {
    this.engine.reset();
  }

  async encode(text: string): Promise<number[]> {
    return this.tokenizer.encode(text, { add_special_tokens: false }).ids;
  }

  async decode(ids: number[]): Promise<string> {
    return this.tokenizer.decode(ids, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  }

  /** Make the running generate() return "aborted" before its next token. */
  abort(): void {
    this.abortRequested = true;
  }

  async generate(
    append: number[],
    maxTokens: number,
    onText?: (text: string, numTokens: number) => void,
  ): Promise<{ ids: number[]; reason: StopReason }> {
    this.abortRequested = false;
    this.engine.append(append);
    const generated: number[] = [];
    let lastUpdate = 0;
    const result = await this.engine.generate(
      maxTokens,
      STOP_IDS,
      (id) => {
        generated.push(id);
        const now = performance.now();
        if (onText && now - lastUpdate >= TEXT_UPDATE_INTERVAL_MS) {
          lastUpdate = now;
          onText(this.tokenizer.decode(generated, { skip_special_tokens: false }), generated.length);
        }
      },
      () => this.abortRequested,
    );
    onText?.(this.tokenizer.decode(generated, { skip_special_tokens: false }), generated.length);
    return result;
  }
}
