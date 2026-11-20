/** The page's handle on the model worker: an LlmBackend whose calls run in worker.ts. */
import type { LlmBackend } from "./agent";
import type { StopReason } from "./engine";
import type { WorkerRequestBody, WorkerResponse } from "./protocol";

export interface LoadProgress {
  phase: "download" | "init";
  loaded: number;
  total: number;
}

type Listener = (message: WorkerResponse) => void;

export class LlmClient implements LlmBackend {
  private worker: Worker;
  private nextId = 1;
  private listeners = new Map<number, Listener>();

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) =>
      this.listeners.get(event.data.id)?.(event.data);
  }

  private call<T>(body: WorkerRequestBody, onMessage?: Listener): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.listeners.set(id, (message) => {
        if (message.type === "result") {
          this.listeners.delete(id);
          resolve(message.value as T);
        } else if (message.type === "error") {
          this.listeners.delete(id);
          reject(new Error(message.message));
        } else {
          onMessage?.(message);
        }
      });
      this.worker.postMessage({ ...body, id });
    });
  }

  load(modelUrl: string, onProgress?: (progress: LoadProgress) => void): Promise<void> {
    return this.call({ type: "load", modelUrl }, (message) => {
      if (message.type === "progress") onProgress?.(message);
    });
  }

  reset(): Promise<void> {
    return this.call({ type: "reset" });
  }

  encode(text: string): Promise<number[]> {
    return this.call({ type: "encode", text });
  }

  decode(ids: number[]): Promise<string> {
    return this.call({ type: "decode", ids });
  }

  generate(
    append: number[],
    maxTokens: number,
    onText?: (text: string, numTokens: number) => void,
  ): Promise<{ ids: number[]; reason: StopReason }> {
    return this.call({ type: "generate", append, maxTokens }, (message) => {
      if (message.type === "text") onText?.(message.text, message.numTokens);
    });
  }

  /** Stop the running generation after its current token. */
  abort(): void {
    void this.call({ type: "abort" });
  }
}
