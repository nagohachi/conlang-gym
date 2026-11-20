/** Messages between the page (client.ts) and the model worker (worker.ts). */

export interface Manifest {
  model: string;
  graph: string;
  /** Changes whenever the files change; appended to file URLs to bust the cache. */
  version?: string;
  /** Published path -> size in bytes. */
  files: Record<string, number>;
}

export type WorkerRequest =
  | { type: "load"; id: number; modelUrl: string }
  | { type: "reset"; id: number }
  | { type: "encode"; id: number; text: string }
  | { type: "decode"; id: number; ids: number[] }
  | { type: "generate"; id: number; append: number[]; maxTokens: number }
  | { type: "abort"; id: number };

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
export type WorkerRequestBody = DistributiveOmit<WorkerRequest, "id">;

export type WorkerResponse =
  | { type: "progress"; id: number; phase: "download" | "init"; loaded: number; total: number }
  | { type: "text"; id: number; text: string; numTokens: number }
  | { type: "result"; id: number; value: unknown }
  | { type: "error"; id: number; message: string };
