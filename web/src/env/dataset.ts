/**
 * The seeds of the published model's dataset (README: make_dataset --num_active_params 0), as
 * web/scripts/dataset_seeds.py replays make_dataset.py's selection.
 */

import data from "./dataset_seeds.json";
import { type EnvConfig, generate, makeEnvConfig } from "./language";
import type { PySeed } from "./pyrandom";
import { solve } from "./solver";

export interface DatasetSeeds {
  /** The EnvConfig of every row of the dataset. */
  readonly envConfig: EnvConfig;
  /** The grammar labels (Grammar.label()) of the training word orders. */
  readonly seenGrammars: readonly string[];
  /** The grammar labels held out of training. */
  readonly unseenGrammars: readonly string[];
  /** The val_seen seeds of each seen grammar label, ascending (64 each). */
  readonly valSeen: Readonly<Record<string, readonly number[]>>;
  /** The val_unseen seeds of each unseen grammar label, ascending (64 each). */
  readonly valUnseen: Readonly<Record<string, readonly number[]>>;
  /** The number of train rows (8000). */
  readonly numTrain: number;
  /** The last seed that went into train.parquet. */
  readonly lastTrainSeed: number;
  /** Seeds 0 .. numCheckedSeeds - 1 were examined; later seeds were never looked at. */
  readonly numCheckedSeeds: number;
  /** How many examined seeds the solver rejected. */
  readonly numRejectedSeeds: number;
}

export const DATASET_SEEDS: DatasetSeeds = Object.freeze({
  ...data,
  envConfig: makeEnvConfig(data.envConfig),
});

/**
 * What the dataset did with a seed (under DATASET_SEEDS.envConfig):
 * - "val_seen" / "val_unseen": a validation seed;
 * - "train": a training seed;
 * - "rejected": examined, but the solver could not solve it within the budget;
 * - "unused": examined and solvable, but its grammar's quota was already full (an unseen grammar
 *   past its validation seeds, or a seen grammar after train was full);
 * - "unchecked": never examined (negative, or numCheckedSeeds or later), so never trained on.
 */
export type DatasetSplit = "val_seen" | "val_unseen" | "train" | "rejected" | "unused" | "unchecked";

const VAL_SPLIT = new Map<number, DatasetSplit>();
for (const seeds of Object.values(DATASET_SEEDS.valSeen)) {
  for (const seed of seeds) VAL_SPLIT.set(seed, "val_seen");
}
for (const seeds of Object.values(DATASET_SEEDS.valUnseen)) {
  for (const seed of seeds) VAL_SPLIT.set(seed, "val_unseen");
}
const UNSEEN = new Set(DATASET_SEEDS.unseenGrammars);

/** Return what the dataset did with seed; it runs the solver (some ms) for train candidates. */
export function datasetSplitOf(seed: PySeed): DatasetSplit {
  const value = typeof seed === "bigint" ? seed : Number.isInteger(seed) ? BigInt(seed) : -1n;
  if (value < 0n || value >= BigInt(DATASET_SEEDS.numCheckedSeeds)) return "unchecked";
  const number = Number(value);
  const val = VAL_SPLIT.get(number);
  if (val !== undefined) return val;
  const config = DATASET_SEEDS.envConfig;
  if (solve(number, config) === null) return "rejected";
  if (UNSEEN.has(generate(number, config).language.grammar.label())) return "unused";
  return number <= DATASET_SEEDS.lastTrainSeed ? "train" : "unused";
}
