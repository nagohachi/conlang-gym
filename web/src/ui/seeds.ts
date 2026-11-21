/** Choosing seeds, and what a seed meant for the model's training data. */
import { DATASET_SEEDS, datasetSplitOf, type EnvConfig, makeEnvConfig, PY_CONFIG_NAMES, solve } from "../env";

export type Mode = "human" | "model";

/** The published model's data: make_dataset.py with every flag at its default (word order only). */
export const TRAINING_CONFIG: EnvConfig = makeEnvConfig({ activeParams: [] });

export function sameConfig(a: EnvConfig, b: EnvConfig): boolean {
  // Parameter lists are compared as sets: their order does not change the languages.
  const normalize = (value: EnvConfig[keyof EnvConfig]) =>
    Array.isArray(value) ? JSON.stringify([...value].sort()) : value;
  return (Object.keys(PY_CONFIG_NAMES) as (keyof EnvConfig)[]).every(
    (key) => normalize(a[key]) === normalize(b[key]),
  );
}

/**
 * A random seed that make_dataset.py never examined (so never trained on) and that the reference
 * solver can solve within the budget, as the dataset requires.
 */
export function pickSeed(config: EnvConfig): number {
  const first = DATASET_SEEDS.numCheckedSeeds;
  let seed = first;
  for (let attempt = 0; attempt < 200; attempt++) {
    seed = first + Math.floor(Math.random() * 1_000_000);
    if (solve(seed, config) !== null) break;
  }
  return seed;
}

export interface SeedInfo {
  /** Questions the greedy solver needs, or null if it can't solve the seed within the budget. */
  solverQuestions: number | null;
  /** How the seed relates to the training data, without giving the grammar away. */
  training: string;
  /** The same, possibly naming the word order (for after the test). */
  trainingAfter: string;
}

export function describeSeed(seed: number, config: EnvConfig): SeedInfo {
  const solverQuestions = solve(seed, config);
  if (!sameConfig(config, TRAINING_CONFIG)) return { solverQuestions, training: "", trainingAfter: "" };
  const same = (text: string) => ({ solverQuestions, training: text, trainingAfter: text });
  switch (datasetSplitOf(seed)) {
    case "train":
      return same("This language was in the model's training data.");
    case "val_seen":
      return {
        solverQuestions,
        training: "This language is in the held-out validation set.",
        trainingAfter: "A held-out validation language; its word order also occurs in training.",
      };
    case "val_unseen":
      return {
        solverQuestions,
        training: "This language is in the held-out validation set.",
        trainingAfter: "A held-out validation language whose word order never occurs in training.",
      };
    case "rejected":
      return same("The dataset skipped this language, so the model never saw it.");
    default:
      return same("The model never saw this language in training.");
  }
}
