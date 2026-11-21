/**
 * Parity tests of the TypeScript environment (web/src/env) against the Python one (conlang_gym/).
 *
 * The fixtures are written by web/scripts/dump_env_fixtures.py; every comparison is exact (strings
 * byte for byte, floats bit for bit).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ALL_GRAMMARS,
  DATASET_SEEDS,
  Episode,
  type EpisodeScore,
  Grammar,
  Language,
  type Meaning,
  PyRandom,
  type PySeed,
  crLog,
  datasetSplitOf,
  envConfigFromPython,
  generate,
  isCompositional,
  makeEnvConfig,
  makeMeaning,
  meaningToText,
  pySum,
  solve,
} from "../src/env";

// ------------------------------------------------------------------------------------------------
// Fixture types and helpers

type PyMeaning = [string, string, string, string, string | null, string | null];
type PyConfig = Record<string, number | boolean | string[]>;
type PyScore = Record<string, number>;

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8")) as T;
}

function toSeed(seed: number | string): PySeed {
  return typeof seed === "string" ? BigInt(seed) : seed;
}

function toMeaning(m: PyMeaning): Meaning {
  return makeMeaning(m[0], m[1], m[2], m[3], m[4], m[5]);
}

function fromMeaning(m: Meaning): PyMeaning {
  return [m.subject, m.subjectNumber, m.verb, m.tense, m.object, m.objectNumber];
}

function scoreToPython(score: EpisodeScore): PyScore {
  return {
    reward: score.reward,
    coverage: score.coverage,
    num_asks: score.numAsks,
    num_judges: score.numJudges,
    num_invalid: score.numInvalid,
    num_scorable: score.numScorable,
    num_correct: score.numCorrect,
    test_truncated: score.testTruncated,
  };
}

/** The bits of a double as hex, so that floats compare bit for bit (including -0). */
function bits(x: number): string {
  const buffer = Buffer.alloc(8);
  buffer.writeDoubleBE(x);
  return buffer.toString("hex");
}

function errorOf(action: () => unknown): string {
  try {
    action();
  } catch (error) {
    return (error as Error).message;
  }
  return "no error";
}

// ------------------------------------------------------------------------------------------------
// PyRandom

type PyRandomFixture = { seed: number | string; ops: (string | number | number[])[][] }[];

describe("PyRandom", () => {
  const fixture = load<PyRandomFixture>("pyrandom.json");
  it.each(fixture.map((entry) => [String(entry.seed), entry] as const))(
    "reproduces random.Random(%s)",
    (_, entry) => {
      const rng = new PyRandom(toSeed(entry.seed));
      const actual = entry.ops.map((op) => {
        const range = (n: number) => Array.from({ length: n }, (_, i) => i);
        switch (op[0]) {
          case "getrandbits":
            return [op[0], op[1], rng.getrandbits(op[1] as number)];
          case "getrandbitsBig":
            return [op[0], op[1], String(rng.getrandbitsBig(op[1] as number))];
          case "random":
            return [op[0], rng.random()];
          case "randbelow":
            return [op[0], op[1], rng.randbelow(op[1] as number)];
          case "choice":
            return [op[0], op[1], rng.choice(range(op[1] as number))];
          case "sample":
            return [op[0], op[1], op[2], rng.sample(range(op[1] as number), op[2] as number)];
          case "shuffle": {
            const items = range(op[1] as number);
            rng.shuffle(items);
            return [op[0], op[1], items];
          }
          default:
            throw new Error(`unknown op ${op[0]}`);
        }
      });
      expect(actual).toEqual(entry.ops);
    },
  );

  it("refuses what CPython refuses", () => {
    const rng = new PyRandom(0);
    expect(errorOf(() => rng.sample([1, 2, 3], 4))).toBe(
      "Sample larger than population or is negative",
    );
    expect(errorOf(() => rng.choice([]))).toBe("Cannot choose from an empty sequence");
  });
});

// ------------------------------------------------------------------------------------------------
// Languages

interface LanguageCase {
  seed: number | string;
  grammar: Record<string, string>;
  label: string;
  nouns: string[];
  transitive: string[];
  intransitive: string[];
  roots: [string, string][];
  affixes: [string, string][];
  tests: [PyMeaning, string, string, boolean][];
  num_meanings: number;
  meanings_sha256: string;
  prompt: string;
}

interface GenerateFixture {
  configs: { name: string; config: PyConfig; cases: LanguageCase[] }[];
  config_errors: { config: PyConfig; error: string }[];
  generate_errors: { config: PyConfig; error: string }[];
}

describe("generate and the initial prompt", () => {
  const fixture = load<GenerateFixture>("generate.json");

  it.each(fixture.configs.map((entry) => [entry.name, entry] as const))(
    "match Python under config %s",
    (_, entry) => {
      const config = envConfigFromPython(entry.config);
      const actual = entry.cases.map((expected) => {
        const episode = new Episode(toSeed(expected.seed), config);
        const lang = episode.language;
        const meanings = lang.listMeanings();
        const lines = meanings.map(
          (m) =>
            `${meaningToText(m)}\t${lang.realize(m)}\t${Number(isCompositional(m))}`,
        );
        return {
          seed: expected.seed,
          grammar: lang.grammar.toPython(),
          label: lang.grammar.label(),
          nouns: [...lang.nouns],
          transitive: [...lang.transitive],
          intransitive: [...lang.intransitive],
          roots: Object.entries(lang.roots),
          affixes: Object.entries(lang.affixes),
          tests: episode.tests.map((m) => [
            fromMeaning(m),
            lang.realize(m),
            meaningToText(m),
            isCompositional(m),
          ]),
          num_meanings: meanings.length,
          meanings_sha256: createHash("sha256").update(lines.join("\n"), "utf8").digest("hex"),
          prompt: episode.buildInitialPrompt(),
        };
      });
      expect(actual).toEqual(entry.cases);
    },
  );

  it("validates EnvConfig with Python's messages", () => {
    for (const { config, error } of fixture.config_errors) {
      expect(errorOf(() => envConfigFromPython(config))).toBe(error);
    }
  });

  it("raises Python's error when the survey sheet is too small for the test set", () => {
    for (const { config, error } of fixture.generate_errors) {
      expect(errorOf(() => generate(0, envConfigFromPython(config)))).toBe(error);
    }
  });
});

interface GrammarsFixture {
  roots: [string, string][];
  affixes: [string, string][];
  nouns: string[];
  transitive: string[];
  intransitive: string[];
  meanings: PyMeaning[];
  grammars: {
    grammar: Record<string, string>;
    label: string;
    has_affix: boolean;
    realizations: string[];
  }[];
}

describe("ALL_GRAMMARS and realize", () => {
  const fixture = load<GrammarsFixture>("grammars.json");

  it("lists the same canonical grammars in the same order, realizing meanings the same way", () => {
    const meanings = fixture.meanings.map(toMeaning);
    const actual = ALL_GRAMMARS.map((grammar) => {
      const lang = new Language(
        grammar,
        fixture.nouns,
        fixture.transitive,
        fixture.intransitive,
        Object.fromEntries(fixture.roots),
        Object.fromEntries(fixture.affixes),
      );
      return {
        grammar: grammar.toPython(),
        label: grammar.label(),
        has_affix: grammar.hasAffix(),
        realizations: meanings.map((m) => lang.realize(m)),
      };
    });
    expect(actual).toEqual(fixture.grammars);
  });

  it("round-trips grammars through Python's field names", () => {
    for (const grammar of ALL_GRAMMARS) {
      expect(Grammar.fromPython(grammar.toPython()).equals(grammar)).toBe(true);
    }
  });
});

// ------------------------------------------------------------------------------------------------
// Episodes

interface EpisodeFixture {
  seed: number | string;
  config: PyConfig;
  steps: {
    name: string;
    text: string;
    truncated: boolean;
    reply: string;
    action: [string, ...unknown[]] | null;
    state: [number, number, number, number, boolean, number];
  }[];
  scores: {
    text: string;
    truncated: boolean;
    score: PyScore;
    items: [string | null, boolean, boolean][];
  }[];
}

describe("Episode", () => {
  const fixture = load<EpisodeFixture[]>("episodes.json");

  it.each(fixture.map((entry, i) => [i, String(entry.seed), entry] as const))(
    "episode %i (seed %s) replies, parses, counts and scores like Python",
    (_, __, entry) => {
      const episode = new Episode(toSeed(entry.seed), envConfigFromPython(entry.config));
      const steps = entry.steps.map((expected) => {
        const reply = episode.step(expected.text, expected.truncated);
        const action = episode.lastAction!;
        let parsed: [string, ...unknown[]];
        switch (action.kind) {
          case "ask":
            parsed = ["ask", fromMeaning(action.meaning)];
            break;
          case "judge":
            parsed = ["judge", action.sentence];
            break;
          case "invalid":
            parsed = ["invalid", action.error];
            break;
          default:
            parsed = ["done"];
        }
        return {
          ...expected,
          reply,
          action: expected.action === null ? null : parsed,
          state: [
            episode.budgetLeft,
            episode.numAsks,
            episode.numJudges,
            episode.numInvalid,
            episode.inTestPhase,
            episode.asked.size,
          ],
        };
      });
      expect(steps).toEqual(entry.steps);

      const scores = entry.scores.map((expected) => {
        const grade = episode.grade(expected.text, expected.truncated);
        expect(episode.score(expected.text, expected.truncated)).toEqual(grade.score);
        grade.items.forEach((item, i) => {
          expect(item.id).toBe(i + 1);
          expect(item.meaning).toBe(episode.tests[i]);
          expect(item.expected).toBe(episode.language.realize(episode.tests[i]!));
        });
        return {
          ...expected,
          score: scoreToPython(grade.score),
          items: grade.items.map((item) => [item.answer, item.scorable, item.correct]),
        };
      });
      expect(scores).toEqual(entry.scores);
      for (const record of entry.scores) {
        const score = episode.score(record.text, record.truncated);
        expect(bits(score.reward)).toBe(bits(record.score["reward"]!));
      }
    },
  );

  it("exposes the test prompt and the asked meanings", () => {
    const episode = new Episode(3, makeEnvConfig());
    const test = episode.tests[0]!;
    const reply = episode.step(`<action>ask(${meaningToText(test)})</action>`);
    expect(reply).toBe(`Informant: ${episode.language.realize(test)}\n(5 questions left)`);
    expect(episode.isAsked(test)).toBe(true);
    expect(episode.step("<action>done</action>")).toBe(episode.buildTestPrompt());
    expect(episode.grade("").items[0]!.scorable).toBe(false);
  });
});

// ------------------------------------------------------------------------------------------------
// The floating-point operations of the solver

interface PyMathFixture {
  max_hypotheses: number;
  num_log_inputs: number;
  log_sha256: string;
  /** [count, numHypotheses, Python's math.log(count / numHypotheses), correctly rounded log]. */
  log_exceptions: [number, number, number, number][];
  sums: [number[], number][];
}

const pymath = load<PyMathFixture>("pymath.json");
const platformExceptions = new Map(pymath.log_exceptions.map(([c, n, value]) => [c / n, value]));
/** math.log of the Python that wrote the fixtures: correctly rounded except at the exceptions. */
const fixtureLog = (x: number): number => platformExceptions.get(x) ?? crLog(x);

describe("log and sum", () => {
  it("crLog is the fixture platform's math.log on every count / numHypotheses but the listed inputs", () => {
    const digest = createHash("sha256");
    const buffer = Buffer.alloc(8);
    let numInputs = 0;
    for (let n = 1; n <= pymath.max_hypotheses; n++) {
      for (let c = 1; c <= n; c++) {
        buffer.writeDoubleLE(fixtureLog(c / n));
        digest.update(buffer);
        numInputs++;
      }
    }
    expect(numInputs).toBe(pymath.num_log_inputs);
    expect(digest.digest("hex")).toBe(pymath.log_sha256);
    // At the listed inputs the platform's log is off by one ulp and crLog is correctly rounded.
    for (const [c, n, platform, correct] of pymath.log_exceptions) {
      expect(bits(crLog(c / n))).toBe(bits(correct));
      expect(bits(crLog(c / n))).not.toBe(bits(platform));
    }
  });

  it("pySum is CPython 3.12's sum()", () => {
    for (const [values, expected] of pymath.sums) {
      expect(bits(pySum(values))).toBe(bits(expected));
    }
  });
});

// ------------------------------------------------------------------------------------------------
// Solver

interface SolverFixture {
  name: string;
  config: PyConfig;
  num_candidates: number;
  results: (number | null)[];
  traces: {
    seed: number;
    result: number | null;
    turns: { num_hypotheses: number; entropies: number[]; question: PyMeaning }[];
  }[];
}

describe("solve", () => {
  const fixture = load<SolverFixture[]>("solver.json");

  it.each(fixture.map((entry) => [entry.name, entry.results.length, entry] as const))(
    "gives Python's result under %s for %i seeds",
    (_, __, entry) => {
      const config = envConfigFromPython(entry.config);
      const actual = entry.results.map((_, seed) => solve(seed, config, entry.num_candidates));
      expect(actual).toEqual(entry.results);
    },
    120_000,
  );

  it.each(fixture.map((entry) => [entry.name, entry] as const))(
    "computes Python's entropies bit for bit under %s",
    (_, entry) => {
      const config = envConfigFromPython(entry.config);
      for (const trace of entry.traces) {
        for (const log of [fixtureLog, crLog]) {
          const turns: { num_hypotheses: number; entropies: string[]; question: PyMeaning }[] =
            [];
          const result = solve(trace.seed, config, entry.num_candidates, {
            log,
            onTurn: (turn) =>
              turns.push({
                num_hypotheses: turn.numHypotheses,
                entropies: turn.entropies.map(bits),
                question: fromMeaning(turn.question),
              }),
          });
          expect(result).toBe(trace.result);
          const expected = trace.turns.map((turn) => ({ ...turn, entropies: turn.entropies.map(bits) }));
          if (log === fixtureLog) {
            expect(turns).toEqual(expected);
          } else {
            // With the correctly rounded log the questions are the same; the entropies can differ
            // in the last bit only where the fixture platform's log is not correctly rounded.
            expect(turns.map((turn) => turn.question)).toEqual(expected.map((turn) => turn.question));
          }
        }
      }
    },
    60_000,
  );
});

// ------------------------------------------------------------------------------------------------
// Dataset seeds

describe("DATASET_SEEDS", () => {
  it("matches make_dataset's selection on the seeds up to the last validation seed", async () => {
    const config = DATASET_SEEDS.envConfig;
    const valSeen: Record<string, number[]> = {};
    const valUnseen: Record<string, number[]> = {};
    const unseen = new Set(DATASET_SEEDS.unseenGrammars);
    const allVal = [...Object.values(DATASET_SEEDS.valSeen), ...Object.values(DATASET_SEEDS.valUnseen)];
    const lastVal = Math.max(...allVal.flat());
    let numTrain = 0;
    for (let seed = 0; seed <= lastVal; seed++) {
      // This loop runs for tens of seconds; yielding lets the vitest worker answer its RPC calls.
      if (seed % 100 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
      let split: string;
      if (solve(seed, config) === null) {
        split = "rejected";
      } else {
        const label = generate(seed, config).language.grammar.label();
        const target = unseen.has(label) ? valUnseen : valSeen;
        const quota = 64;
        if ((target[label] ??= []).length < quota) {
          target[label].push(seed);
          split = unseen.has(label) ? "val_unseen" : "val_seen";
        } else if (unseen.has(label) || numTrain >= DATASET_SEEDS.numTrain) {
          split = "unused";
        } else {
          numTrain++;
          split = "train";
        }
      }
      expect(datasetSplitOf(seed)).toBe(split);
    }
    expect(valSeen).toEqual(DATASET_SEEDS.valSeen);
    expect(valUnseen).toEqual(DATASET_SEEDS.valUnseen);
    expect(Object.keys(valSeen).sort()).toEqual(DATASET_SEEDS.seenGrammars);
    expect(Object.keys(valUnseen).sort()).toEqual(DATASET_SEEDS.unseenGrammars);
    expect(datasetSplitOf(DATASET_SEEDS.numCheckedSeeds)).toBe("unchecked");
    expect(datasetSplitOf(-1)).toBe("unchecked");
  }, 120_000);
});
