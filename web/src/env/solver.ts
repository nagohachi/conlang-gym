/**
 * A port of conlang_gym/solver.py: the greedy hypothesis solver that decides whether a seed can be
 * solved within the question budget (spec 8), and so whether make_dataset.py keeps it.
 *
 * A hypothesis is a grammar. Morpheme forms are not part of it: they are read off the observations
 * by position and kept as the hypothesis's bindings, and a hypothesis is dropped when no binding
 * fits every observation. Questions are chosen greedily by the entropy of the predicted
 * observation, which is computed exactly as CPython 3.12 does (insertion-ordered counts,
 * Neumaier-compensated sum()) so that ties break the same way.
 */

import {
  type EnvConfig,
  Grammar,
  type Meaning,
  type Morpheme,
  PARAM_CHOICES,
  PARAM_NAMES,
  type Template,
  buildSentenceTemplate,
  generate,
  meaningKey,
  meaningsEqual,
  morpheme,
} from "./language";
import { crLog } from "./pymath";
import { PyRandom, type PySeed } from "./pyrandom";

/** Compare two grammars by their field values in order, like Python's tuple comparison. */
function compareGrammars(a: Grammar, b: Grammar): number {
  const av = a.values();
  const bv = b.values();
  for (let i = 0; i < av.length; i++) {
    if (av[i]! < bv[i]!) return -1;
    if (av[i]! > bv[i]!) return 1;
  }
  return 0;
}

function buildAllGrammars(): Grammar[] {
  const unique = new Map<string, Grammar>();
  const product = (index: number, fields: Record<string, string>): void => {
    if (index === PARAM_NAMES.length) {
      const grammar = new Grammar(fields).toCanonical();
      unique.set(grammar.key(), grammar);
      return;
    }
    const name = PARAM_NAMES[index]!;
    for (const value of PARAM_CHOICES[name]) product(index + 1, { ...fields, [name]: value });
  };
  product(0, {});
  return [...unique.values()].sort(compareGrammars);
}

/** Every canonical grammar, sorted by its field values (498 grammars). */
export const ALL_GRAMMARS: readonly Grammar[] = Object.freeze(buildAllGrammars());
const GRAMMAR_INDEX = new Map(ALL_GRAMMARS.map((grammar, i) => [grammar.key(), i]));

// The templates of each meaning, per index in ALL_GRAMMARS (Python caches them with lru_cache).
// Templates are never mutated, so sharing them is safe; the cache is cleared when it grows large.
const MAX_CACHED_MEANINGS = 20_000;
const templateCache = new Map<string, (Template | undefined)[]>();

function cachedTemplates(meaning: Meaning): (Template | undefined)[] {
  const key = meaningKey(meaning);
  let templates = templateCache.get(key);
  if (templates === undefined) {
    if (templateCache.size >= MAX_CACHED_MEANINGS) templateCache.clear();
    templates = new Array<Template | undefined>(ALL_GRAMMARS.length);
    templateCache.set(key, templates);
  }
  return templates;
}

function templateOf(
  templates: (Template | undefined)[],
  grammarIndex: number,
  meaning: Meaning,
): Template {
  let template = templates[grammarIndex];
  if (template === undefined) {
    template = buildSentenceTemplate(ALL_GRAMMARS[grammarIndex]!, meaning);
    templates[grammarIndex] = template;
  }
  return template;
}

/**
 * CPython 3.12's sum() of a sequence of floats with the default start 0: the first item is added
 * to the int 0, then the rest with Neumaier's compensated summation.
 */
export function pySum(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let total = 0.0 + values[0]!;
  let compensation = 0.0;
  for (let i = 1; i < values.length; i++) {
    const x = values[i]!;
    const t = total + x;
    if (Math.abs(total) >= Math.abs(x)) {
      compensation += total - t + x;
    } else {
      compensation += x - t + total;
    }
    total = t;
  }
  // Avoid losing the sign on a negative result, and don't let adding the compensation convert an
  // infinite or overflowed sum to a NaN.
  if (compensation !== 0 && Number.isFinite(compensation)) total += compensation;
  return total;
}

/**
 * A grammar that fits every observation so far, with the morpheme forms they fix.
 *
 * Under vowel harmony an affix binding holds only the consonant; the vowel copies the root's.
 * Morphemes are interned (language.morpheme), so a Map keyed by them compares by value.
 */
export interface Hypothesis {
  readonly grammar: Grammar;
  readonly grammarIndex: number;
  readonly bindings: Map<Morpheme, string>;
}

/** The grammars that are consistent with the observations of one language. */
export class HypothesisSolver {
  hypotheses: Hypothesis[];

  /**
   * Start from every canonical grammar; knownRoots is null if the dictionary is hidden. log is the
   * natural logarithm the entropies use (correctly rounded, like the C library's log).
   */
  constructor(
    readonly knownRoots: Readonly<Record<string, string>> | null,
    private readonly log: (x: number) => number = crLog,
  ) {
    this.hypotheses = ALL_GRAMMARS.map((grammar, grammarIndex) => ({
      grammar,
      grammarIndex,
      bindings: new Map(),
    }));
  }

  /** Drop the hypotheses that cannot produce sentence for meaning and extend the others. */
  observe(meaning: Meaning, sentence: string): void {
    const words = sentence.split(" ");
    const knownRoots = this.knownRoots;

    const match = (template: Template, harmony: string, bindings: Map<Morpheme, string>) => {
      if (words.length !== template.length) return false;
      for (let w = 0; w < template.length; w++) {
        const wordTemplate = template[w]!;
        const word = words[w]!;
        let length = 0;
        for (const slot of wordTemplate) length += slot.isRoot ? 4 : 2;
        if (word.length !== length) return false;
        const pieces: string[] = [];
        let position = 0;
        for (const slot of wordTemplate) {
          const size = slot.isRoot ? 4 : 2;
          pieces.push(word.slice(position, position + size));
          position += size;
        }
        const rootPiece = pieces[wordTemplate.findIndex((slot) => slot.isRoot)]!;
        for (let m = 0; m < wordTemplate.length; m++) {
          const slot = wordTemplate[m]!;
          let piece = pieces[m]!;
          let expected: string | undefined;
          if (slot.isRoot && knownRoots !== null) {
            expected = knownRoots[slot.name];
          } else if (!slot.isRoot && harmony === "copy") {
            if (piece[1] !== rootPiece[rootPiece.length - 1]) return false;
            piece = piece[0]!;
            expected = bindings.get(slot);
          } else {
            expected = bindings.get(slot);
          }
          if (expected === undefined) {
            bindings.set(slot, piece);
          } else if (expected !== piece) {
            return false;
          }
        }
      }
      // Every morpheme has its own form (spec 3.2).
      for (const isRoot of [false, true]) {
        const forms: string[] = [];
        for (const [slot, form] of bindings) if (slot.isRoot === isRoot) forms.push(form);
        if (forms.length !== new Set(forms).size) return false;
      }
      return true;
    };

    const templates = cachedTemplates(meaning);
    const survivors: Hypothesis[] = [];
    for (const hypothesis of this.hypotheses) {
      const bindings = new Map(hypothesis.bindings);
      const template = templateOf(templates, hypothesis.grammarIndex, meaning);
      if (match(template, hypothesis.grammar.vowelHarmony, bindings)) {
        survivors.push({ ...hypothesis, bindings });
      }
    }
    this.hypotheses = survivors;
  }

  /**
   * Return each hypothesis's sentence for meaning.
   *
   * A form that no observation has fixed yet becomes <name> withPlaceholders, and makes the
   * prediction null otherwise.
   */
  predict(meaning: Meaning, withPlaceholders: boolean): (string | null)[] {
    const knownRoots = this.knownRoots;
    const render = (
      template: Template,
      harmony: string,
      bindings: Map<Morpheme, string>,
    ): string | null => {
      const words: string[] = [];
      for (const wordTemplate of template) {
        const rootName = wordTemplate.find((slot) => slot.isRoot)!.name;
        const root =
          knownRoots !== null
            ? knownRoots[rootName]!
            : (bindings.get(morpheme(true, rootName)) ?? null);
        let pieces = "";
        for (const slot of wordTemplate) {
          let form: string | null;
          if (slot.isRoot) {
            form = root;
          } else {
            form = bindings.get(slot) ?? null;
            if (form !== null && harmony === "copy") {
              form = root !== null ? form + root[root.length - 1]! : null;
            }
          }
          if (form === null) {
            if (!withPlaceholders) return null;
            form = `<${slot.name}>`;
          }
          pieces += form;
        }
        words.push(pieces);
      }
      return words.join(" ");
    };

    const templates = cachedTemplates(meaning);
    return this.hypotheses.map((hypothesis) =>
      render(
        templateOf(templates, hypothesis.grammarIndex, meaning),
        hypothesis.grammar.vowelHarmony,
        hypothesis.bindings,
      ),
    );
  }

  /** Return whether every hypothesis gives the same complete answer to every test item. */
  isSolved(tests: readonly Meaning[]): boolean {
    for (const meaning of tests) {
      const predictions = this.predict(meaning, false);
      if (predictions.length === 0) return false;
      const first = predictions[0];
      if (predictions.some((prediction) => prediction === null || prediction !== first)) {
        return false;
      }
    }
    return true;
  }

  /** Return the entropy of the predicted observation of meaning, exactly as Python computes it. */
  entropy(meaning: Meaning): number {
    const numHypotheses = this.hypotheses.length;
    // Map keeps first-occurrence order, like collections.Counter, so the sum runs in the same order.
    const counts = new Map<string | null, number>();
    for (const prediction of this.predict(meaning, true)) {
      counts.set(prediction, (counts.get(prediction) ?? 0) + 1);
    }
    const terms: number[] = [];
    for (const count of counts.values()) {
      terms.push((count / numHypotheses) * this.log(count / numHypotheses));
    }
    return -pySum(terms);
  }

  /**
   * Return the candidate whose predicted observation splits the hypotheses most evenly; the
   * first one wins ties. onEntropy, if given, receives every candidate's entropy in order.
   */
  chooseQuestion(candidates: readonly Meaning[], onEntropy?: (entropy: number) => void): Meaning {
    if (candidates.length === 0) throw new RangeError("chooseQuestion: no candidates");
    let best = candidates[0]!;
    let bestEntropy = -1.0;
    for (const meaning of candidates) {
      const entropy = this.entropy(meaning);
      onEntropy?.(entropy);
      if (entropy > bestEntropy) {
        best = meaning;
        bestEntropy = entropy;
      }
    }
    return best;
  }
}

/** One turn of solve(): the candidates' entropies in order and the question chosen. */
export interface SolverTurn {
  readonly entropies: number[];
  readonly question: Meaning;
  readonly numHypotheses: number;
}

export interface SolveOptions {
  /** Receives each turn's entropies and question. */
  readonly onTurn?: (turn: SolverTurn) => void;
  /** The natural logarithm the entropies use (default: correctly rounded). */
  readonly log?: (x: number) => number;
}

/**
 * Return the number of questions the greedy solver needs for seed, or null if over budget.
 *
 * Each turn the candidates are a random subset of the non-test meanings; asking a test item would
 * only make it unscorable.
 */
export function solve(
  seed: PySeed,
  config: EnvConfig,
  numCandidates: number = 48,
  options: SolveOptions = {},
): number | null {
  const onTurn = options.onTurn;
  const { language, tests } = generate(seed, config);
  const solver = new HypothesisSolver(
    config.showDictionary ? { ...language.roots } : null,
    options.log,
  );
  const rng = new PyRandom(BigInt(seed) * 7919n + 1n);
  const testSet = new Set(tests.map(meaningKey));
  const pool = language.listMeanings().filter((meaning) => !testSet.has(meaningKey(meaning)));
  for (let numQuestions = 1; numQuestions <= config.budget; numQuestions++) {
    const candidates = rng.sample(pool, Math.min(numCandidates, pool.length));
    const entropies: number[] = [];
    const numHypotheses = solver.hypotheses.length;
    const question = solver.chooseQuestion(
      candidates,
      onTurn && ((entropy) => entropies.push(entropy)),
    );
    onTurn?.({ entropies, question, numHypotheses });
    pool.splice(
      pool.findIndex((meaning) => meaningsEqual(meaning, question)),
      1,
    );
    solver.observe(question, language.realize(question));
    if (solver.isSolved(tests)) return numQuestions;
  }
  return null;
}

/** Return the index of grammar in ALL_GRAMMARS (after canonicalizing it). */
export function grammarIndex(grammar: Grammar): number {
  return GRAMMAR_INDEX.get(grammar.toCanonical().key())!;
}
