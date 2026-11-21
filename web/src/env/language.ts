/**
 * A line-by-line port of conlang_gym/language.py: language generation, the informant, action
 * parsing and scoring. For the same seed and config every language, test set, prompt and reply is
 * identical to the Python environment (web/tests/env.test.ts checks this against Python fixtures).
 *
 * Python names map to camelCase here (num_nouns -> numNouns, subject_number -> subjectNumber);
 * parameter values ("plural_first", "suffix", ...) and every user-visible string are unchanged.
 */

import { PyRandom, type PySeed } from "./pyrandom";

export const CONSONANTS = "ptkbdgmnslrvzfh";
export const VOWELS = "aeiou";
export const NOUN_POOL = [
  "dog",
  "cat",
  "bird",
  "fish",
  "horse",
  "frog",
  "child",
  "woman",
  "man",
  "bear",
] as const;
export const TRANSITIVE_POOL = ["see", "chase", "bite", "call", "help", "hear"] as const;
export const INTRANSITIVE_POOL = ["sleep", "run", "sing", "swim"] as const;
export const NUMBERS = ["singular", "plural"] as const;
export const TENSES = ["present", "past"] as const;

// The key order matches Grammar's field order: the solver builds grammars from the product of these.
export const PARAM_CHOICES = {
  wordOrder: ["SOV", "SVO", "VSO", "VOS", "OSV", "OVS"],
  objectMarking: ["none", "suffix"],
  plural: ["none", "suffix", "prefix"],
  pluralCaseOrder: ["plural_first", "case_first"],
  past: ["none", "suffix", "prefix"],
  subjectAgreement: ["none", "suffix"],
  vowelHarmony: ["none", "copy"],
  ergativeMarking: ["none", "suffix"],
  objectPlural: ["none", "distinct"],
  verbNumberStem: ["none", "subject", "absolutive"],
} as const satisfies Record<string, readonly string[]>;

export type ParamName = keyof typeof PARAM_CHOICES;
/** The parameters in Python's PARAM_CHOICES order (Grammar's field order). */
export const PARAM_NAMES: readonly ParamName[] = [
  "wordOrder",
  "objectMarking",
  "plural",
  "pluralCaseOrder",
  "past",
  "subjectAgreement",
  "vowelHarmony",
  "ergativeMarking",
  "objectPlural",
  "verbNumberStem",
];
/** The Python name of each parameter, as grammar labels and Python JSON use it. */
export const PY_PARAM_NAMES: Readonly<Record<ParamName, string>> = {
  wordOrder: "word_order",
  objectMarking: "object_marking",
  plural: "plural",
  pluralCaseOrder: "plural_case_order",
  past: "past",
  subjectAgreement: "subject_agreement",
  vowelHarmony: "vowel_harmony",
  ergativeMarking: "ergative_marking",
  objectPlural: "object_plural",
  verbNumberStem: "verb_number_stem",
};
// Parameters that can be switched on in addition to word order, which is always active.
export const OPTIONAL_PARAMS: readonly ParamName[] = [
  "objectMarking",
  "plural",
  "past",
  "subjectAgreement",
  "vowelHarmony",
  "ergativeMarking",
  "objectPlural",
  "verbNumberStem",
];
// Parameters that put an affix on a word; vowel harmony is only visible with one of them.
export const AFFIX_PARAMS: readonly ParamName[] = [
  "objectMarking",
  "plural",
  "past",
  "subjectAgreement",
  "ergativeMarking",
];
// Affixes every language has a form for, whether or not its grammar uses them.
export const AFFIX_NAMES = ["plural", "object", "past", "agreement"] as const;
// Affixes a language has only when its grammar uses them, with the parameter that does.
export const EXTRA_AFFIX_PARAMS: readonly (readonly [string, ParamName])[] = [
  ["ergative", "ergativeMarking"],
  ["object_plural", "objectPlural"],
];

/** Return the key of a verb's second stem in Language.roots, used under verbNumberStem. */
export function pluralStemKey(verb: string): string {
  return `${verb}.plural`;
}

export const TRANSITIVE_SLOTS = [
  "Subject",
  "SubjectNumber",
  "Object",
  "ObjectNumber",
  "Verb",
  "Tense",
] as const;
export const INTRANSITIVE_SLOTS = ["Subject", "SubjectNumber", "Verb", "Tense"] as const;

export const ERROR_FORMAT = "Error: format error.";
export const ERROR_RANGE = "Error: out of range.";
export const ERROR_TYPE = "Error: type error.";

// ---------------------------------------------------------------------------------------------
// Python str and re semantics that differ from JavaScript's.

/** The characters for which Python 3.12's str.isspace() is true (also what re's \s matches). */
const PY_WHITESPACE =
  "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007" +
  "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
const PY_WS_CLASS = "[\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";

/**
 * The code point of the digit zero of every run of decimal digits (category Nd) in Python 3.12's
 * Unicode 15.0 database; each run holds the ten digits 0-9 in order. Python's re \d and int()
 * accept all of them.
 */
const PY_DIGIT_ZEROS = [
  0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6,
  0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50,
  0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0,
  0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0,
  0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50,
  0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0,
];
const PY_DIGIT_CLASS =
  "[" +
  PY_DIGIT_ZEROS.map(
    (zero) => `\\u{${zero.toString(16)}}-\\u{${(zero + 9).toString(16)}}`,
  ).join("") +
  "]";

/** Return the value of a Python decimal digit character. */
function pyDigitValue(codePoint: number): number {
  for (const zero of PY_DIGIT_ZEROS) {
    if (codePoint >= zero && codePoint <= zero + 9) return codePoint - zero;
  }
  throw new RangeError(`not a decimal digit: U+${codePoint.toString(16)}`);
}

/**
 * Return int(digits) of a string of Python decimal digits, or null if it exceeds
 * Number.MAX_SAFE_INTEGER (such an id cannot name a test item, so nothing is lost by dropping it;
 * Python's int() would even raise ValueError beyond 4300 digits).
 */
function pyIntValue(digits: string): number | null {
  let ascii = "";
  for (const char of digits) ascii += String(pyDigitValue(char.codePointAt(0)!));
  ascii = ascii.replace(/^0+(?=\d)/, "");
  if (ascii.length > 16) return null;
  const value = Number(ascii);
  return Number.isSafeInteger(value) ? value : null;
}

/** Python's str.strip() without arguments. */
export function pyStrip(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && PY_WHITESPACE.includes(text[start]!)) start++;
  while (end > start && PY_WHITESPACE.includes(text[end - 1]!)) end--;
  return text.slice(start, end);
}

// The body of an action is stripped first, so Python's $ (which also matches before a final
// newline) and JavaScript's $ match at the same places.
const ACTION_RE = /<action>([\s\S]*?)<\/action>/g;
const ASK_RE = /^ask\(([\s\S]*)\)$/;
const JUDGE_RE = new RegExp(`^judge\\(${PY_WS_CLASS}*"([\\s\\S]*)"${PY_WS_CLASS}*\\)$`);
const ANSWER_RE = new RegExp(`<answer id="(${PY_DIGIT_CLASS}+)">([\\s\\S]*?)</answer>`, "gu");
const WHITESPACE_RUN_RE = new RegExp(`${PY_WS_CLASS}+`, "g");

// ---------------------------------------------------------------------------------------------
// Grammar and configuration.

export interface GrammarFields {
  readonly wordOrder: string;
  readonly objectMarking: string;
  readonly plural: string;
  readonly pluralCaseOrder: string;
  readonly past: string;
  readonly subjectAgreement: string;
  readonly vowelHarmony: string;
  readonly ergativeMarking: string;
  readonly objectPlural: string;
  readonly verbNumberStem: string;
}

/**
 * One value per grammatical parameter (spec 3.3); "none" switches a parameter off.
 *
 * A is the subject of a transitive verb, S the subject of an intransitive verb and O the object.
 * objectMarking puts a case suffix on O and ergativeMarking one on A (with both, the marking is
 * tripartite). objectPlural gives O its own plural affix, in the position plural gives. Under
 * verbNumberStem the verb has a second stem, used when the subject (S or A) is plural, or when
 * the absolutive argument (S or O) is plural.
 */
export class Grammar implements GrammarFields {
  readonly wordOrder: string;
  readonly objectMarking: string;
  readonly plural: string;
  readonly pluralCaseOrder: string;
  readonly past: string;
  readonly subjectAgreement: string;
  readonly vowelHarmony: string;
  readonly ergativeMarking: string;
  readonly objectPlural: string;
  readonly verbNumberStem: string;

  constructor(fields: Partial<GrammarFields> = {}) {
    this.wordOrder = fields.wordOrder ?? "SVO";
    this.objectMarking = fields.objectMarking ?? "none";
    this.plural = fields.plural ?? "none";
    this.pluralCaseOrder = fields.pluralCaseOrder ?? "plural_first";
    this.past = fields.past ?? "none";
    this.subjectAgreement = fields.subjectAgreement ?? "none";
    this.vowelHarmony = fields.vowelHarmony ?? "none";
    this.ergativeMarking = fields.ergativeMarking ?? "none";
    this.objectPlural = fields.objectPlural ?? "none";
    this.verbNumberStem = fields.verbNumberStem ?? "none";
    Object.freeze(this);
  }

  /** Return whether any parameter puts an affix on a word. */
  hasAffix(): boolean {
    return AFFIX_PARAMS.some((name) => this[name] !== "none");
  }

  /** Return whether a case suffix (accusative or ergative) exists. */
  hasCaseSuffix(): boolean {
    return this.objectMarking === "suffix" || this.ergativeMarking === "suffix";
  }

  /**
   * Return the grammar with the parameters that cannot affect a realization reset.
   *
   * Two canonical grammars are equal exactly when they realize every meaning the same way.
   */
  toCanonical(): Grammar {
    const orderMatters = this.hasCaseSuffix() && this.plural === "suffix";
    return new Grammar({
      wordOrder: this.wordOrder,
      objectMarking: this.objectMarking,
      plural: this.plural,
      pluralCaseOrder: orderMatters ? this.pluralCaseOrder : "plural_first",
      past: this.past,
      subjectAgreement: this.subjectAgreement,
      vowelHarmony: this.hasAffix() ? this.vowelHarmony : "none",
      ergativeMarking: this.ergativeMarking,
      objectPlural: this.plural !== "none" ? this.objectPlural : "none",
      verbNumberStem: this.verbNumberStem,
    });
  }

  /** The field values in Python's field order (what the solver sorts grammars by). */
  values(): string[] {
    return PARAM_NAMES.map((name) => this[name]);
  }

  /** A string that is equal for equal grammars, for Map and Set keys. */
  key(): string {
    return this.values().join("|");
  }

  equals(other: Grammar): boolean {
    return PARAM_NAMES.every((name) => this[name] === other[name]);
  }

  /** The label make_dataset.py gives the grammar: the word order, then each active parameter. */
  label(): string {
    const parts = [this.wordOrder];
    for (const name of OPTIONAL_PARAMS) {
      if (this[name] !== "none") parts.push(`${PY_PARAM_NAMES[name]}-${this[name]}`);
    }
    if (this.hasCaseSuffix() && this.plural === "suffix") {
      parts.push(this.pluralCaseOrder);
    }
    return parts.join("_");
  }

  /** The grammar as Python's dataclasses.asdict would give it. */
  toPython(): Record<string, string> {
    return Object.fromEntries(PARAM_NAMES.map((name) => [PY_PARAM_NAMES[name], this[name]]));
  }

  static fromPython(fields: Readonly<Record<string, string>>): Grammar {
    const values: Partial<Record<ParamName, string>> = {};
    for (const name of PARAM_NAMES) {
      const value = fields[PY_PARAM_NAMES[name]];
      if (value !== undefined) values[name] = value;
    }
    return new Grammar(values);
  }
}

/**
 * Difficulty settings (spec 10); the defaults are those of the published word-order model.
 *
 * activeParams names the parameters switched on in every language, besides word order; each one
 * takes a random non-"none" value per seed (e.g. past is a suffix in some languages and a prefix
 * in others). solverParams names the parameters the solver treats as possibly switched on when it
 * checks that a seed can be solved (spec 8); the default, the five parameters of spec v0.1,
 * reproduces the data of the published models.
 */
export interface EnvConfig {
  readonly numNouns: number;
  readonly numTransitive: number;
  readonly numIntransitive: number;
  readonly activeParams: readonly ParamName[];
  readonly showDictionary: boolean;
  readonly enableJudge: boolean;
  readonly budget: number;
  readonly numTest: number;
  readonly numCompositional: number;
  readonly solverParams: readonly ParamName[];
}

export const DEFAULT_ENV_CONFIG: EnvConfig = Object.freeze({
  numNouns: 4,
  numTransitive: 2,
  numIntransitive: 1,
  activeParams: Object.freeze([]) as readonly ParamName[],
  showDictionary: true,
  enableJudge: false,
  budget: 6,
  numTest: 8,
  numCompositional: 4,
  solverParams: Object.freeze([
    "objectMarking",
    "plural",
    "past",
    "subjectAgreement",
    "vowelHarmony",
  ]) as readonly ParamName[],
});

/** The Python name of each EnvConfig field, in Python's field order. */
export const PY_CONFIG_NAMES: Readonly<Record<keyof EnvConfig, string>> = {
  numNouns: "num_nouns",
  numTransitive: "num_transitive",
  numIntransitive: "num_intransitive",
  activeParams: "active_params",
  showDictionary: "show_dictionary",
  enableJudge: "enable_judge",
  budget: "budget",
  numTest: "num_test",
  numCompositional: "num_compositional",
  solverParams: "solver_params",
};

const PARAM_LIST_FIELDS = ["activeParams", "solverParams"] as const;
const PARAM_NAME_OF_PY: Readonly<Record<string, ParamName>> = Object.fromEntries(
  PARAM_NAMES.map((name) => [PY_PARAM_NAMES[name], name]),
);

/** Return the Python name of a parameter; a name that is no parameter is returned unchanged. */
function pyParamName(name: string): string {
  return PY_PARAM_NAMES[name as ParamName] ?? name;
}

/** Python's repr of a list of strings. */
function pyListRepr(items: readonly string[]): string {
  return `[${items.map((item) => `'${item}'`).join(", ")}]`;
}

/** Python's repr of a tuple of strings. */
function pyTupleRepr(items: readonly string[]): string {
  const inner = items.map((item) => `'${item}'`).join(", ");
  return items.length === 1 ? `(${inner},)` : `(${inner})`;
}

/** Check that the settings can generate an episode; throw like EnvConfig.__post_init__. */
export function validateEnvConfig(config: EnvConfig): void {
  const require = (name: keyof EnvConfig, low: number, high: number | null): void => {
    const value = config[name] as number;
    if (!Number.isInteger(value)) {
      throw new RangeError(`EnvConfig.${PY_CONFIG_NAMES[name]}=${value}: must be an integer`);
    }
    if (value < low || (high !== null && value > high)) {
      const limit = high !== null ? `between ${low} and ${high}` : `at least ${low}`;
      throw new RangeError(`EnvConfig.${PY_CONFIG_NAMES[name]}=${value}: must be ${limit}`);
    }
  };
  // The prompt's example actions use two nouns, a transitive verb and an intransitive verb.
  require("numNouns", 2, NOUN_POOL.length);
  require("numTransitive", 1, TRANSITIVE_POOL.length);
  require("numIntransitive", 1, INTRANSITIVE_POOL.length);
  require("budget", 1, null);
  require("numTest", 1, null);
  require("numCompositional", 0, config.numTest);
  for (const field of PARAM_LIST_FIELDS) {
    const params = config[field].map(pyParamName);
    const unknown = [...new Set(config[field].filter((name) => !OPTIONAL_PARAMS.includes(name)))]
      .map(pyParamName)
      .sort();
    if (unknown.length > 0) {
      throw new RangeError(
        `EnvConfig.${PY_CONFIG_NAMES[field]}: unknown ${pyListRepr(unknown)}; ` +
          `choose from ${pyListRepr(OPTIONAL_PARAMS.map(pyParamName))}`,
      );
    }
    if (new Set(params).size !== params.length) {
      throw new RangeError(`EnvConfig.${PY_CONFIG_NAMES[field]}=${pyTupleRepr(params)}: has duplicates`);
    }
  }
  const active = new Set(config.activeParams);
  const activeRepr = pyTupleRepr(config.activeParams.map(pyParamName));
  if (active.has("vowelHarmony") && !AFFIX_PARAMS.some((name) => active.has(name))) {
    throw new RangeError(
      `EnvConfig.active_params=${activeRepr}: vowel harmony only changes affix vowels, so it ` +
        `needs one of ${pyListRepr(AFFIX_PARAMS.map(pyParamName))} as well`,
    );
  }
  if (active.has("objectPlural") && !active.has("plural")) {
    throw new RangeError(
      `EnvConfig.active_params=${activeRepr}: object_plural changes the plural affix of ` +
        "objects, so it needs plural as well",
    );
  }
  const missing = [...active]
    .filter((name) => !config.solverParams.includes(name))
    .map(pyParamName)
    .sort();
  if (missing.length > 0) {
    throw new RangeError(
      `EnvConfig.solver_params=${pyTupleRepr(config.solverParams.map(pyParamName))}: misses ` +
        `active parameters ${pyListRepr(missing)}, so the solver could never confirm the true ` +
        "grammar",
    );
  }
}

/** Return the default config with overrides applied, validated and frozen. */
export function makeEnvConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  const merged = { ...DEFAULT_ENV_CONFIG, ...overrides };
  const config: EnvConfig = Object.freeze({
    ...merged,
    activeParams: Object.freeze([...merged.activeParams]),
    solverParams: Object.freeze([...merged.solverParams]),
  });
  validateEnvConfig(config);
  return config;
}

/** Read a config with Python's field names (dataclasses.asdict(EnvConfig), e.g. extra_info). */
export function envConfigFromPython(fields: Readonly<Record<string, unknown>>): EnvConfig {
  const overrides: Record<string, unknown> = {};
  for (const [name, pyName] of Object.entries(PY_CONFIG_NAMES)) {
    const value = fields[pyName];
    if (value === undefined) continue;
    // A name that is no parameter is kept, so that validation reports it like Python.
    overrides[name] = (PARAM_LIST_FIELDS as readonly string[]).includes(name)
      ? (value as string[]).map((item) => PARAM_NAME_OF_PY[item] ?? item)
      : value;
  }
  return makeEnvConfig(overrides as Partial<EnvConfig>);
}

/** Return the config with Python's field names, as dataclasses.asdict would give it. */
export function envConfigToPython(config: EnvConfig): Record<string, number | boolean | string[]> {
  return Object.fromEntries(
    (Object.keys(PY_CONFIG_NAMES) as (keyof EnvConfig)[]).map((name) => {
      const value = config[name];
      return [
        PY_CONFIG_NAMES[name],
        typeof value === "object" ? value.map(pyParamName) : value,
      ];
    }),
  );
}

// ---------------------------------------------------------------------------------------------
// Meanings, morphemes and languages.

/** A meaning on the survey sheet; object and objectNumber are null for intransitive verbs. */
export interface Meaning {
  readonly subject: string;
  readonly subjectNumber: string;
  readonly verb: string;
  readonly tense: string;
  readonly object: string | null;
  readonly objectNumber: string | null;
}

export function makeMeaning(
  subject: string,
  subjectNumber: string,
  verb: string,
  tense: string,
  object: string | null = null,
  objectNumber: string | null = null,
): Meaning {
  return Object.freeze({ subject, subjectNumber, verb, tense, object, objectNumber });
}

/** Whether the meaning has an object. */
export function isTransitive(meaning: Meaning): boolean {
  return meaning.object !== null;
}

/** Format the meaning as the Slot=value list the prompts use. */
export function meaningToText(meaning: Meaning): string {
  const slots: [string, string | null][] = [
    ["Subject", meaning.subject],
    ["SubjectNumber", meaning.subjectNumber],
  ];
  if (isTransitive(meaning)) {
    slots.push(["Object", meaning.object], ["ObjectNumber", meaning.objectNumber]);
  }
  slots.push(["Verb", meaning.verb], ["Tense", meaning.tense]);
  return slots.map(([slot, value]) => `${slot}=${value ?? "None"}`).join(", ");
}

/** Return whether two or more features are marked, which no single question shows. */
export function isCompositional(meaning: Meaning): boolean {
  const marked = [
    meaning.subjectNumber === "plural",
    meaning.objectNumber === "plural",
    meaning.tense === "past",
  ];
  return isTransitive(meaning) && marked.filter(Boolean).length >= 2;
}

/** A string that is equal exactly for equal meanings (Python tuple equality), for Sets and Maps. */
export function meaningKey(meaning: Meaning): string {
  return JSON.stringify([
    meaning.subject,
    meaning.subjectNumber,
    meaning.verb,
    meaning.tense,
    meaning.object,
    meaning.objectNumber,
  ]);
}

export function meaningsEqual(a: Meaning, b: Meaning): boolean {
  return (
    a.subject === b.subject &&
    a.subjectNumber === b.subjectNumber &&
    a.verb === b.verb &&
    a.tense === b.tense &&
    a.object === b.object &&
    a.objectNumber === b.objectNumber
  );
}

/** A slot of a word: a root (name is a key of Language.roots) or an affix (name is an affix). */
export interface Morpheme {
  readonly isRoot: boolean;
  readonly name: string;
}
export type Word = readonly Morpheme[];
export type Template = readonly Word[];

/**
 * A generated language: its grammar, the vocabulary of its survey sheet and its morphemes.
 *
 * roots maps each concept to its root and, under verbNumberStem, each verb's pluralStemKey to its
 * second stem.
 */
export class Language {
  constructor(
    readonly grammar: Grammar,
    readonly nouns: readonly string[],
    readonly transitive: readonly string[],
    readonly intransitive: readonly string[],
    readonly roots: Readonly<Record<string, string>>,
    readonly affixes: Readonly<Record<string, string>>,
  ) {}

  /** Return every meaning the survey sheet allows, in a fixed order. */
  listMeanings(): Meaning[] {
    const meanings: Meaning[] = [];
    for (const subject of this.nouns) {
      for (const subjectNumber of NUMBERS) {
        for (const tense of TENSES) {
          for (const verb of this.intransitive) {
            meanings.push(makeMeaning(subject, subjectNumber, verb, tense));
          }
          for (const verb of this.transitive) {
            for (const obj of this.nouns) {
              for (const objectNumber of NUMBERS) {
                meanings.push(makeMeaning(subject, subjectNumber, verb, tense, obj, objectNumber));
              }
            }
          }
        }
      }
    }
    return meanings;
  }

  /** Return the sentence of the language that says meaning. */
  realize(meaning: Meaning): string {
    const words: string[] = [];
    for (const word of buildSentenceTemplate(this.grammar, meaning)) {
      const root = this.roots[word.find((morpheme) => morpheme.isRoot)!.name]!;
      const pieces: string[] = [];
      for (const morpheme of word) {
        if (morpheme.isRoot) {
          pieces.push(this.roots[morpheme.name]!);
        } else if (this.grammar.vowelHarmony === "copy") {
          pieces.push(this.affixes[morpheme.name]![0]! + root[root.length - 1]!);
        } else {
          pieces.push(this.affixes[morpheme.name]!);
        }
      }
      words.push(pieces.join(""));
    }
    return words.join(" ");
  }
}

const ROOT_MORPHEMES = new Map<string, Morpheme>();
const AFFIX_MORPHEMES = new Map<string, Morpheme>();

/** Return the shared, frozen morpheme object of (isRoot, name). */
export function morpheme(isRoot: boolean, name: string): Morpheme {
  const table = isRoot ? ROOT_MORPHEMES : AFFIX_MORPHEMES;
  let result = table.get(name);
  if (result === undefined) {
    result = Object.freeze({ isRoot, name });
    table.set(name, result);
  }
  return result;
}

/**
 * Return the words that say meaning as morpheme slots, without their forms.
 *
 * Language.realize fills the slots with the language's forms; the solver fills them with the
 * forms it has inferred from observations.
 */
export function buildSentenceTemplate(grammar: Grammar, meaning: Meaning): Template {
  const buildNoun = (concept: string, number: string | null, role: "S" | "A" | "O"): Word => {
    const plural = number === "plural";
    const pluralAffix = role === "O" && grammar.objectPlural === "distinct" ? "object_plural" : "plural";
    const prefixes = plural && grammar.plural === "prefix" ? [pluralAffix] : [];
    const suffixes = plural && grammar.plural === "suffix" ? [pluralAffix] : [];
    let caseAffix: string | null = null;
    if (role === "O" && grammar.objectMarking === "suffix") caseAffix = "object";
    if (role === "A" && grammar.ergativeMarking === "suffix") caseAffix = "ergative";
    if (caseAffix !== null) {
      if (grammar.pluralCaseOrder === "case_first") {
        suffixes.unshift(caseAffix);
      } else {
        suffixes.push(caseAffix);
      }
    }
    return [
      ...prefixes.map((name) => morpheme(false, name)),
      morpheme(true, concept),
      ...suffixes.map((name) => morpheme(false, name)),
    ];
  };

  const verbPrefixes: string[] = [];
  const verbSuffixes: string[] = [];
  if (meaning.tense === "past" && grammar.past === "prefix") verbPrefixes.push("past");
  if (meaning.tense === "past" && grammar.past === "suffix") verbSuffixes.push("past");
  if (meaning.subjectNumber === "plural" && grammar.subjectAgreement === "suffix") {
    verbSuffixes.push("agreement");
  }
  const transitive = isTransitive(meaning);
  const absolutiveNumber = transitive ? meaning.objectNumber : meaning.subjectNumber;
  const pluralStem =
    (grammar.verbNumberStem === "subject" && meaning.subjectNumber === "plural") ||
    (grammar.verbNumberStem === "absolutive" && absolutiveNumber === "plural");
  const stem = pluralStem ? pluralStemKey(meaning.verb) : meaning.verb;
  const words: Record<string, Word> = {
    S: buildNoun(meaning.subject, meaning.subjectNumber, transitive ? "A" : "S"),
    V: [
      ...verbPrefixes.map((name) => morpheme(false, name)),
      morpheme(true, stem),
      ...verbSuffixes.map((name) => morpheme(false, name)),
    ],
  };
  if (transitive) {
    words["O"] = buildNoun(meaning.object!, meaning.objectNumber, "O");
    return [...grammar.wordOrder].map((role) => words[role]!);
  }
  const subjectFirst = grammar.wordOrder.indexOf("S") < grammar.wordOrder.indexOf("V");
  return subjectFirst ? [words["S"]!, words["V"]!] : [words["V"]!, words["S"]!];
}

/** Generate the language and the test set of seed; equal seeds and configs give equal ones. */
export function generate(
  seed: PySeed,
  config: EnvConfig,
): { language: Language; tests: Meaning[] } {
  validateEnvConfig(config);
  const rng = new PyRandom(seed);

  const sampleGrammar = (): Grammar => {
    const wordOrder = rng.choice(PARAM_CHOICES.wordOrder);
    // Values are drawn in OPTIONAL_PARAMS order, so the order of activeParams does not change
    // the languages.
    const values: Partial<Record<ParamName, string>> = {};
    for (const p of OPTIONAL_PARAMS) {
      if (!config.activeParams.includes(p)) continue;
      const choices: readonly string[] = PARAM_CHOICES[p];
      values[p] = rng.choice(choices.filter((v) => v !== "none"));
    }
    values.pluralCaseOrder = rng.choice(PARAM_CHOICES.pluralCaseOrder);
    return new Grammar({ ...values, wordOrder }).toCanonical();
  };

  const grammar = sampleGrammar();
  const nouns = rng.sample(NOUN_POOL, config.numNouns);
  const transitive = rng.sample(TRANSITIVE_POOL, config.numTransitive);
  const intransitive = rng.sample(INTRANSITIVE_POOL, config.numIntransitive);

  // Affixes and roots draw from disjoint consonant sets: every morpheme differs, and an affix
  // syllable never occurs inside a root, so morpheme boundaries are unique (roots have two CV
  // syllables, affixes one). Extra affixes take the last consonants and draw after the others,
  // so grammars without them get the same forms as before the extra affixes existed.
  const extraAffixes = EXTRA_AFFIX_PARAMS.filter(([, param]) => grammar[param] !== "none").map(
    ([name]) => name,
  );
  const consonants = rng.sample([...CONSONANTS], CONSONANTS.length);
  const numRootConsonants = CONSONANTS.length - AFFIX_NAMES.length - extraAffixes.length;
  const rootConsonants = consonants.slice(AFFIX_NAMES.length, AFFIX_NAMES.length + numRootConsonants);
  const affixNames = [...AFFIX_NAMES, ...extraAffixes];
  const affixConsonants = [
    ...consonants.slice(0, AFFIX_NAMES.length),
    ...consonants.slice(consonants.length - extraAffixes.length),
  ];
  const affixes: Record<string, string> = {};
  affixNames.forEach((name, i) => {
    affixes[name] = affixConsonants[i]! + rng.choice(VOWELS);
  });
  const roots: Record<string, string> = {};
  const used = new Set<string>();
  const addRoot = (key: string): void => {
    let form: string;
    for (;;) {
      form = "";
      for (let i = 0; i < 2; i++) form += rng.choice(rootConsonants) + rng.choice(VOWELS);
      if (!used.has(form)) break;
    }
    roots[key] = form;
    used.add(form);
  };
  for (const concept of [...nouns, ...transitive, ...intransitive]) addRoot(concept);
  if (grammar.verbNumberStem !== "none") {
    for (const verb of [...transitive, ...intransitive]) addRoot(pluralStemKey(verb));
  }
  const language = new Language(
    grammar,
    Object.freeze(nouns),
    Object.freeze(transitive),
    Object.freeze(intransitive),
    Object.freeze(roots),
    Object.freeze(affixes),
  );

  const meanings = language.listMeanings();
  const compositional = meanings.filter(isCompositional);
  const tests = rng.sample(compositional, config.numCompositional);
  const testKeys = new Set(tests.map(meaningKey));
  const rest = meanings.filter((meaning) => !testKeys.has(meaningKey(meaning)));
  tests.push(...rng.sample(rest, config.numTest - config.numCompositional));
  rng.shuffle(tests);
  return { language, tests };
}

/** Lowercase, strip, collapse whitespace and drop hyphens, in this order (spec 7.1). */
export function normalize(text: string): string {
  let result = pyStrip(text.toLowerCase());
  result = result.replace(WHITESPACE_RUN_RE, " ");
  return result.replaceAll("-", "");
}

// ---------------------------------------------------------------------------------------------
// Actions and episodes.

/** An investigation action read from the agent's turn. */
export type ParsedAction =
  | { readonly kind: "ask"; readonly meaning: Meaning }
  | { readonly kind: "judge"; readonly sentence: string }
  | { readonly kind: "done" }
  | { readonly kind: "invalid"; readonly error: string };

const INVALID_FORMAT: ParsedAction = Object.freeze({ kind: "invalid", error: ERROR_FORMAT });
const INVALID_RANGE: ParsedAction = Object.freeze({ kind: "invalid", error: ERROR_RANGE });
const INVALID_TYPE: ParsedAction = Object.freeze({ kind: "invalid", error: ERROR_TYPE });

/** Check the slots of an ask action against the survey sheet (Episode.step's check_meaning). */
function checkMeaning(language: Language, slots: Map<string, string>): ParsedAction {
  const allowed = new Map<string, readonly string[]>([
    ["Subject", language.nouns],
    ["SubjectNumber", NUMBERS],
    ["Object", language.nouns],
    ["ObjectNumber", NUMBERS],
    ["Verb", [...language.transitive, ...language.intransitive]],
    ["Tense", TENSES],
  ]);
  for (const [slot, value] of slots) {
    const values = allowed.get(slot);
    if (values !== undefined && !values.includes(value)) return INVALID_RANGE;
  }
  const verb = slots.get("Verb");
  if (verb === undefined) return INVALID_TYPE;
  const required: readonly string[] = language.transitive.includes(verb)
    ? TRANSITIVE_SLOTS
    : INTRANSITIVE_SLOTS;
  if (slots.size !== required.length || !required.every((slot) => slots.has(slot))) {
    return INVALID_TYPE;
  }
  const meaning = makeMeaning(
    slots.get("Subject")!,
    slots.get("SubjectNumber")!,
    verb,
    slots.get("Tense")!,
    slots.get("Object") ?? null,
    slots.get("ObjectNumber") ?? null,
  );
  return { kind: "ask", meaning };
}

/**
 * Read the action of one investigation turn (Episode.step's parse). A turn that was cut off at
 * the token limit is a format error whatever it contains; step() handles that before parsing.
 */
export function parseAction(text: string, language: Language, enableJudge: boolean): ParsedAction {
  const actions = [...text.matchAll(ACTION_RE)].map((match) => match[1]!);
  if (actions.length !== 1) return INVALID_FORMAT;
  const body = pyStrip(actions[0]!);
  if (body === "done") return { kind: "done" };
  const judge = JUDGE_RE.exec(body);
  if (judge) {
    if (enableJudge) return { kind: "judge", sentence: judge[1]! };
    return INVALID_FORMAT;
  }
  const ask = ASK_RE.exec(body);
  if (!ask) return INVALID_FORMAT;
  const slots = new Map<string, string>();
  for (const pair of ask[1]!.split(",")) {
    const eq = pair.indexOf("=");
    if (eq < 0) return INVALID_FORMAT;
    const slot = pyStrip(pair.slice(0, eq));
    const value = pyStrip(pair.slice(eq + 1));
    if (!slot || !value || slots.has(slot)) return INVALID_FORMAT;
    slots.set(slot, value);
  }
  return checkMeaning(language, slots);
}

/** Reward of an episode (spec 7) and the counts that verl logs with it. */
/**
 * Conditions an investigation can cover; EpisodeScore.coverage is the fraction of them that occur
 * in at least one asked meaning. Each pairs a marked feature with a verb type, since a language may
 * mark a feature differently on transitive and intransitive sentences.
 */
export const COVERAGE_CONDITIONS: readonly (readonly [string, (meaning: Meaning) => boolean])[] = [
  ["transitive_plural_subject", (m) => isTransitive(m) && m.subjectNumber === "plural"],
  ["transitive_plural_object", (m) => isTransitive(m) && m.objectNumber === "plural"],
  ["transitive_past", (m) => isTransitive(m) && m.tense === "past"],
  ["intransitive_plural_subject", (m) => !isTransitive(m) && m.subjectNumber === "plural"],
  ["intransitive_past", (m) => !isTransitive(m) && m.tense === "past"],
];

export interface EpisodeScore {
  readonly reward: number;
  /** The fraction of COVERAGE_CONDITIONS covered by the asked meanings. */
  readonly coverage: number;
  readonly numAsks: number;
  readonly numJudges: number;
  readonly numInvalid: number;
  readonly numScorable: number;
  readonly numCorrect: number;
  /** 1 if the test answer was cut off at the token limit. */
  readonly testTruncated: number;
}

/** One graded test item (Episode.grade). */
export interface GradedItem {
  /** The 1-based item id of the test prompt. */
  readonly id: number;
  readonly meaning: Meaning;
  /** The first answer given for the id, unnormalized, or null if there is none. */
  readonly answer: string | null;
  /** The language's realization of the meaning. */
  readonly expected: string;
  /** False if the meaning was asked during the investigation (it is not scored). */
  readonly scorable: boolean;
  /** scorable && !truncated && normalize(answer ?? "") === expected. */
  readonly correct: boolean;
}

export interface EpisodeGrade {
  readonly score: EpisodeScore;
  /** Every test item, in test order. */
  readonly items: readonly GradedItem[];
}

/**
 * Return the answer that text gives to each test item id, like score()'s parsing: the first
 * <answer id="N"> for an id wins, and N may use any Unicode decimal digits, as Python's \d and
 * int() accept them. Ids above Number.MAX_SAFE_INTEGER are dropped (no test item has one).
 */
export function parseAnswers(text: string): Map<number, string> {
  const answers = new Map<number, string>();
  for (const match of text.matchAll(ANSWER_RE)) {
    const id = pyIntValue(match[1]!);
    if (id !== null && !answers.has(id)) answers.set(id, match[2]!);
  }
  return answers;
}

/** One episode: the informant's side of the investigation and the scoring of the test. */
export class Episode {
  readonly language: Language;
  readonly tests: readonly Meaning[];
  budgetLeft: number;
  /** The meanings asked so far, by meaningKey, in the order they were first asked. */
  readonly asked = new Map<string, Meaning>();
  numAsks = 0;
  numJudges = 0;
  numInvalid = 0;
  inTestPhase = false;
  /** The action that the last step() read (not part of the Python environment). */
  lastAction: ParsedAction | null = null;
  private sentences: Set<string> | null = null;

  /** Generate the language and the test set of seed. */
  constructor(
    readonly seed: PySeed,
    readonly config: EnvConfig,
  ) {
    const { language, tests } = generate(seed, config);
    this.language = language;
    this.tests = Object.freeze(tests);
    this.budgetLeft = config.budget;
  }

  /** Return whether meaning has been asked (the test items asked are not scored). */
  isAsked(meaning: Meaning): boolean {
    return this.asked.has(meaningKey(meaning));
  }

  /** Return the first user message: the task, survey sheet, dictionary and actions. */
  buildInitialPrompt(): string {
    const lang = this.language;
    const config = this.config;
    let lines = [
      "You are a field linguist studying an unknown language. A native speaker (the " +
        "informant) will tell you how to say any meaning you ask about. Find out the grammar, " +
        "then translate new meanings.",
      "",
      "## Survey sheet",
      `Nouns: ${lang.nouns.join(", ")}`,
      `Transitive verbs: ${lang.transitive.join(", ")}`,
      `Intransitive verbs: ${lang.intransitive.join(", ")}`,
      `SubjectNumber, ObjectNumber: ${NUMBERS.join(", ")}`,
      `Tense: ${TENSES.join(", ")}`,
      "A sentence with a transitive verb needs exactly: " + TRANSITIVE_SLOTS.join(", ") + ".",
      "A sentence with an intransitive verb needs exactly: " + INTRANSITIVE_SLOTS.join(", ") + ".",
    ];
    if (config.showDictionary) {
      lines.push("", "## Root dictionary");
      const concepts = [...lang.nouns, ...lang.transitive, ...lang.intransitive];
      lines = lines.concat(concepts.map((concept) => `${concept} = ${lang.roots[concept]}`));
    }
    const noun1 = lang.nouns[0]!;
    const noun2 = lang.nouns[1]!;
    lines.push(
      "",
      "## Actions",
      "Each turn, output exactly one action inside <action></action>, using only names from " +
        "the survey sheet. The actions are:",
      `<action>ask(Subject=${noun1}, SubjectNumber=plural, Object=${noun2}, ` +
        `ObjectNumber=singular, Verb=${lang.transitive[0]}, Tense=past)</action>`,
      `<action>ask(Subject=${noun2}, SubjectNumber=singular, Verb=${lang.intransitive[0]}, ` +
        "Tense=present)</action>",
      "  The informant says that meaning in the language. A transitive verb needs all six " +
        "slots; an intransitive verb takes no Object or ObjectNumber. Slots may come in any " +
        "order.",
    );
    if (config.enableJudge) {
      lines.push(
        '<action>judge("sentence in the language")</action>',
        "  The informant says whether the sentence is grammatical.",
      );
    }
    lines.push(
      "<action>done</action>",
      "  End the investigation.",
      `You have ${config.budget} questions. Every action except done uses one question, ` +
        "including invalid actions. When the questions run out, the investigation ends.",
      'An invalid action gets one of three errors: "format error" (the action cannot be ' +
        'read), "out of range" (a value is not on the survey sheet), or "type error" (the ' +
        "slots do not match the verb).",
      "",
      "After the investigation you will translate meanings you have not seen. The language " +
        "is written in lowercase Latin letters with words separated by single spaces and no " +
        "punctuation.",
    );
    return lines.join("\n");
  }

  /** Return the test prompt that ends the investigation (step's build_test_prompt). */
  buildTestPrompt(): string {
    const lines = [
      "The investigation is over. Translate each meaning below into the language. " +
        'Write every answer as <answer id="N">sentence</answer>.',
      "",
      ...this.tests.map((meaning, i) => `${i + 1}. ${meaningToText(meaning)}`),
    ];
    return lines.join("\n");
  }

  /**
   * Apply one investigation turn and return the informant's reply.
   *
   * The reply ends with the test prompt when the investigation ends. A turn cut off at the token
   * limit is a format error (spec 5.4).
   */
  step(text: string, truncated: boolean = false): string {
    const lang = this.language;
    const action = truncated ? INVALID_FORMAT : parseAction(text, lang, this.config.enableJudge);
    this.lastAction = action;
    if (action.kind === "done") {
      this.inTestPhase = true;
      return this.buildTestPrompt();
    }
    this.budgetLeft -= 1;
    let reply: string;
    if (action.kind === "ask") {
      this.numAsks += 1;
      const key = meaningKey(action.meaning);
      if (!this.asked.has(key)) this.asked.set(key, action.meaning);
      reply = `Informant: ${lang.realize(action.meaning)}`;
    } else if (action.kind === "judge") {
      this.numJudges += 1;
      if (this.sentences === null) {
        this.sentences = new Set(lang.listMeanings().map((meaning) => lang.realize(meaning)));
      }
      const grammatical = this.sentences.has(normalize(action.sentence));
      reply = `Informant: ${grammatical ? "grammatical" : "ungrammatical"}`;
    } else {
      this.numInvalid += 1;
      reply = action.error;
    }
    if (this.budgetLeft <= 0) {
      this.inTestPhase = true;
      return reply + "\nYou have used all your questions.\n\n" + this.buildTestPrompt();
    }
    const plural = this.budgetLeft > 1 ? "s" : "";
    return reply + `\n(${this.budgetLeft} question${plural} left)`;
  }

  /**
   * Grade the test answers in text item by item. The score is exactly what score() returns:
   * score() is computed from it.
   */
  grade(text: string, truncated: boolean = false): EpisodeGrade {
    const answers = parseAnswers(text);
    const items: GradedItem[] = this.tests.map((meaning, i) => {
      const answer = answers.get(i + 1) ?? null;
      const expected = this.language.realize(meaning);
      const scorable = !this.isAsked(meaning);
      const correct = scorable && !truncated && normalize(answer ?? "") === expected;
      return { id: i + 1, meaning, answer, expected, scorable, correct };
    });
    const numScorable = items.filter((item) => item.scorable).length;
    const numCorrect = items.filter((item) => item.correct).length;
    const asked = [...this.asked.values()];
    const numCovered = COVERAGE_CONDITIONS.filter(([, condition]) => asked.some(condition)).length;
    const score: EpisodeScore = {
      reward: numScorable ? numCorrect / numScorable : 0.0,
      coverage: numCovered / COVERAGE_CONDITIONS.length,
      numAsks: this.numAsks,
      numJudges: this.numJudges,
      numInvalid: this.numInvalid,
      numScorable,
      numCorrect,
      testTruncated: truncated ? 1 : 0,
    };
    return { score, items };
  }

  /**
   * Score the test answers in text (spec 7).
   *
   * The reward is the fraction of scored test items answered exactly; items identical to an
   * asked meaning are not scored, and an answer cut off at the token limit scores 0.
   */
  score(text: string, truncated: boolean = false): EpisodeScore {
    return this.grade(text, truncated).score;
  }
}
