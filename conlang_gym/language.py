"""conlang-gym: an artificial-language field-work environment (spec v0.1).

A seed fixes a random artificial language and a test set. The agent asks an informant how to say
meanings, then translates meanings it has not seen. The reward is an exact match against the
language's own realization, so no human or LLM grading is needed.
"""

import random
import re
from dataclasses import dataclass
from typing import Literal, NamedTuple

CONSONANTS = "ptkbdgmnslrvzfh"
VOWELS = "aeiou"
NOUN_POOL = ("dog", "cat", "bird", "fish", "horse", "frog", "child", "woman", "man", "bear")
TRANSITIVE_POOL = ("see", "chase", "bite", "call", "help", "hear")
INTRANSITIVE_POOL = ("sleep", "run", "sing", "swim")
NUMBERS = ("singular", "plural")
TENSES = ("present", "past")

# The key order matches Grammar's field order: the solver builds grammars from the product of these.
PARAM_CHOICES = {
    "word_order": ("SOV", "SVO", "VSO", "VOS", "OSV", "OVS"),
    "object_marking": ("none", "suffix"),
    "plural": ("none", "suffix", "prefix"),
    "plural_case_order": ("plural_first", "case_first"),
    "past": ("none", "suffix", "prefix"),
    "subject_agreement": ("none", "suffix"),
    "vowel_harmony": ("none", "copy"),
}
# Parameters that can be switched on in addition to word order, which is always active.
OPTIONAL_PARAMS = ("object_marking", "plural", "past", "subject_agreement", "vowel_harmony")
AFFIX_NAMES = ("plural", "object", "past", "agreement")

TRANSITIVE_SLOTS = ("Subject", "SubjectNumber", "Object", "ObjectNumber", "Verb", "Tense")
INTRANSITIVE_SLOTS = ("Subject", "SubjectNumber", "Verb", "Tense")

ERROR_FORMAT = "Error: format error."
ERROR_RANGE = "Error: out of range."
ERROR_TYPE = "Error: type error."

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_ASK_RE = re.compile(r"^ask\((.*)\)$", re.DOTALL)
_JUDGE_RE = re.compile(r'^judge\(\s*"(.*)"\s*\)$', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer id="(\d+)">(.*?)</answer>', re.DOTALL)


@dataclass(frozen=True)
class Grammar:
    """One value per grammatical parameter (spec 3.3); "none" switches a parameter off."""

    word_order: str = "SVO"
    object_marking: str = "none"
    plural: str = "none"
    plural_case_order: str = "plural_first"
    past: str = "none"
    subject_agreement: str = "none"
    vowel_harmony: str = "none"

    def has_affix(self) -> bool:
        """Return whether any parameter puts an affix on a word."""
        return (
            self.object_marking != "none"
            or self.plural != "none"
            or self.past != "none"
            or self.subject_agreement != "none"
        )

    def to_canonical(self) -> "Grammar":
        """Return the grammar with the parameters that cannot affect a realization reset.

        Two canonical grammars are equal exactly when they realize every meaning the same way.
        """
        order_matters = self.object_marking == "suffix" and self.plural == "suffix"
        return Grammar(
            word_order=self.word_order,
            object_marking=self.object_marking,
            plural=self.plural,
            plural_case_order=self.plural_case_order if order_matters else "plural_first",
            past=self.past,
            subject_agreement=self.subject_agreement,
            vowel_harmony=self.vowel_harmony if self.has_affix() else "none",
        )


@dataclass(frozen=True)
class EnvConfig:
    """Difficulty settings (spec 10); the defaults are the Qwen3.5-2B initial settings (spec 11)."""

    num_nouns: int = 4
    num_transitive: int = 2
    num_intransitive: int = 1
    num_active_params: int = 1
    show_dictionary: bool = True
    enable_judge: bool = False
    budget: int = 6
    num_test: int = 8
    num_compositional: int = 4

    def __post_init__(self) -> None:
        """Check that the settings can generate an episode."""

        def require(name: str, low: int, high: int | None) -> None:
            value = getattr(self, name)
            if value < low or (high is not None and value > high):
                limit = f"between {low} and {high}" if high is not None else f"at least {low}"
                raise ValueError(f"EnvConfig.{name}={value}: must be {limit}")

        # The prompt's example actions use two nouns, a transitive verb and an intransitive verb.
        require("num_nouns", 2, len(NOUN_POOL))
        require("num_transitive", 1, len(TRANSITIVE_POOL))
        require("num_intransitive", 1, len(INTRANSITIVE_POOL))
        require("num_active_params", 0, len(OPTIONAL_PARAMS))
        require("budget", 1, None)
        require("num_test", 1, None)
        require("num_compositional", 0, self.num_test)


class Meaning(NamedTuple):
    """A meaning on the survey sheet; object and object_number are None for intransitive verbs."""

    subject: str
    subject_number: str
    verb: str
    tense: str
    object: str | None = None
    object_number: str | None = None

    @property
    def transitive(self) -> bool:
        """Whether the meaning has an object."""
        return self.object is not None

    def to_text(self) -> str:
        """Format the meaning as the Slot=value list the prompts use."""
        slots = {"Subject": self.subject, "SubjectNumber": self.subject_number}
        if self.transitive:
            slots |= {"Object": self.object, "ObjectNumber": self.object_number}
        slots |= {"Verb": self.verb, "Tense": self.tense}
        return ", ".join(f"{slot}={value}" for slot, value in slots.items())

    def is_compositional(self) -> bool:
        """Return whether two or more features are marked, which no single question shows."""
        marked = [
            self.subject_number == "plural",
            self.object_number == "plural",
            self.tense == "past",
        ]
        return self.transitive and sum(marked) >= 2


class Morpheme(NamedTuple):
    """A slot of a word: a root (name is a concept) or an affix (name is one of AFFIX_NAMES)."""

    is_root: bool
    name: str


type Word = list[Morpheme]
type Template = list[Word]


@dataclass(frozen=True)
class Language:
    """A generated language: its grammar, the vocabulary of its survey sheet and its morphemes."""

    grammar: Grammar
    nouns: tuple[str, ...]
    transitive: tuple[str, ...]
    intransitive: tuple[str, ...]
    roots: dict[str, str]
    affixes: dict[str, str]

    def list_meanings(self) -> list[Meaning]:
        """Return every meaning the survey sheet allows, in a fixed order."""
        meanings = []
        for subject in self.nouns:
            for subject_number in NUMBERS:
                for tense in TENSES:
                    for verb in self.intransitive:
                        meanings.append(Meaning(subject, subject_number, verb, tense))
                    for verb in self.transitive:
                        for obj in self.nouns:
                            for object_number in NUMBERS:
                                meanings.append(
                                    Meaning(
                                        subject, subject_number, verb, tense, obj, object_number
                                    )
                                )
        return meanings

    def realize(self, meaning: Meaning) -> str:
        """Return the sentence of the language that says meaning."""
        words = []
        for word in build_sentence_template(self.grammar, meaning):
            root = next(self.roots[morpheme.name] for morpheme in word if morpheme.is_root)
            pieces = []
            for morpheme in word:
                if morpheme.is_root:
                    pieces.append(self.roots[morpheme.name])
                elif self.grammar.vowel_harmony == "copy":
                    pieces.append(self.affixes[morpheme.name][0] + root[-1])
                else:
                    pieces.append(self.affixes[morpheme.name])
            words.append("".join(pieces))
        return " ".join(words)


def build_sentence_template(grammar: Grammar, meaning: Meaning) -> Template:
    """Return the words that say meaning as morpheme slots, without their forms.

    Language.realize fills the slots with the language's forms; the solver fills them with the forms
    it has inferred from observations.
    """

    def build_noun(concept: str, number: str, is_object: bool) -> Word:
        plural = number == "plural"
        prefixes = ["plural"] if plural and grammar.plural == "prefix" else []
        suffixes = ["plural"] if plural and grammar.plural == "suffix" else []
        if is_object and grammar.object_marking == "suffix":
            if grammar.plural_case_order == "case_first":
                suffixes.insert(0, "object")
            else:
                suffixes.append("object")
        return (
            [Morpheme(False, name) for name in prefixes]
            + [Morpheme(True, concept)]
            + [Morpheme(False, name) for name in suffixes]
        )

    verb_prefixes, verb_suffixes = [], []
    if meaning.tense == "past" and grammar.past == "prefix":
        verb_prefixes.append("past")
    if meaning.tense == "past" and grammar.past == "suffix":
        verb_suffixes.append("past")
    if meaning.subject_number == "plural" and grammar.subject_agreement == "suffix":
        verb_suffixes.append("agreement")
    words = {
        "S": build_noun(meaning.subject, meaning.subject_number, False),
        "V": [Morpheme(False, name) for name in verb_prefixes]
        + [Morpheme(True, meaning.verb)]
        + [Morpheme(False, name) for name in verb_suffixes],
    }
    if meaning.transitive:
        words["O"] = build_noun(meaning.object, meaning.object_number, True)
        return [words[role] for role in grammar.word_order]
    subject_first = grammar.word_order.index("S") < grammar.word_order.index("V")
    return [words["S"], words["V"]] if subject_first else [words["V"], words["S"]]


def generate(seed: int, config: EnvConfig) -> tuple[Language, list[Meaning]]:
    """Generate the language and the test set of seed; equal seeds and configs give equal ones."""
    rng = random.Random(seed)

    def sample_grammar() -> Grammar:
        word_order = rng.choice(PARAM_CHOICES["word_order"])
        while True:
            active = rng.sample(OPTIONAL_PARAMS, config.num_active_params)
            # Vowel harmony is invisible unless some affix exists, so it never comes alone.
            if "vowel_harmony" not in active or any(p != "vowel_harmony" for p in active):
                break
        values = {p: rng.choice([v for v in PARAM_CHOICES[p] if v != "none"]) for p in active}
        values["plural_case_order"] = rng.choice(PARAM_CHOICES["plural_case_order"])
        return Grammar(word_order=word_order, **values).to_canonical()

    grammar = sample_grammar()
    nouns = tuple(rng.sample(NOUN_POOL, config.num_nouns))
    transitive = tuple(rng.sample(TRANSITIVE_POOL, config.num_transitive))
    intransitive = tuple(rng.sample(INTRANSITIVE_POOL, config.num_intransitive))

    # Affixes and roots draw from disjoint consonant sets: every morpheme differs, and an affix
    # syllable never occurs inside a root, so morpheme boundaries are unique (roots have two CV
    # syllables, affixes one).
    consonants = rng.sample(CONSONANTS, len(CONSONANTS))
    affix_consonants = consonants[: len(AFFIX_NAMES)]
    root_consonants = consonants[len(AFFIX_NAMES) :]
    affixes = {
        name: consonant + rng.choice(VOWELS)
        for name, consonant in zip(AFFIX_NAMES, affix_consonants, strict=True)
    }
    roots: dict[str, str] = {}
    for concept in nouns + transitive + intransitive:
        while True:
            form = "".join(rng.choice(root_consonants) + rng.choice(VOWELS) for _ in range(2))
            if form not in roots.values():
                break
        roots[concept] = form
    language = Language(grammar, nouns, transitive, intransitive, roots, affixes)

    meanings = language.list_meanings()
    compositional = [meaning for meaning in meanings if meaning.is_compositional()]
    tests = rng.sample(compositional, config.num_compositional)
    rest = [meaning for meaning in meanings if meaning not in tests]
    tests += rng.sample(rest, config.num_test - config.num_compositional)
    rng.shuffle(tests)
    return language, tests


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace and drop hyphens, in this order (spec 7.1)."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace("-", "")


@dataclass(frozen=True)
class ParsedAction:
    """An investigation action read from the agent's turn."""

    kind: Literal["ask", "judge", "done", "invalid"]
    meaning: Meaning | None = None  # ask
    sentence: str | None = None  # judge
    error: str | None = None  # invalid: the fixed error message


@dataclass(frozen=True)
class EpisodeScore:
    """Reward of an episode (spec 7) and the counts that verl logs with it."""

    reward: float
    num_asks: int
    num_judges: int
    num_invalid: int
    num_scorable: int
    num_correct: int
    test_truncated: int  # 1 if the test answer was cut off at the token limit


class Episode:
    """One episode: the informant's side of the investigation and the scoring of the test."""

    def __init__(self, seed: int, config: EnvConfig) -> None:
        """Generate the language and the test set of seed."""
        self.seed = seed
        self.config = config
        self.language, self.tests = generate(seed, config)
        self.budget_left = config.budget
        self.asked: set[Meaning] = set()
        self.num_asks = 0
        self.num_judges = 0
        self.num_invalid = 0
        self.in_test_phase = False
        self._sentences: set[str] | None = None

    def build_initial_prompt(self) -> str:
        """Return the first user message: the task, survey sheet, dictionary and actions."""
        lang, config = self.language, self.config
        lines = [
            "You are a field linguist studying an unknown language. A native speaker (the "
            "informant) will tell you how to say any meaning you ask about. Find out the grammar, "
            "then translate new meanings.",
            "",
            "## Survey sheet",
            f"Nouns: {', '.join(lang.nouns)}",
            f"Transitive verbs: {', '.join(lang.transitive)}",
            f"Intransitive verbs: {', '.join(lang.intransitive)}",
            f"SubjectNumber, ObjectNumber: {', '.join(NUMBERS)}",
            f"Tense: {', '.join(TENSES)}",
            "A sentence with a transitive verb needs exactly: " + ", ".join(TRANSITIVE_SLOTS) + ".",
            "A sentence with an intransitive verb needs exactly: "
            + ", ".join(INTRANSITIVE_SLOTS)
            + ".",
        ]
        if config.show_dictionary:
            lines += ["", "## Root dictionary"]
            concepts = lang.nouns + lang.transitive + lang.intransitive
            lines += [f"{concept} = {lang.roots[concept]}" for concept in concepts]
        noun1, noun2 = lang.nouns[0], lang.nouns[1]
        lines += [
            "",
            "## Actions",
            "Each turn, output exactly one action inside <action></action>, using only names from "
            "the survey sheet. The actions are:",
            f"<action>ask(Subject={noun1}, SubjectNumber=plural, Object={noun2}, "
            f"ObjectNumber=singular, Verb={lang.transitive[0]}, Tense=past)</action>",
            f"<action>ask(Subject={noun2}, SubjectNumber=singular, Verb={lang.intransitive[0]}, "
            "Tense=present)</action>",
            "  The informant says that meaning in the language. A transitive verb needs all six "
            "slots; an intransitive verb takes no Object or ObjectNumber. Slots may come in any "
            "order.",
        ]
        if config.enable_judge:
            lines += [
                '<action>judge("sentence in the language")</action>',
                "  The informant says whether the sentence is grammatical.",
            ]
        lines += [
            "<action>done</action>",
            "  End the investigation.",
            f"You have {config.budget} questions. Every action except done uses one question, "
            "including invalid actions. When the questions run out, the investigation ends.",
            'An invalid action gets one of three errors: "format error" (the action cannot be '
            'read), "out of range" (a value is not on the survey sheet), or "type error" (the '
            "slots do not match the verb).",
            "",
            "After the investigation you will translate meanings you have not seen. The language "
            "is written in lowercase Latin letters with words separated by single spaces and no "
            "punctuation.",
        ]
        return "\n".join(lines)

    def step(self, text: str, truncated: bool) -> str:
        """Apply one investigation turn and return the informant's reply.

        The reply ends with the test prompt when the investigation ends. A turn cut off at the token
        limit is a format error (spec 5.4).
        """
        lang = self.language

        def check_meaning(slots: dict[str, str]) -> ParsedAction:
            allowed = {
                "Subject": lang.nouns,
                "SubjectNumber": NUMBERS,
                "Object": lang.nouns,
                "ObjectNumber": NUMBERS,
                "Verb": lang.transitive + lang.intransitive,
                "Tense": TENSES,
            }
            if any(slot in allowed and value not in allowed[slot] for slot, value in slots.items()):
                return ParsedAction("invalid", error=ERROR_RANGE)
            verb = slots.get("Verb")
            if verb is None:
                return ParsedAction("invalid", error=ERROR_TYPE)
            required = TRANSITIVE_SLOTS if verb in lang.transitive else INTRANSITIVE_SLOTS
            if set(slots) != set(required):
                return ParsedAction("invalid", error=ERROR_TYPE)
            meaning = Meaning(
                slots["Subject"],
                slots["SubjectNumber"],
                verb,
                slots["Tense"],
                slots.get("Object"),
                slots.get("ObjectNumber"),
            )
            return ParsedAction("ask", meaning=meaning)

        def parse() -> ParsedAction:
            actions = _ACTION_RE.findall(text)
            if len(actions) != 1:
                return ParsedAction("invalid", error=ERROR_FORMAT)
            body = actions[0].strip()
            if body == "done":
                return ParsedAction("done")
            judge = _JUDGE_RE.match(body)
            if judge:
                if self.config.enable_judge:
                    return ParsedAction("judge", sentence=judge.group(1))
                return ParsedAction("invalid", error=ERROR_FORMAT)
            ask = _ASK_RE.match(body)
            if not ask:
                return ParsedAction("invalid", error=ERROR_FORMAT)
            slots: dict[str, str] = {}
            for pair in ask.group(1).split(","):
                if "=" not in pair:
                    return ParsedAction("invalid", error=ERROR_FORMAT)
                slot, value = (part.strip() for part in pair.split("=", 1))
                if not slot or not value or slot in slots:
                    return ParsedAction("invalid", error=ERROR_FORMAT)
                slots[slot] = value
            return check_meaning(slots)

        def build_test_prompt() -> str:
            lines = [
                "The investigation is over. Translate each meaning below into the language. "
                'Write every answer as <answer id="N">sentence</answer>.',
                "",
            ]
            lines += [f"{i}. {meaning.to_text()}" for i, meaning in enumerate(self.tests, start=1)]
            return "\n".join(lines)

        action = ParsedAction("invalid", error=ERROR_FORMAT) if truncated else parse()
        if action.kind == "done":
            self.in_test_phase = True
            return build_test_prompt()
        self.budget_left -= 1
        if action.kind == "ask":
            self.num_asks += 1
            self.asked.add(action.meaning)
            reply = f"Informant: {lang.realize(action.meaning)}"
        elif action.kind == "judge":
            self.num_judges += 1
            if self._sentences is None:
                self._sentences = {lang.realize(meaning) for meaning in lang.list_meanings()}
            grammatical = normalize(action.sentence) in self._sentences
            reply = f"Informant: {'grammatical' if grammatical else 'ungrammatical'}"
        else:
            self.num_invalid += 1
            reply = action.error
        if self.budget_left <= 0:
            self.in_test_phase = True
            return reply + "\nYou have used all your questions.\n\n" + build_test_prompt()
        plural = "s" if self.budget_left > 1 else ""
        return reply + f"\n({self.budget_left} question{plural} left)"

    def score(self, text: str, truncated: bool) -> EpisodeScore:
        """Score the test answers in text (spec 7).

        The reward is the fraction of scored test items answered exactly; items identical to an
        asked meaning are not scored, and an answer cut off at the token limit scores 0.
        """
        answers: dict[int, str] = {}
        for item_id, answer in _ANSWER_RE.findall(text):
            answers.setdefault(int(item_id), answer)
        scored = {
            item_id: meaning
            for item_id, meaning in enumerate(self.tests, start=1)
            if meaning not in self.asked
        }
        num_correct = 0
        if not truncated:
            num_correct = sum(
                normalize(answers.get(item_id, "")) == self.language.realize(meaning)
                for item_id, meaning in scored.items()
            )
        return EpisodeScore(
            reward=num_correct / len(scored) if scored else 0.0,
            num_asks=self.num_asks,
            num_judges=self.num_judges,
            num_invalid=self.num_invalid,
            num_scorable=len(scored),
            num_correct=num_correct,
            test_truncated=int(truncated),
        )
