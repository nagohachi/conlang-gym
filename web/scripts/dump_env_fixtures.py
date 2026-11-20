"""Dump reference outputs of the Python environment for the TypeScript port's parity tests.

Run from the repository root:
    PYTHONPATH=. python web/scripts/dump_env_fixtures.py
It writes web/tests/fixtures/*.json, which web/tests/env.test.ts compares with web/src/env/ for
exact equality. Floats are written with repr, which JSON.parse reads back to the same double.
"""

import dataclasses
import hashlib
import json
import logging
import math
import random
import struct
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path

from conlang_gym import language as language_module
from conlang_gym.language import (
    NOUN_POOL,
    OPTIONAL_PARAMS,
    TRANSITIVE_POOL,
    EnvConfig,
    Episode,
    Grammar,
    Language,
    Meaning,
    generate,
)
from conlang_gym.solver import ALL_GRAMMARS, HypothesisSolver, solve

logger = logging.getLogger(__name__)

type Json = dict[str, Json] | list[Json] | str | int | float | bool | None

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
MAX_SAFE_INTEGER = 2**53 - 1
# The largest number of hypotheses the solver can have: len(ALL_GRAMMARS).
MAX_HYPOTHESES = len(ALL_GRAMMARS)


def seed_to_json(seed: int) -> int | str:
    """Return seed as a JSON number when JavaScript can hold it exactly, else as a string."""
    return seed if abs(seed) <= MAX_SAFE_INTEGER else str(seed)


def meaning_to_json(meaning: Meaning) -> list[str | None]:
    """Return a meaning as [subject, subject_number, verb, tense, object, object_number]."""
    return list(meaning)


def config_to_json(config: EnvConfig) -> dict[str, int | bool]:
    """Return the config with Python's field names."""
    return dataclasses.asdict(config)


def grammar_label(grammar: Grammar) -> str:
    """Return make_dataset.py's build_grammar_label(grammar) (make_dataset imports pandas)."""
    active = [
        f"{p}-{getattr(grammar, p)}" for p in OPTIONAL_PARAMS if getattr(grammar, p) != "none"
    ]
    parts = [grammar.word_order, *active]
    if grammar.object_marking == "suffix" and grammar.plural == "suffix":
        parts.append(grammar.plural_case_order)
    return "_".join(parts)


def sha256_lines(lines: list[str]) -> str:
    """Return the SHA-256 of the lines joined by newlines, encoded as UTF-8."""
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def write_fixture(name: str, data: Json) -> None:
    """Write data to OUT_DIR/name as compact JSON."""
    path = OUT_DIR / name
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    logger.info("wrote %s (%d bytes)", path, path.stat().st_size)


# ------------------------------------------------------------------------------------------------
# PyRandom


def dump_pyrandom() -> Json:
    """Record the results of a fixed sequence of random.Random calls for several seeds."""
    seeds = [
        0,
        1,
        2,
        42,
        7919 * 5 + 1,
        2**31 - 1,
        2**32 - 1,
        2**32,
        2**32 + 1,
        123456789 * 7919 + 1,
        2**53 + 1,
        2**64 - 1,
        2**64,
        10**30 + 7,
        -1,
        -42,
    ]
    sample_cases = [
        (0, 0),
        (1, 1),
        (5, 0),
        (5, 3),
        (5, 5),
        (6, 6),
        (10, 4),
        (15, 15),
        (21, 5),
        (22, 5),
        (22, 6),
        (85, 7),
        (86, 7),
        (85, 21),
        (86, 21),
        (100, 6),
        (272, 4),
        (264, 48),
        (277, 22),
        (278, 22),
        (277, 48),
        (278, 48),
        (300, 48),
        (1000, 10),
        (50, 50),
        (500, 100),
    ]
    randbelow_cases = [1, 2, 3, 5, 6, 7, 8, 15, 16, 17, 100, 2**31 - 1, 2**31, 2**32 - 1, 2**32]
    randbelow_cases += [2**32 + 1, 2**40 + 3, 2**53 - 1]
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        ops: list[Json] = []
        for k in [*range(0, 33), 33, 40, 52, 53]:
            ops.append(["getrandbits", k, rng.getrandbits(k)])
        for k in (64, 100, 200):
            ops.append(["getrandbitsBig", k, str(rng.getrandbits(k))])
        for _ in range(5):
            ops.append(["random", rng.random()])
        for n in randbelow_cases:
            ops.append(["randbelow", n, rng._randbelow(n)])
        for n in (1, 2, 3, 6, 10, 15, 100):
            ops.append(["choice", n, rng.choice(range(n))])
        for n, k in sample_cases:
            ops.append(["sample", n, k, rng.sample(range(n), k)])
        for n in (0, 1, 2, 8, 50, 300):
            items = list(range(n))
            rng.shuffle(items)
            ops.append(["shuffle", n, items])
        ops.append(["getrandbits", 32, rng.getrandbits(32)])
        results.append({"seed": seed_to_json(seed), "ops": ops})
    return results


# ------------------------------------------------------------------------------------------------
# Languages


def describe_language(seed: int, config: EnvConfig) -> Json:
    """Return everything generate() and Episode.build_initial_prompt() give for seed."""
    episode = Episode(seed, config)
    lang = episode.language
    meanings = lang.list_meanings()
    return {
        "seed": seed_to_json(seed),
        "grammar": dataclasses.asdict(lang.grammar),
        "label": grammar_label(lang.grammar),
        "nouns": list(lang.nouns),
        "transitive": list(lang.transitive),
        "intransitive": list(lang.intransitive),
        "roots": [[concept, form] for concept, form in lang.roots.items()],
        "affixes": [[name, form] for name, form in lang.affixes.items()],
        "tests": [
            [meaning_to_json(m), lang.realize(m), m.to_text(), m.is_compositional()]
            for m in episode.tests
        ],
        "num_meanings": len(meanings),
        "meanings_sha256": sha256_lines(
            [f"{m.to_text()}\t{lang.realize(m)}\t{int(m.is_compositional())}" for m in meanings]
        ),
        "prompt": episode.build_initial_prompt(),
    }


GENERATE_CONFIGS: list[tuple[str, dict[str, int | bool], list[int]]] = [
    ("default", {}, [*range(60), 2**32 + 5, 10**12 + 7, 2**64 + 1, 123456789]),
    ("nap0", {"num_active_params": 0}, list(range(25))),
    ("nap2", {"num_active_params": 2}, list(range(25))),
    ("nap3", {"num_active_params": 3}, list(range(25))),
    ("nap4", {"num_active_params": 4}, list(range(25))),
    ("nap5", {"num_active_params": 5}, list(range(25))),
    ("no_dictionary", {"show_dictionary": False}, list(range(15))),
    ("judge", {"enable_judge": True}, list(range(15))),
    ("judge_no_dictionary", {"enable_judge": True, "show_dictionary": False}, list(range(5))),
    ("max_vocab", {"num_nouns": 10, "num_transitive": 6, "num_intransitive": 4}, list(range(10))),
    ("min_vocab", {"num_nouns": 2, "num_transitive": 1, "num_intransitive": 1}, list(range(15))),
    (
        "mid_vocab",
        {"num_nouns": 7, "num_transitive": 3, "num_intransitive": 2, "num_active_params": 3},
        list(range(10)),
    ),
    ("budget1", {"budget": 1}, list(range(5))),
    ("budget12", {"budget": 12}, list(range(5))),
    ("test20_comp10", {"num_test": 20, "num_compositional": 10}, list(range(10))),
    ("test60_comp50", {"num_test": 60, "num_compositional": 50}, list(range(5))),
    ("test3_comp0", {"num_test": 3, "num_compositional": 0}, list(range(10))),
    ("test1_comp1", {"num_test": 1, "num_compositional": 1}, list(range(10))),
]


def dump_generate() -> Json:
    """Record languages, test sets and prompts under several configs, plus config errors."""
    configs = []
    for name, overrides, seeds in GENERATE_CONFIGS:
        config = EnvConfig(**overrides)
        configs.append(
            {
                "name": name,
                "config": config_to_json(config),
                "cases": [describe_language(seed, config) for seed in seeds],
            }
        )

    def error_of(action: Callable[[], object]) -> str:
        """Return the message of the ValueError that action raises."""
        try:
            action()
        except ValueError as error:
            return str(error)
        raise AssertionError("expected a ValueError")

    invalid_configs = [
        {"num_nouns": 1},
        {"num_nouns": 11},
        {"num_transitive": 0},
        {"num_transitive": 7},
        {"num_intransitive": 0},
        {"num_intransitive": 5},
        {"num_active_params": -1},
        {"num_active_params": 6},
        {"budget": 0},
        {"num_test": 0},
        {"num_compositional": -1},
        {"num_compositional": 9},
        {"num_test": 3, "num_compositional": 4},
    ]
    config_errors = [
        {"config": overrides, "error": error_of(lambda o=overrides: EnvConfig(**o))}
        for overrides in invalid_configs
    ]
    # Valid configs that ask for more test items than the survey sheet has.
    impossible = [
        {
            "num_nouns": 2,
            "num_transitive": 1,
            "num_intransitive": 1,
            "num_test": 20,
            "num_compositional": 17,
        },
        {"num_nouns": 2, "num_transitive": 1, "num_intransitive": 1, "num_test": 60},
    ]
    generate_errors = [
        {
            "config": config_to_json(EnvConfig(**overrides)),
            "error": error_of(lambda o=overrides: generate(0, EnvConfig(**o))),
        }
        for overrides in impossible
    ]
    return {
        "configs": configs,
        "config_errors": config_errors,
        "generate_errors": generate_errors,
    }


def dump_grammars() -> Json:
    """Record ALL_GRAMMARS, their labels and the realizations of a few meanings under each."""
    lang, _ = generate(0, EnvConfig(num_active_params=0))
    n1, n2 = lang.nouns[0], lang.nouns[1]
    t, i = lang.transitive[0], lang.intransitive[0]
    meanings = [
        Meaning(n1, "plural", t, "past", n2, "plural"),
        Meaning(n1, "singular", t, "present", n2, "singular"),
        Meaning(n2, "plural", t, "present", n1, "singular"),
        Meaning(n2, "singular", t, "past", n1, "plural"),
        Meaning(n1, "plural", i, "past"),
        Meaning(n2, "singular", i, "present"),
    ]
    rows = []
    for grammar in ALL_GRAMMARS:
        variant = Language(
            grammar, lang.nouns, lang.transitive, lang.intransitive, lang.roots, lang.affixes
        )
        rows.append(
            {
                "grammar": dataclasses.asdict(grammar),
                "label": grammar_label(grammar),
                "has_affix": grammar.has_affix(),
                "realizations": [variant.realize(m) for m in meanings],
            }
        )
    return {
        "roots": [[concept, form] for concept, form in lang.roots.items()],
        "affixes": [[name, form] for name, form in lang.affixes.items()],
        "nouns": list(lang.nouns),
        "transitive": list(lang.transitive),
        "intransitive": list(lang.intransitive),
        "meanings": [meaning_to_json(m) for m in meanings],
        "grammars": rows,
    }


# ------------------------------------------------------------------------------------------------
# Episodes


def build_action_texts(episode: Episode, rng: random.Random) -> dict[str, str]:
    """Return named investigation turns that cover every reply of the informant."""
    lang = episode.language
    n0, n1 = lang.nouns[0], lang.nouns[1]
    t0, i0 = lang.transitive[0], lang.intransitive[0]
    outside_noun = next((noun for noun in NOUN_POOL if noun not in lang.nouns), "dragon")
    outside_verb = next((verb for verb in TRANSITIVE_POOL if verb not in lang.transitive), "fly")
    test = episode.tests[0]
    random_meaning = rng.choice(lang.list_meanings())
    random_slots = random_meaning.to_text().split(", ")
    rng.shuffle(random_slots)
    sentence = lang.realize(random_meaning)
    shouted = "-".join(sentence.upper())
    full = f"Subject={n0}, SubjectNumber=plural, Object={n1}, ObjectNumber=singular, Verb={t0}"
    intransitive = f"Subject={n0}, SubjectNumber=singular, Verb={i0}, Tense=past"
    return {
        "ask_transitive": f"<action>ask({full}, Tense=past)</action>",
        "ask_shuffled": "Let me ask.\n<action>ask(Verb="
        + t0
        + f", Tense=present, ObjectNumber=plural, Object={n0}, SubjectNumber=singular, "
        f"Subject={n1})</action>\nThat is my question.",
        "ask_spaces": f"<action> ask( Subject = {n1} ,SubjectNumber= singular , Verb={i0},"
        "Tense=past ) </action>",
        "ask_newlines": f"<action>\nask(Subject={n0},SubjectNumber=plural,Verb={i0},"
        "Tense=present)\n</action>",
        "ask_test_item": f"<action>ask({test.to_text()})</action>",
        "ask_random": f"<action>ask({', '.join(random_slots)})</action>",
        "ask_unicode_space": f"<action>\u00a0ask({intransitive})\u3000</action>",
        "ask_python_space": f"<action>\x1cask(Subject=\x1d{n0}\x85, SubjectNumber=singular, "
        f"Verb={i0}, Tense=past)\x1f</action>",
        "ask_bom": f"<action>\ufeffask({intransitive})</action>",
        "ask_bom_value": f"<action>ask(Subject={n0}\ufeff, SubjectNumber=singular, Verb={i0}, "
        "Tense=past)</action>",
        "ask_em_space": f"<action>ask(Subject = {n0}\u2003, SubjectNumber=singular, Verb={i0}, "
        "Tense=past)</action>",
        "no_action": f"ask({intransitive})",
        "two_actions": "<action>done</action> <action>done</action>",
        "unclosed": f"<action>ask({intransitive})",
        "empty_ask": "<action>ask()</action>",
        "trailing_comma": f"<action>ask({intransitive},)</action>",
        "duplicate_slot": f"<action>ask(Subject={n0}, Subject={n1}, SubjectNumber=plural, "
        f"Verb={i0}, Tense=past)</action>",
        "empty_value": f"<action>ask(Subject=, SubjectNumber=plural, Verb={i0}, "
        "Tense=past)</action>",
        "empty_slot": f"<action>ask(={n0}, SubjectNumber=plural, Verb={i0}, Tense=past)</action>",
        "no_parens": f"<action>ask Subject={n0}</action>",
        "upper_done": "<action>DONE</action>",
        "done_period": "<action>done.</action>",
        "capital_ask": f"<action>Ask({intransitive})</action>",
        "empty_action": "<action></action>",
        "nested_action": "<action><action>done</action></action>",
        "missing_comma": f"<action>ask(Subject={n0} SubjectNumber=plural, Verb={i0}, "
        "Tense=past)</action>",
        "double_parens": f"<action>ask(({intransitive}))</action>",
        "value_with_equals": f"<action>ask(Subject={n0}={n1}, SubjectNumber=plural, Verb={i0}, "
        "Tense=past)</action>",
        "range_subject": f"<action>ask(Subject=unicorn, SubjectNumber=plural, Verb={i0}, "
        "Tense=past)</action>",
        "range_number": f"<action>ask(Subject={n0}, SubjectNumber=dual, Verb={i0}, "
        "Tense=past)</action>",
        "range_verb": f"<action>ask(Subject={n0}, SubjectNumber=plural, Verb=fly, "
        "Tense=past)</action>",
        "range_tense": f"<action>ask(Subject={n0}, SubjectNumber=plural, Verb={i0}, "
        "Tense=future)</action>",
        "range_object": f"<action>ask(Subject={n0}, SubjectNumber=plural, Object=unicorn, "
        f"ObjectNumber=plural, Verb={t0}, Tense=past)</action>",
        "range_pool_noun": f"<action>ask(Subject={outside_noun}, SubjectNumber=plural, "
        f"Verb={i0}, Tense=past)</action>",
        "range_pool_verb": f"<action>ask(Subject={n0}, SubjectNumber=plural, Object={n1}, "
        f"ObjectNumber=plural, Verb={outside_verb}, Tense=past)</action>",
        "range_case": f"<action>ask(Subject={n0.capitalize()}, SubjectNumber=plural, "
        f"Verb={i0}, Tense=past)</action>",
        "type_intransitive_object": f"<action>ask(Subject={n0}, SubjectNumber=plural, "
        f"Object={n1}, ObjectNumber=plural, Verb={i0}, Tense=past)</action>",
        "type_missing_object": f"<action>ask(Subject={n0}, SubjectNumber=plural, Verb={t0}, "
        "Tense=past)</action>",
        "type_missing_object_number": f"<action>ask(Subject={n0}, SubjectNumber=plural, "
        f"Object={n1}, Verb={t0}, Tense=past)</action>",
        "type_missing_verb": f"<action>ask(Subject={n0}, SubjectNumber=plural, "
        "Tense=past)</action>",
        "type_unknown_slot": f"<action>ask({intransitive}, Mood=indicative)</action>",
        "type_missing_tense": f"<action>ask(Subject={n0}, SubjectNumber=plural, "
        f"Verb={i0})</action>",
        "judge_exact": f'<action>judge("{sentence}")</action>',
        "judge_shouted": f'<action>judge(  "  {shouted}\t " )</action>',
        "judge_wrong": '<action>judge("ba ba ba")</action>',
        "judge_empty": '<action>judge("")</action>',
        "judge_single_quotes": f"<action>judge('{sentence}')</action>",
        "judge_inner_quotes": f'<action>judge("{sentence} "x" ")</action>',
        "judge_unicode_space": f'<action>judge("{sentence.replace(" ", "\u2003 ")}")</action>',
        "judge_newline": f'<action>judge("{sentence}\n")</action>',
        "judge_extra_word": f'<action>judge("{sentence} {sentence}")</action>',
        "done": "<action>done</action>",
        "done_spaces": "I am done.\n<action>  done \n</action>",
    }


def run_episode(
    seed: int, config: EnvConfig, script: list[tuple[str, bool]], rng: random.Random
) -> Json:
    """Play the named turns of script and record every reply, parsed action and counter."""
    episode = Episode(seed, config)
    texts = build_action_texts(episode, rng)

    class RecordingSet(set[Meaning]):
        """A set that remembers the last item added (the meaning of an ask)."""

        last: Meaning | None = None

        def add(self, item: Meaning) -> None:
            """Add item and remember it."""
            self.last = item
            super().add(item)

    episode.asked = RecordingSet()
    judged: list[str] = []
    original_normalize = language_module.normalize

    def recording_normalize(text: str) -> str:
        """Remember the sentence that a judge action normalizes."""
        judged.append(text)
        return original_normalize(text)

    steps = []
    language_module.normalize = recording_normalize
    try:
        for name, truncated in script:
            before = (episode.num_asks, episode.num_judges, episode.num_invalid)
            budget_before = episode.budget_left
            judged.clear()
            reply = episode.step(texts[name], truncated)
            if episode.num_asks > before[0]:
                action: Json = ["ask", meaning_to_json(episode.asked.last)]
            elif episode.num_judges > before[1]:
                action = ["judge", judged[0]]
            elif episode.num_invalid > before[2]:
                action = ["invalid", reply.split("\n")[0]]
            else:
                assert episode.budget_left == budget_before and episode.in_test_phase
                action = ["done"]
            steps.append(
                {
                    "name": name,
                    "text": texts[name],
                    "truncated": truncated,
                    "reply": reply,
                    "action": action,
                    "state": [
                        episode.budget_left,
                        episode.num_asks,
                        episode.num_judges,
                        episode.num_invalid,
                        episode.in_test_phase,
                        len(episode.asked),
                    ],
                }
            )
    finally:
        language_module.normalize = original_normalize

    scores = []
    for text, truncated in build_answer_texts(episode, rng):
        scores.append(score_record(episode, text, truncated))
    return {
        "seed": seed_to_json(seed),
        "config": config_to_json(config),
        "steps": steps,
        "scores": scores,
    }


def score_record(episode: Episode, text: str, truncated: bool) -> Json:
    """Return Episode.score() of text and, per test item, what score() compares.

    Each item is [answer or None, scorable, correct], computed with score()'s own regex and
    normalize(), for the TypeScript Episode.grade().
    """
    answers: dict[int, str] = {}
    for item_id, answer in language_module._ANSWER_RE.findall(text):
        answers.setdefault(int(item_id), answer)
    items: list[Json] = []
    for item_id, meaning in enumerate(episode.tests, start=1):
        scorable = meaning not in episode.asked
        answer = answers.get(item_id)
        expected = episode.language.realize(meaning)
        correct = scorable and not truncated and language_module.normalize(answer or "") == expected
        items.append([answer, scorable, correct])
    score = episode.score(text, truncated)
    assert score.num_correct == sum(item[2] for item in items)
    return {
        "text": text,
        "truncated": truncated,
        "score": dataclasses.asdict(score),
        "items": items,
    }


def build_answer_texts(episode: Episode, rng: random.Random) -> list[tuple[str, bool]]:
    """Return test answers that are right, wrong, missing, repeated or oddly written."""
    lang = episode.language
    right = [lang.realize(meaning) for meaning in episode.tests]

    def answers(items: list[tuple[str, str]]) -> str:
        """Format (id, sentence) pairs as answer tags, one per line."""
        return "\n".join(
            f'<answer id="{item_id}">{sentence}</answer>' for item_id, sentence in items
        )

    def mangle(sentence: str) -> str:
        """Write sentence with capitals, hyphens and odd whitespace that normalize() undoes."""
        words = ["-".join(word[:2].upper() + word[2:]) for word in sentence.split(" ")]
        return " \t" + "  \n ".join(words) + "\u3000"

    def wrong(sentence: str) -> str:
        """Return sentence with its words reversed (still wrong if it has one word)."""
        words = sentence.split(" ")
        return " ".join(reversed(words)) + ("" if len(words) > 1 else "x")

    ids = [str(i) for i in range(1, len(right) + 1)]
    texts = [
        answers(list(zip(ids, right, strict=True))),
        "Here are my answers:\n" + answers(list(zip(ids, map(mangle, right), strict=True))),
        answers([(i, s if int(i) % 2 else wrong(s)) for i, s in zip(ids, right, strict=True)]),
        answers([(i, s) for i, s in zip(ids, right, strict=True) if int(i) % 3]),
        answers(
            [(ids[0], wrong(right[0])), (ids[0], right[0]), *zip(ids[1:], right[1:], strict=True)]
        ),
        answers(
            [(ids[0], right[0]), (ids[0], wrong(right[0])), *zip(ids[1:], right[1:], strict=True)]
        ),
        answers([("0" + ids[0], right[0]), ("0", right[0]), ("99", right[-1])]),
        answers([("\u0661", right[0]), ("\uff12", right[min(1, len(right) - 1)])]),
        answers([("\U0001d7cf", right[0]), ("\u0967" + "\u0966", right[-1])]),
        answers(list(reversed(list(zip(ids, right, strict=True))))),
        f'<answer id="1" >{right[0]}</answer><answer id="1">{right[0]}',
        f'<answer id="1">x<answer id="2">{right[min(1, len(right) - 1)]}</answer>',
        "",
    ]
    shuffled = list(zip(ids, right, strict=True))
    rng.shuffle(shuffled)
    texts.append(answers(shuffled))
    return [(text, False) for text in texts] + [(texts[0], True), ("", True)]


EVERY_TURN = [
    "ask_transitive",
    "ask_shuffled",
    "ask_spaces",
    "ask_newlines",
    "ask_test_item",
    "ask_random",
    "ask_unicode_space",
    "ask_python_space",
    "ask_bom",
    "ask_bom_value",
    "ask_em_space",
    "no_action",
    "two_actions",
    "unclosed",
    "empty_ask",
    "trailing_comma",
    "duplicate_slot",
    "empty_value",
    "empty_slot",
    "no_parens",
    "upper_done",
    "done_period",
    "capital_ask",
    "empty_action",
    "nested_action",
    "missing_comma",
    "double_parens",
    "value_with_equals",
    "range_subject",
    "range_number",
    "range_verb",
    "range_tense",
    "range_object",
    "range_pool_noun",
    "range_pool_verb",
    "range_case",
    "type_intransitive_object",
    "type_missing_object",
    "type_missing_object_number",
    "type_missing_verb",
    "type_unknown_slot",
    "type_missing_tense",
    "judge_exact",
    "judge_shouted",
    "judge_wrong",
    "judge_empty",
    "judge_single_quotes",
    "judge_inner_quotes",
    "judge_unicode_space",
    "judge_newline",
    "judge_extra_word",
]


def dump_episodes() -> Json:
    """Record scripted episodes: every kind of turn, done, budget exhaustion and scoring."""
    default = EnvConfig()
    judge = EnvConfig(enable_judge=True)
    plain = [(name, False) for name in EVERY_TURN]
    scripts: list[tuple[list[int], EnvConfig, list[tuple[str, bool]]]] = [
        # Budget exhaustion after a mix of turns.
        (
            list(range(6)),
            default,
            [
                ("ask_transitive", False),
                ("no_action", False),
                ("range_subject", False),
                ("type_intransitive_object", False),
                ("ask_shuffled", False),
                ("ask_spaces", False),
            ],
        ),
        # Done after asking a test item, then a turn after done.
        (
            list(range(6, 10)),
            default,
            [
                ("ask_test_item", False),
                ("ask_transitive", False),
                ("done", False),
                ("ask_newlines", False),
            ],
        ),
        # A truncated turn is a format error whatever it says; done with whitespace.
        (
            list(range(10, 13)),
            default,
            [("ask_transitive", True), ("done", True), ("done_spaces", False)],
        ),
        # Every kind of turn with the judge disabled and enabled; the budget runs out at the end.
        ([13, 14], EnvConfig(budget=len(plain)), plain),
        ([15, 16], EnvConfig(budget=len(plain), enable_judge=True), plain),
        ([17], EnvConfig(budget=len(plain) + 1, enable_judge=True), [*plain, ("done", False)]),
        # Judges only.
        (
            [18, 19],
            judge,
            [
                ("judge_exact", False),
                ("judge_shouted", False),
                ("judge_wrong", False),
                ("judge_unicode_space", False),
                ("judge_newline", False),
                ("judge_inner_quotes", False),
            ],
        ),
        # One question: the first turn ends the investigation; then two questions.
        ([20, 21], EnvConfig(budget=1), [("ask_random", False)]),
        ([22], EnvConfig(budget=1), [("done", False)]),
        ([23, 24], EnvConfig(budget=2), [("ask_random", False), ("judge_exact", False)]),
        # Other configs.
        (
            [25, 26],
            EnvConfig(num_active_params=3, show_dictionary=False),
            [("ask_random", False), ("ask_shuffled", False), ("done", False)],
        ),
        (
            [27],
            EnvConfig(num_nouns=10, num_transitive=6, num_intransitive=4, num_active_params=5),
            [("ask_random", False), ("ask_transitive", False), ("ask_random", False)],
        ),
        (
            [28, 2**40 + 3],
            EnvConfig(num_test=3, num_compositional=0, budget=3),
            [("ask_test_item", False), ("ask_random", False), ("ask_random", False)],
        ),
    ]
    episodes = []
    rng = random.Random(2024)
    for seeds, config, script in scripts:
        for seed in seeds:
            episodes.append(run_episode(seed, config, script, rng))

    # Ask every test item, so that no item is scored.
    for seed in (30, 31):
        config = EnvConfig(budget=8, num_test=4, num_compositional=2)
        tests = Episode(seed, config).tests
        texts = [f"<action>ask({meaning.to_text()})</action>" for meaning in tests]
        episodes.append(run_episode_texts(seed, config, texts))
    return episodes


def run_episode_texts(seed: int, config: EnvConfig, texts: list[str]) -> Json:
    """Play literal turns and record replies, counters and the scores of the right answers."""
    episode = Episode(seed, config)
    steps = []
    for text in texts:
        reply = episode.step(text, False)
        steps.append(
            {
                "name": "literal",
                "text": text,
                "truncated": False,
                "reply": reply,
                "action": None,
                "state": [
                    episode.budget_left,
                    episode.num_asks,
                    episode.num_judges,
                    episode.num_invalid,
                    episode.in_test_phase,
                    len(episode.asked),
                ],
            }
        )
    right = "".join(
        f'<answer id="{i}">{episode.language.realize(m)}</answer>'
        for i, m in enumerate(episode.tests, start=1)
    )
    return {
        "seed": seed_to_json(seed),
        "config": config_to_json(config),
        "steps": steps,
        "scores": [score_record(episode, right, False)],
    }


# ------------------------------------------------------------------------------------------------
# Solver and the floating-point operations it depends on


def correctly_rounded_log(x: float) -> float:
    """Return log(x) correctly rounded to a double, computed with 60 significant digits."""
    with localcontext() as context:
        context.prec = 60
        return float(Decimal(x).ln())


def dump_pymath() -> Json:
    """Record math.log on every count / num_hypotheses and sum() on a few float lists.

    The C library's log is not always correctly rounded; the inputs where this platform's log
    differs from the correctly rounded value are listed so that the tests can tell both apart.
    """
    digest = hashlib.sha256()
    exceptions = []
    num_inputs = 0
    for num_hypotheses in range(1, MAX_HYPOTHESES + 1):
        for count in range(1, num_hypotheses + 1):
            x = count / num_hypotheses
            value = math.log(x)
            digest.update(struct.pack("<d", value))
            num_inputs += 1
            if count != num_hypotheses and value != correctly_rounded_log(x):
                exceptions.append([count, num_hypotheses, value, correctly_rounded_log(x)])
    rng = random.Random(7)
    lists: list[list[float]] = [
        [],
        [0.1] * 10,
        [1e16, 1.0, -1e16],
        [1.0, 1e100, 1.0, -1e100],
        [-0.0],
        [0.0, -0.0],
        [2.0**-1074, 1.0, -1.0],
    ]
    for size in (1, 2, 3, 5, 8, 20, 50):
        for _ in range(20):
            lists.append([rng.uniform(-1, 0) * 10 ** rng.randint(-20, 5) for _ in range(size)])
    for _ in range(50):
        counts = [rng.randint(1, 60) for _ in range(rng.randint(1, 12))]
        total = sum(counts)
        lists.append([c / total * math.log(c / total) for c in counts])
    return {
        "max_hypotheses": MAX_HYPOTHESES,
        "num_log_inputs": num_inputs,
        "log_sha256": digest.hexdigest(),
        "log_exceptions": exceptions,
        "sums": [[values, sum(values)] for values in lists],
    }


SOLVER_CONFIGS: list[tuple[str, dict[str, int | bool], int, int]] = [
    # name, EnvConfig overrides, number of seeds, num_candidates
    ("nap0", {"num_active_params": 0}, 300, 48),
    ("nap1", {"num_active_params": 1}, 300, 48),
    ("nap2", {"num_active_params": 2}, 300, 48),
    ("no_dictionary", {"show_dictionary": False}, 60, 48),
    ("nap2_budget2", {"num_active_params": 2, "budget": 2}, 60, 48),
    ("nap2_candidates8", {"num_active_params": 2}, 40, 8),
    ("nap1_candidates_all", {}, 20, 1000),
]
NUM_TRACED_SEEDS = 12


def solve_case(case: tuple[int, dict[str, int | bool], int]) -> int | None:
    """Run solve() on one (seed, overrides, num_candidates); module-level for the worker pool."""
    seed, overrides, num_candidates = case
    return solve(seed, EnvConfig(**overrides), num_candidates)


def trace_solve(seed: int, config: EnvConfig, num_candidates: int) -> Json:
    """Run solve() and record every candidate's entropy and the question chosen each turn."""
    turns: list[Json] = []
    original = HypothesisSolver.choose_question

    def recording_choose(self: HypothesisSolver, candidates: list[Meaning]) -> Meaning:
        """Compute the entropies as choose_question does and check that it picks the same."""
        num_hypotheses = len(self.hypotheses)
        entropies = []
        for meaning in candidates:
            counts = Counter(self.predict(meaning, with_placeholders=True))
            entropies.append(
                -sum(
                    count / num_hypotheses * math.log(count / num_hypotheses)
                    for count in counts.values()
                )
            )
        chosen = original(self, candidates)
        best = max(range(len(candidates)), key=lambda index: (entropies[index], -index))
        assert candidates[best] == chosen
        turns.append(
            {
                "num_hypotheses": num_hypotheses,
                "entropies": entropies,
                "question": meaning_to_json(chosen),
            }
        )
        return chosen

    HypothesisSolver.choose_question = recording_choose
    try:
        result = solve(seed, config, num_candidates)
    finally:
        HypothesisSolver.choose_question = original
    return {"seed": seed, "result": result, "turns": turns}


def dump_solver(pool: ProcessPoolExecutor) -> Json:
    """Record solve() for hundreds of seeds and the entropy traces of a few."""
    runs = []
    for name, overrides, num_seeds, num_candidates in SOLVER_CONFIGS:
        cases = [(seed, overrides, num_candidates) for seed in range(num_seeds)]
        results = list(pool.map(solve_case, cases, chunksize=10))
        config = EnvConfig(**overrides)
        traces = [
            trace_solve(seed, config, num_candidates)
            for seed in range(min(NUM_TRACED_SEEDS, num_seeds))
        ]
        for trace in traces:
            assert trace["result"] == results[trace["seed"]]
        runs.append(
            {
                "name": name,
                "config": config_to_json(config),
                "num_candidates": num_candidates,
                "results": results,
                "traces": traces,
            }
        )
        logger.info("solver %s: %d seeds", name, num_seeds)
    return runs


def main() -> None:
    """Write every fixture to web/tests/fixtures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_fixture("pyrandom.json", dump_pyrandom())
    write_fixture("generate.json", dump_generate())
    write_fixture("grammars.json", dump_grammars())
    write_fixture("episodes.json", dump_episodes())
    write_fixture("pymath.json", dump_pymath())
    with ProcessPoolExecutor() as pool:
        write_fixture("solver.json", dump_solver(pool))


if __name__ == "__main__":
    main()
