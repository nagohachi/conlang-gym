"""Hypothesis solver that decides whether a seed can be solved within the question budget (spec 8).

A hypothesis is a grammar. Morpheme forms are not part of it: they are read off the observations
by position and kept as the hypothesis's bindings, and a hypothesis is dropped when no binding fits
every observation. Questions are chosen greedily by the entropy of the predicted observation.
Reaching "every remaining hypothesis gives the same answer to every test item" within the budget is
a sufficient condition for solvability; seeds that need a cleverer strategy are discarded.
"""

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from .language import (
    PARAM_CHOICES,
    EnvConfig,
    Grammar,
    Meaning,
    Morpheme,
    Template,
    build_sentence_template,
    generate,
)

ALL_GRAMMARS: tuple[Grammar, ...] = tuple(
    sorted(
        {Grammar(*combo).to_canonical() for combo in itertools.product(*PARAM_CHOICES.values())},
        key=lambda grammar: tuple(vars(grammar).values()),
    )
)

_build_template_cached = lru_cache(maxsize=500_000)(build_sentence_template)


@dataclass
class Hypothesis:
    """A grammar that fits every observation so far, with the morpheme forms they fix.

    Under vowel harmony an affix binding holds only the consonant; the vowel copies the root's.
    """

    grammar: Grammar
    bindings: dict[Morpheme, str] = field(default_factory=dict)


class HypothesisSolver:
    """The grammars that are consistent with the observations of one language."""

    def __init__(self, known_roots: dict[str, str] | None) -> None:
        """Start from every canonical grammar; known_roots is None if the dictionary is hidden."""
        self.known_roots = known_roots
        self.hypotheses = [Hypothesis(grammar) for grammar in ALL_GRAMMARS]

    def observe(self, meaning: Meaning, sentence: str) -> None:
        """Drop the hypotheses that cannot produce sentence for meaning and extend the others."""
        words = sentence.split(" ")

        def match(template: Template, harmony: str, bindings: dict[Morpheme, str]) -> bool:
            if len(words) != len(template):
                return False
            for word_template, word in zip(template, words, strict=True):
                if len(word) != sum(4 if morpheme.is_root else 2 for morpheme in word_template):
                    return False
                pieces, position = [], 0
                for morpheme in word_template:
                    length = 4 if morpheme.is_root else 2
                    pieces.append(word[position : position + length])
                    position += length
                root_piece = next(
                    piece
                    for morpheme, piece in zip(word_template, pieces, strict=True)
                    if morpheme.is_root
                )
                for morpheme, piece in zip(word_template, pieces, strict=True):
                    if morpheme.is_root and self.known_roots is not None:
                        expected = self.known_roots[morpheme.name]
                    elif not morpheme.is_root and harmony == "copy":
                        if piece[1] != root_piece[-1]:
                            return False
                        piece = piece[0]
                        expected = bindings.get(morpheme)
                    else:
                        expected = bindings.get(morpheme)
                    if expected is None:
                        bindings[morpheme] = piece
                    elif expected != piece:
                        return False
            # Every morpheme has its own form (spec 3.2).
            for is_root in (False, True):
                forms = [form for slot, form in bindings.items() if slot.is_root == is_root]
                if len(forms) != len(set(forms)):
                    return False
            return True

        survivors = []
        for hypothesis in self.hypotheses:
            bindings = dict(hypothesis.bindings)
            template = _build_template_cached(hypothesis.grammar, meaning)
            if match(template, hypothesis.grammar.vowel_harmony, bindings):
                survivors.append(Hypothesis(hypothesis.grammar, bindings))
        self.hypotheses = survivors

    def predict(self, meaning: Meaning, with_placeholders: bool) -> list[str | None]:
        """Return each hypothesis's sentence for meaning.

        A form that no observation has fixed yet becomes <name> with_placeholders, and makes the
        prediction None otherwise.
        """

        def render(template: Template, harmony: str, bindings: dict[Morpheme, str]) -> str | None:
            words = []
            for word_template in template:
                root_name = next(morpheme.name for morpheme in word_template if morpheme.is_root)
                if self.known_roots is not None:
                    root = self.known_roots[root_name]
                else:
                    root = bindings.get(Morpheme(True, root_name))
                pieces = []
                for morpheme in word_template:
                    if morpheme.is_root:
                        form = root
                    else:
                        form = bindings.get(morpheme)
                        if form is not None and harmony == "copy":
                            form = form + root[-1] if root is not None else None
                    if form is None:
                        if not with_placeholders:
                            return None
                        form = f"<{morpheme.name}>"
                    pieces.append(form)
                words.append("".join(pieces))
            return " ".join(words)

        return [
            render(
                _build_template_cached(hypothesis.grammar, meaning),
                hypothesis.grammar.vowel_harmony,
                hypothesis.bindings,
            )
            for hypothesis in self.hypotheses
        ]

    def is_solved(self, tests: list[Meaning]) -> bool:
        """Return whether every hypothesis gives the same complete answer to every test item."""
        for meaning in tests:
            predictions = set(self.predict(meaning, with_placeholders=False))
            if None in predictions or len(predictions) != 1:
                return False
        return True

    def choose_question(self, candidates: list[Meaning]) -> Meaning:
        """Return the candidate whose predicted observation splits the hypotheses most evenly."""
        num_hypotheses = len(self.hypotheses)
        best, best_entropy = candidates[0], -1.0
        for meaning in candidates:
            counts = Counter(self.predict(meaning, with_placeholders=True))
            entropy = -sum(
                count / num_hypotheses * math.log(count / num_hypotheses)
                for count in counts.values()
            )
            if entropy > best_entropy:
                best, best_entropy = meaning, entropy
        return best


def solve(seed: int, config: EnvConfig, num_candidates: int = 48) -> int | None:
    """Return the number of questions the greedy solver needs for seed, or None if over budget.

    Each turn the candidates are a random subset of the non-test meanings; asking a test item would
    only make it unscorable.
    """
    language, tests = generate(seed, config)
    solver = HypothesisSolver(dict(language.roots) if config.show_dictionary else None)
    rng = random.Random(seed * 7919 + 1)
    test_set = set(tests)
    pool = [meaning for meaning in language.list_meanings() if meaning not in test_set]
    for num_questions in range(1, config.budget + 1):
        question = solver.choose_question(rng.sample(pool, min(num_candidates, len(pool))))
        pool.remove(question)
        solver.observe(question, language.realize(question))
        if solver.is_solved(tests):
            return num_questions
    return None
