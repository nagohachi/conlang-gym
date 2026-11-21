"""Write the seed selection of the published model's dataset to web/src/env/dataset_seeds.json.

It replays conlang_gym/make_dataset.py's selection, without pandas and without writing parquet
files. --active_params selects the data: the site's model (v2) was trained on
--active_params plural past, every other flag at its default (8000 train rows, 64 validation seeds
per grammar, unseen grammar fraction 0.2, chunks of 2000 seeds). Run from the repository root:
    PYTHONPATH=. python web/scripts/dataset_seeds.py --active_params plural past
The website uses the file to tell which seeds were validation seeds, and which may have been
trained on: a seed up to lastTrainSeed of a seen grammar that the solver keeps and that is not a
validation seed is a training seed, and seeds from numCheckedSeeds on were never examined.
"""

import argparse
import collections
import dataclasses
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from conlang_gym.language import OPTIONAL_PARAMS, EnvConfig, Grammar, generate
from conlang_gym.solver import ALL_GRAMMARS, solve

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parent.parent / "src" / "env" / "dataset_seeds.json"
NUM_TRAIN = 8000
NUM_VAL_SEEN_PER_GRAMMAR = 64
NUM_VAL_UNSEEN_PER_GRAMMAR = 64
UNSEEN_GRAMMAR_FRACTION = 0.2
CHUNK_SIZE = 2000
# EnvConfig field names in the camelCase that the TypeScript port uses.
CAMEL_CASE = {
    "num_nouns": "numNouns",
    "num_transitive": "numTransitive",
    "num_intransitive": "numIntransitive",
    "active_params": "activeParams",
    "show_dictionary": "showDictionary",
    "enable_judge": "enableJudge",
    "budget": "budget",
    "num_test": "numTest",
    "num_compositional": "numCompositional",
    "solver_params": "solverParams",
}


def to_camel_case(name: str) -> str:
    """Return a snake_case name in camelCase, as the TypeScript port names fields and parameters."""
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


@dataclasses.dataclass(frozen=True)
class SeedCheck:
    """The grammar of a seed and the number of questions the solver needed (None if it failed)."""

    seed: int
    grammar: Grammar
    num_questions: int | None


def check_seed(seed: int, config: EnvConfig) -> SeedCheck:
    """Generate the language of seed and run the solver on it (make_dataset's _check_seed)."""
    language, _ = generate(seed, config)
    return SeedCheck(seed, language.grammar, solve(seed, config))


def build_grammar_key(grammar: Grammar) -> str:
    """Return make_dataset's sort key of a grammar."""
    return json.dumps(asdict(grammar), sort_keys=True)


def build_grammar_label(grammar: Grammar) -> str:
    """Return make_dataset's label of a grammar: the word order, then each active parameter."""
    active = [
        f"{p}-{getattr(grammar, p)}" for p in OPTIONAL_PARAMS if getattr(grammar, p) != "none"
    ]
    parts = [grammar.word_order, *active]
    has_case_suffix = "suffix" in (grammar.object_marking, grammar.ergative_marking)
    if has_case_suffix and grammar.plural == "suffix":
        parts.append(grammar.plural_case_order)
    return "_".join(parts)


def main() -> None:
    """Replay make_dataset's seed selection and write the summary the website needs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--active_params", nargs="*", default=[])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = EnvConfig(active_params=tuple(args.active_params))

    grammars = sorted(
        (
            grammar
            for grammar in ALL_GRAMMARS
            if {p for p in OPTIONAL_PARAMS if getattr(grammar, p) != "none"}
            == set(config.active_params)
        ),
        key=build_grammar_key,
    )
    random.Random(0).shuffle(grammars)
    num_unseen = max(1, round(len(grammars) * UNSEEN_GRAMMAR_FRACTION))
    unseen, seen = set(grammars[:num_unseen]), set(grammars[num_unseen:])

    quotas = {"val_seen": NUM_VAL_SEEN_PER_GRAMMAR, "val_unseen": NUM_VAL_UNSEEN_PER_GRAMMAR}
    val_seeds: dict[str, dict[str, list[int]]] = {"val_seen": {}, "val_unseen": {}}
    num_rows_per_grammar = {
        split: collections.Counter() for split in ("train", "val_seen", "val_unseen")
    }
    num_train = num_checked = num_rejected = 0
    last_train_seed = -1
    next_seed = 0
    with ProcessPoolExecutor(args.num_workers) as pool:
        while (
            num_train < NUM_TRAIN
            or any(num_rows_per_grammar["val_seen"][g] < quotas["val_seen"] for g in seen)
            or any(num_rows_per_grammar["val_unseen"][g] < quotas["val_unseen"] for g in unseen)
        ):
            seeds = range(next_seed, next_seed + CHUNK_SIZE)
            next_seed += CHUNK_SIZE
            configs = [config] * len(seeds)
            for check in pool.map(check_seed, seeds, configs, chunksize=50):
                num_checked += 1
                if check.num_questions is None:
                    num_rejected += 1
                    continue
                # Validation quotas are filled first, so each grammar's validation seeds are its
                # earliest ones.
                split = "val_unseen" if check.grammar in unseen else "val_seen"
                if num_rows_per_grammar[split][check.grammar] >= quotas[split]:
                    if split == "val_unseen" or num_train >= NUM_TRAIN:
                        continue
                    split = "train"
                num_rows_per_grammar[split][check.grammar] += 1
                if split == "train":
                    num_train += 1
                    last_train_seed = check.seed
                else:
                    label = build_grammar_label(check.grammar)
                    val_seeds[split].setdefault(label, []).append(check.seed)
            logger.info(
                "checked %d seeds, rejected %d, train %d", num_checked, num_rejected, num_train
            )

    summary = {
        "envConfig": {
            CAMEL_CASE[name]: [to_camel_case(p) for p in value]
            if isinstance(value, tuple)
            else value
            for name, value in asdict(config).items()
        },
        "seenGrammars": sorted(map(build_grammar_label, seen)),
        "unseenGrammars": sorted(map(build_grammar_label, unseen)),
        "valSeen": dict(sorted(val_seeds["val_seen"].items())),
        "valUnseen": dict(sorted(val_seeds["val_unseen"].items())),
        "numTrain": num_train,
        "lastTrainSeed": last_train_seed,
        "numCheckedSeeds": num_checked,
        "numRejectedSeeds": num_rejected,
    }
    args.out.write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    logger.info("wrote %s (%d bytes)", args.out, args.out.stat().st_size)


if __name__ == "__main__":
    main()
