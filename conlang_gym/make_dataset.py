"""Build the train and validation parquet files of conlang-gym.

Each row is one seed; the agent loop regenerates the language from extra_info, so the answers never
appear in the data. Seeds are kept only if the greedy solver finishes within the budget (spec 8).

Grammars (canonical parameter combinations) are split into seen and unseen ones, and unseen grammars
never occur in training (spec 9). Three files are written, with disjoint seeds:
    train.parquet       seen grammars
    val_seen.parquet    seen grammars, num_val_seen_per_grammar seeds each
    val_unseen.parquet  unseen grammars, num_val_unseen_per_grammar seeds each
Every row's data_source is conlang_gym_<split>_<grammar label>. verl reports validation metrics
per data_source (val-core/<data_source>/reward/mean@1), so each grammar of each validation split
gets its own accuracy curve.
"""

import argparse
import collections
import itertools
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .language import OPTIONAL_PARAMS, EnvConfig, Episode, Grammar, generate
from .solver import ALL_GRAMMARS, solve

logger = logging.getLogger(__name__)

type Row = dict[str, str | list[dict[str, str]] | dict[str, str | int | None]]


@dataclass(frozen=True)
class SeedCheck:
    """The grammar of a seed and the number of questions the solver needed (None if it failed)."""

    seed: int
    grammar: Grammar
    num_questions: int | None


def _check_seed(seed: int, config: EnvConfig) -> SeedCheck:
    """Generate the language of seed and run the solver on it.

    It is a module-level function so that the worker processes can unpickle it.
    """
    language, _ = generate(seed, config)
    return SeedCheck(seed, language.grammar, solve(seed, config))


def main() -> None:
    """Build train.parquet, val_seen.parquet and val_unseen.parquet in --out_dir."""

    def parse_bool(text: str) -> bool:
        # type=bool would turn any non-empty string, even "False", into True.
        if text.lower() in ("true", "1", "yes"):
            return True
        if text.lower() in ("false", "0", "no"):
            return False
        raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")

    def build_grammar_key(grammar: Grammar) -> str:
        return json.dumps(asdict(grammar), sort_keys=True)

    def build_grammar_label(grammar: Grammar) -> str:
        # The word order, then each active parameter (e.g. OVS_past-prefix).
        active = [
            f"{p}-{getattr(grammar, p)}" for p in OPTIONAL_PARAMS if getattr(grammar, p) != "none"
        ]
        parts = [grammar.word_order, *active]
        has_case_suffix = "suffix" in (grammar.object_marking, grammar.ergative_marking)
        if has_case_suffix and grammar.plural == "suffix":
            parts.append(grammar.plural_case_order)
        return "_".join(parts)

    def build_row(check: SeedCheck, split: str, index: int) -> Row:
        return {
            "data_source": f"conlang_gym_{split}_{build_grammar_label(check.grammar)}",
            "prompt": [
                {"role": "user", "content": Episode(check.seed, config).build_initial_prompt()}
            ],
            "ability": "conlang_gym",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "split": split,
                "index": index,
                "seed": check.seed,
                "env_config": json.dumps(asdict(config)),
                "grammar": build_grammar_key(check.grammar),
                "num_solver_questions": check.num_questions,
            },
        }

    parser = argparse.ArgumentParser(
        description="Build the conlang-gym train and validation files."
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_val_seen_per_grammar", type=int, default=64)
    parser.add_argument("--num_val_unseen_per_grammar", type=int, default=64)
    parser.add_argument("--unseen_grammar_fraction", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)
    for name, config_field in EnvConfig.__dataclass_fields__.items():
        default = config_field.default
        if isinstance(default, tuple):
            parser.add_argument(f"--{name}", nargs="*", default=list(default))
            continue
        arg_type = parse_bool if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{name}", type=arg_type, default=default)
    args = parser.parse_args()
    config = EnvConfig(**{name: getattr(args, name) for name in EnvConfig.__dataclass_fields__})
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
    num_unseen = max(1, round(len(grammars) * args.unseen_grammar_fraction))
    unseen, seen = set(grammars[:num_unseen]), set(grammars[num_unseen:])
    logger.info(
        "%d grammars with active parameters %s: seen %s, unseen %s",
        len(grammars),
        list(config.active_params),
        sorted(map(build_grammar_label, seen)),
        sorted(map(build_grammar_label, unseen)),
    )

    quotas = {
        "val_seen": args.num_val_seen_per_grammar,
        "val_unseen": args.num_val_unseen_per_grammar,
    }
    rows: dict[str, list[Row]] = {"train": [], "val_seen": [], "val_unseen": []}
    num_rows_per_grammar = {split: collections.Counter() for split in rows}
    num_checked = num_rejected = 0
    next_seed, chunk_size = 0, 2000
    with ProcessPoolExecutor(args.num_workers) as pool:
        while (
            len(rows["train"]) < args.num_train
            or any(num_rows_per_grammar["val_seen"][g] < quotas["val_seen"] for g in seen)
            or any(num_rows_per_grammar["val_unseen"][g] < quotas["val_unseen"] for g in unseen)
        ):
            seeds = range(next_seed, next_seed + chunk_size)
            next_seed += chunk_size
            for check in pool.map(_check_seed, seeds, itertools.repeat(config), chunksize=50):
                num_checked += 1
                if check.num_questions is None:
                    num_rejected += 1
                    continue
                # Validation quotas are filled first, so each grammar's validation seeds are its
                # earliest ones.
                split = "val_unseen" if check.grammar in unseen else "val_seen"
                if num_rows_per_grammar[split][check.grammar] >= quotas[split]:
                    if split == "val_unseen" or len(rows["train"]) >= args.num_train:
                        continue
                    split = "train"
                num_rows_per_grammar[split][check.grammar] += 1
                rows[split].append(build_row(check, split, len(rows[split])))
            logger.info(
                "checked %d seeds, rejected %d, %s",
                num_checked,
                num_rejected,
                ", ".join(f"{split} {len(split_rows)}" for split, split_rows in rows.items()),
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        pd.DataFrame(split_rows).to_parquet(args.out_dir / f"{split}.parquet")
    summary = {
        **asdict(config),
        "seen_grammars": sorted(map(build_grammar_label, seen)),
        "unseen_grammars": sorted(map(build_grammar_label, unseen)),
        **{f"num_{split}": len(split_rows) for split, split_rows in rows.items()},
        "num_checked_seeds": num_checked,
        "num_rejected_seeds": num_rejected,
    }
    (args.out_dir / "env_config.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
