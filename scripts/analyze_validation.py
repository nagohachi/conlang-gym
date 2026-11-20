"""Report accuracy per word order and the word order the model wrote, from verl validation dumps.

verl writes one jsonl per validation step to trainer.validation_data_dir. The dumps carry the
prompt but not the seed, so each episode is matched to its row in val_seen/val_unseen by the root
dictionary in the prompt, and its language and test set are regenerated from the seed.
"""

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conlang_gym.language import EnvConfig, Episode, normalize

logger = logging.getLogger(__name__)

DICT_RE = re.compile(r"## Root dictionary\n(.*?)\n\n## Actions", re.S)
ANSWER_RE = re.compile(r'<answer id="(\d+)">(.*?)</answer>', re.S)
WORD_ORDERS = ["SOV", "SVO", "VSO", "OSV", "OVS", "VOS"]


def main() -> None:
    """Log, for each validation step, the accuracy and the written word orders per word order."""
    parser = argparse.ArgumentParser(description="Analyze conlang-gym validation dumps.")
    parser.add_argument("val_dir", type=Path, help="trainer.validation_data_dir of a run")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("outputs/data"),
        help="directory with val_seen.parquet and val_unseen.parquet",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    episode_settings = {}
    for split in ("val_seen", "val_unseen"):
        df = pd.read_parquet(args.data_dir / f"{split}.parquet")
        for prompt, extra_info in zip(df.prompt, df.extra_info, strict=True):
            episode_settings[DICT_RE.search(prompt[0]["content"]).group(1)] = extra_info

    def summarize_step(dump: Path) -> str:
        num_correct, num_items = collections.Counter(), collections.Counter()
        written_orders = collections.defaultdict(collections.Counter)
        for line in dump.open():
            record = json.loads(line)
            extra_info = episode_settings[DICT_RE.search(record["input"]).group(1)]
            config = EnvConfig(**json.loads(extra_info["env_config"]))
            episode = Episode(int(extra_info["seed"]), config)
            language = episode.language
            word_order = language.grammar.word_order
            test_turn = record["output"].rsplit("The investigation is over.", 1)[-1]
            test_answer = test_turn.rsplit("</think>", 1)[-1]
            answers = {int(i): a for i, a in reversed(ANSWER_RE.findall(test_answer))}
            for item_id, meaning in enumerate(episode.tests, start=1):
                answer_words = normalize(answers.get(item_id, "")).split()
                num_correct[word_order] += answer_words == language.realize(meaning).split()
                num_items[word_order] += 1
                if meaning.transitive and meaning.subject != meaning.object:
                    roles = {
                        language.roots[meaning.subject]: "S",
                        language.roots[meaning.object]: "O",
                        language.roots[meaning.verb]: "V",
                    }
                    is_permutation = sorted(answer_words) == sorted(roles)
                    written = "".join(roles[w] for w in answer_words) if is_permutation else "other"
                    written_orders[word_order][written] += 1
        parts = []
        for word_order in sorted(num_items, key=WORD_ORDERS.index):
            counts = written_orders[word_order]
            total = sum(counts.values())
            top = ", ".join(f"{order} {100 * n / total:.0f}%" for order, n in counts.most_common(3))
            accuracy = num_correct[word_order] / num_items[word_order]
            parts.append(f"{word_order} {accuracy:.2f} [{top}]")
        return " | ".join(parts)

    dumps = sorted(args.val_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    logger.info("accuracy per word order [word order written in transitive answers, top 3]")
    for dump in dumps:
        logger.info("step %3s: %s", dump.stem, summarize_step(dump))


if __name__ == "__main__":
    main()
