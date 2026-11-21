"""verl agent loop for conlang-gym.

One trajectory is one episode: the model acts in the investigation phase (one <action> per turn, the
informant's reply comes back as a user turn), then answers the whole test set in a single turn. The
reward comes from the environment, so no reward function is involved. Reply and prompt tokens get
response_mask 0, so only the model's own tokens are trained.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from transformers import PreTrainedTokenizerBase, ProcessorMixin
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    DictConfigWrap,
    ToolListWrap,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from .language import EnvConfig, Episode

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_FORCE_ANSWER_TEXT = (
    "\n\nConsidering the limited time, I have to give the answer based on the thinking so far."
    "\n</think>\n\n"
)


@dataclass
class Trajectory:
    """The token stream of an episode so far, with the per-token data that verl trains on."""

    token_ids: list[int]
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    # Fields the rollout server attaches to its outputs; the trainer reads min/max_global_steps,
    # the weight versions a trajectory was generated with.
    server_fields: dict[str, JsonValue] = field(default_factory=dict)


class ConlangGymAgentLoop(AgentLoopBase):
    """Agent loop that plays one conlang-gym episode against verl's rollout server and scores it."""

    merges_context_messages = True

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin | None,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        hf_model_type: str | None = None,
        tools: ToolListWrap | None = None,
        name: str | None = None,
        turn_max_tokens: int = 256,
        test_max_tokens: int = 512,
        think_budget_turn: int = 0,
        think_budget_test: int = 0,
        coverage_weight: float = 0.0,
    ) -> None:
        """Set the token limits of the turns.

        verl passes tools and the loop's registry name (the name in agent_loop.yaml) to every agent
        loop; this loop uses neither.

        turn_max_tokens is L in spec 5.4 (a longer investigation turn is a format error), and a test
        answer longer than test_max_tokens scores 0 (spec 7.3). With thinking mode, a nonzero think
        budget caps the reasoning of each turn: once it is spent without </think>, the loop appends
        a fixed closing sentence and </think> (the thinking-budget scheme of the Qwen3 technical
        report) and the model then writes the visible part within turn/test_max_tokens. The
        appended tokens are the environment's, so they get response_mask 0.

        A nonzero coverage_weight adds coverage_weight x EpisodeScore.coverage to the reward, which
        rewards investigations that ask about every condition; the test accuracy stays in
        reward_extra_info as accuracy.
        """
        super().__init__(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls,
            data_config,
            hf_model_type,
        )
        self.turn_max_tokens = turn_max_tokens
        self.test_max_tokens = test_max_tokens
        self.think_budget_turn = think_budget_turn
        self.think_budget_test = think_budget_test
        self.coverage_weight = coverage_weight
        self.response_length = self.rollout_config.response_length
        self.think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.force_answer_ids = self.tokenizer.encode(_FORCE_ANSWER_TEXT, add_special_tokens=False)

    @rollout_trace_op
    async def run(
        self, sampling_params: dict[str, JsonValue], priority: int = 0, **kwargs: JsonValue
    ) -> AgentLoopOutput:
        """Play one episode (investigation turns, then the test turn) and return it, scored."""
        extra_info = kwargs["extra_info"]
        episode = Episode(
            int(extra_info["seed"]), EnvConfig(**json.loads(extra_info["env_config"]))
        )
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        metrics: dict[str, float] = {}
        trajectory = Trajectory(token_ids=await self.ct_build_initial_tokens(messages))

        async def generate(max_tokens: int) -> bool:
            """Generate up to max_tokens with mask 1; return whether generation hit the limit."""
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=trajectory.token_ids,
                    sampling_params={**sampling_params, "max_tokens": max_tokens},
                )
            old_logprobs = trajectory.response_logprobs
            merge_result, trajectory.response_mask, logprobs = await self.ct_merge_assistant_token(
                trajectory.token_ids,
                output.token_ids,
                trajectory.response_mask,
                old_logprobs if (old_logprobs or output.log_probs) else None,
                assistant_logprobs=output.log_probs or None,
            )
            trajectory.token_ids = merge_result.token_ids
            if logprobs is not None:
                trajectory.response_logprobs = logprobs
            if not trajectory.server_fields:
                trajectory.server_fields.update(output.extra_fields)
            elif output.extra_fields.get("max_global_steps") is not None:
                trajectory.server_fields["max_global_steps"] = output.extra_fields[
                    "max_global_steps"
                ]
            if output.stop_reason is None:
                return len(output.token_ids) >= max_tokens
            return output.stop_reason == "length"

        async def run_assistant_turn(in_test: bool) -> bool | None:
            """Run one assistant turn; return whether it hit a limit, or None if no room is left."""
            answer_max = self.test_max_tokens if in_test else self.turn_max_tokens
            think_budget = self.think_budget_test if in_test else self.think_budget_turn
            room = self.response_length - len(trajectory.response_mask)
            if min(think_budget or answer_max, room) <= 0:
                return None
            if think_budget <= 0:
                return await generate(min(answer_max, room))
            turn_start = len(trajectory.token_ids)
            if not await generate(min(think_budget, room)):
                return False
            if self.think_end_id not in trajectory.token_ids[turn_start:]:
                forced = self.force_answer_ids
                trajectory.token_ids = trajectory.token_ids + forced
                trajectory.response_mask = trajectory.response_mask + [0] * len(forced)
                if trajectory.response_logprobs:
                    trajectory.response_logprobs = trajectory.response_logprobs + [0.0] * len(
                        forced
                    )
            max_tokens = min(answer_max, self.response_length - len(trajectory.response_mask))
            if max_tokens <= 0:
                return True
            return await generate(max_tokens)

        num_assistant_turns = num_user_turns = 0
        score = None
        while True:
            in_test = episode.in_test_phase
            turn_start = len(trajectory.token_ids)
            truncated = await run_assistant_turn(in_test)
            if truncated is None:
                break
            num_assistant_turns += 1
            # In thinking mode the reasoning may quote <action> or <answer> tags, so only the text
            # after the last </think> counts.
            turn_text = self.tokenizer.decode(
                trajectory.token_ids[turn_start:], skip_special_tokens=False
            )
            text = _SPECIAL_TOKEN_RE.sub("", turn_text.rsplit("</think>", 1)[-1])
            if in_test:
                score = episode.score(text, truncated)
                break
            reply = episode.step(text, truncated)
            messages.append({"role": "assistant", "content": text})
            previous_messages = list(messages)
            messages.append({"role": "user", "content": reply})
            merge_result, new_mask, logprobs = await self.ct_merge_context_msg(
                previous_messages,
                messages,
                trajectory.token_ids,
                trajectory.response_mask,
                trajectory.response_logprobs or None,
            )
            if len(new_mask) >= self.response_length:
                break
            trajectory.token_ids, trajectory.response_mask = merge_result.token_ids, new_mask
            if trajectory.response_logprobs:
                trajectory.response_logprobs = logprobs or []
            num_user_turns += 1

        if score is None:
            # Ran out of response length before the test turn finished: the episode scores 0.
            score = episode.score("", truncated=True)
        response_mask = trajectory.response_mask
        if score.num_scorable == 0:
            # Spec 7.3 drops such episodes; with a zero mask the trajectory contributes no gradient.
            response_mask = [0] * len(response_mask)
        num_response_tokens = len(response_mask)
        token_ids = trajectory.token_ids
        prompt_ids = token_ids[: len(token_ids) - num_response_tokens]
        response_ids = (
            token_ids[len(token_ids) - num_response_tokens :] if num_response_tokens else []
        )
        # The trainer records the reward itself under "reward", so the accuracy is renamed.
        reward_extra_info = {
            ("accuracy" if name == "reward" else name): value
            for name, value in asdict(score).items()
        }
        logprobs = trajectory.response_logprobs
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=logprobs[: self.response_length] if logprobs else None,
            reward_score=score.reward + self.coverage_weight * score.coverage,
            num_turns=num_assistant_turns + num_user_turns + 1,
            metrics=metrics,
            extra_fields={
                **trajectory.server_fields,
                "turn_scores": [],
                "tool_rewards": [],
                "reward_extra_info": reward_extra_info,
            },
        )
