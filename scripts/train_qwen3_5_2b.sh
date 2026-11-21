#!/usr/bin/env bash
# Agentic GRPO | Qwen3.5-2B | conlang-gym | FSDP2 training on 2 GPUs + vLLM rollout on 1 separate GPU (verl V1 separate_async)
#
# Usage, from the conlang-gym root with libs/verl/.venv activated. Build the data first:
#   python -m conlang_gym.make_dataset --out_dir outputs/data
# (word order only: with word order + 1 parameter, the spec's initial setting, Qwen3.5-2B scored 4.5% before training, below the spec's 20-60% calibration window.)
# Validation has two files: val_seen (training word orders, held-out seeds) and val_unseen (a word order never seen in training). Metrics are per data_source, i.e. per split and word order:
#   val-core/conlang_gym_val_seen_<order>/reward/mean@1, val-core/conlang_gym_val_unseen_<order>/reward/mean@1
# A checkpoint every 10 steps (2 kept) lets an interrupted run resume (trainer.resume_mode=auto).
# The argument picks Qwen3.5's thinking mode (think) or not (nothink); coverage is nothink with a reward for asking about every condition (conlang_gym_agent_coverage in conlang_gym/agent_loop.yaml). Ray hands out the visible GPUs in order and creates the trainer pool first, so the first two visible GPUs train and the third rolls out:
#   bash scripts/train_qwen3_5_2b.sh nothink
set -xeuo pipefail

# One episode = up to 7 investigation turns (L tokens each, plus a ~30-token observation) + the test prompt (~250 tokens) + the test answer; the initial prompt is ~330 tokens.
# L and the answer limit are set per agent loop in conlang_gym/agent_loop.yaml (think: reasoning budget + 22 forced-closing tokens + answer).
MODE=$1
case "${MODE}" in
    nothink) THINKING=False; AGENT=conlang_gym_agent; RESPONSE_LENGTH=6144 ;;        # 7 x 542 + 260 + 2,048 = 6,102
    coverage) THINKING=False; AGENT=conlang_gym_agent_coverage; RESPONSE_LENGTH=6144 ;;  # nothink + coverage reward
    think) THINKING=True; AGENT=conlang_gym_agent_think; RESPONSE_LENGTH=7168 ;;     # 7 x (512+22+128+30) + 260 + 1,024+22+512 = 6,662
    *) echo "usage: $0 nothink|think|coverage" >&2; exit 1 ;;
esac

# Ray workers import the agent loop as conlang_gym.agent_loop.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p outputs/logs

# separate_async requires data.train_batch_size = parameter_sync_step x ppo_mini_batch_size (128 = 4 x 32).
# The actor and the log-prob passes pack at most 16,384 tokens per micro-batch: its logits are tokens x 248,320 vocab x 2 bytes (8 GB here), and 24,576 tokens (12 GB) ran a full-parameter update out of memory on a 48 GB GPU. The cap only changes how a mini-batch is split, not the update.
# TransferQueue storage units (1 Ray CPU each), reward workers (~1 GB RAM each) and agent-loop workers (~1.5 GB RAM each) are cut down from 8 to 2; 2 are enough for one node and leave CPU slots and host memory to the trainer and the rollout.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${PWD}/outputs/data/train.parquet" \
    data.val_files="['${PWD}/outputs/data/val_seen.parquet','${PWD}/outputs/data/val_unseen.parquet']" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=${RESPONSE_LENGTH} \
    +data.apply_chat_template_kwargs.enable_thinking=${THINKING} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    reward.num_workers=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-2B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=conlang_gym/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT} \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.parameter_sync_step=4 \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=conlang_gym \
    trainer.experiment_name=qwen3_5_2b_${MODE} \
    trainer.log_val_generations=8 \
    trainer.rollout_data_dir="${PWD}/outputs/rollouts/qwen3_5_2b_${MODE}" \
    trainer.validation_data_dir="${PWD}/outputs/val_generations/qwen3_5_2b_${MODE}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir="${PWD}/outputs/ckpts/qwen3_5_2b_${MODE}" \
    trainer.total_epochs=1 \
    2>&1 | tee "outputs/logs/qwen3_5_2b_${MODE}_$(date +%Y%m%d_%H%M%S).log"
