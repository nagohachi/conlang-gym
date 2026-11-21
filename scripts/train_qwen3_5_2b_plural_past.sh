#!/usr/bin/env bash
# Agentic GRPO | Qwen3.5-2B, thinking off | conlang-gym with word order + plural + past | FSDP2 training on 2 GPUs + vLLM rollout on 1 separate GPU (verl V1 separate_async)
#
# Usage, from the conlang-gym root with the verl venv activated. Build the data first:
#   python -m conlang_gym.make_dataset --out_dir outputs/data/plural_past --active_params plural past
# The first two visible GPUs train and the third rolls out:
#   bash scripts/train_qwen3_5_2b_plural_past.sh lora   # or: full, curriculum
# lora follows "LoRA Without Regret" (Thinking Machines, 2025): every linear layer of the language model, alpha 32, 10x the full-parameter learning rate, constant schedule. Rank 32 is our choice; the post found RL insensitive to rank down to 1.
# curriculum is full-parameter training that starts from the published word-order model (trained by scripts/train_qwen3_5_2b.sh nothink) instead of Qwen3.5-2B.
# The trainer and the rollout are on separate GPUs, so weights reach vLLM through the checkpoint engine, which sends plain tensors: LoRA is merged into the base weights before each sync (lora.merge).
set -xeuo pipefail

MODE=$1
MODEL=Qwen/Qwen3.5-2B
case "${MODE}" in
    lora)
        MODE_ARGS=(
            actor_rollout_ref.actor.optim.lr=1e-5
            actor_rollout_ref.model.lora_rank=32
            actor_rollout_ref.model.lora_alpha=32
            actor_rollout_ref.model.target_modules=all-linear
            "actor_rollout_ref.model.exclude_modules=.*visual.*"
            actor_rollout_ref.model.lora.merge=True
        ) ;;
    full) MODE_ARGS=(actor_rollout_ref.actor.optim.lr=1e-6) ;;
    curriculum)
        MODEL=nagohachi/Qwen3.5-2B-conlang-gym-v1-word-order-with-dictionary
        MODE_ARGS=(actor_rollout_ref.actor.optim.lr=1e-6) ;;
    *) echo "usage: $0 lora|full|curriculum" >&2; exit 1 ;;
esac
RUN=qwen3_5_2b_plural_past_${MODE}

# Ray workers import the agent loop as conlang_gym.agent_loop.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p outputs/logs

# One episode = up to 7 investigation turns (512 tokens each, plus a ~30-token observation) + the test prompt (~260 tokens) + the test answer (2,048): 7 x 542 + 260 + 2,048 = 6,102 <= 6,144.
# separate_async requires data.train_batch_size = parameter_sync_step x ppo_mini_batch_size (128 = 4 x 32).
# The actor packs at most 16,384 tokens per micro-batch: its logits are tokens x 248,320 vocab x 2 bytes (8 GB here), and with 24,576 tokens (12 GB) the full-parameter update ran out of memory on a 48 GB GPU at step 13. The cap only changes how a mini-batch is split, not the update.
# TransferQueue storage units (1 Ray CPU each), reward workers (~1 GB RAM each) and agent-loop workers (~1.5 GB RAM each) are cut down from 8 to 2; 2 are enough for one node and leave CPU slots and host memory to the trainer and the rollout.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${PWD}/outputs/data/plural_past/train.parquet" \
    data.val_files="['${PWD}/outputs/data/plural_past/val_seen.parquet','${PWD}/outputs/data/plural_past/val_unseen.parquet']" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=6144 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    reward.num_workers=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=conlang_gym/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=conlang_gym_agent \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.parameter_sync_step=4 \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=conlang_gym \
    trainer.experiment_name=${RUN} \
    trainer.log_val_generations=8 \
    trainer.rollout_data_dir="${PWD}/outputs/rollouts/${RUN}" \
    trainer.validation_data_dir="${PWD}/outputs/val_generations/${RUN}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir="${PWD}/outputs/ckpts/${RUN}" \
    trainer.total_epochs=1 \
    "${MODE_ARGS[@]}" \
    2>&1 | tee "outputs/logs/${RUN}_$(date +%Y%m%d_%H%M%S).log"
