#!/usr/bin/env bash
# Validation-only run | Qwen3.5-2B word-order models, thinking off | conlang-gym languages with phenomena absent from their training data
#
# Compares (a) the published word-order model (scripts/train_qwen3_5_2b.sh nothink) with (b) the same training plus the coverage reward (scripts/train_qwen3_5_2b.sh coverage). Build the test data first:
#   ALL="object_marking plural past subject_agreement vowel_harmony ergative_marking object_plural verb_number_stem"
#   python -m conlang_gym.make_dataset --out_dir outputs/data/test_ergative --active_params ergative_marking --solver_params $ALL --num_train 0
#   python -m conlang_gym.make_dataset --out_dir outputs/data/test_object_plural --active_params plural object_plural --solver_params $ALL --num_train 0
#   python -m conlang_gym.make_dataset --out_dir outputs/data/test_verb_number_stem --active_params verb_number_stem --solver_params $ALL --num_train 0
# The test sets have no training split, so val_seen and val_unseen are both test data. The word-order validation files are included as a control.
# Usage, from the conlang-gym root with the verl venv activated, on three visible GPUs:
#   bash scripts/eval_qwen3_5_2b_transfer.sh a   # or: b
# b loads the last checkpoint of the coverage run (trainer.resume_mode=auto) before validating.
set -xeuo pipefail

MODE=$1
case "${MODE}" in
    a) MODEL=nagohachi/Qwen3.5-2B-conlang-gym-v1-word-order-with-dictionary; CKPT_DIR="${PWD}/outputs/ckpts/eval_transfer_a" ;;
    b) MODEL=Qwen/Qwen3.5-2B; CKPT_DIR="${PWD}/outputs/ckpts/qwen3_5_2b_coverage" ;;
    *) echo "usage: $0 a|b" >&2; exit 1 ;;
esac
RUN=qwen3_5_2b_transfer_eval_${MODE}
VAL_FILES=""
for dir in outputs/data/test_ergative outputs/data/test_object_plural outputs/data/test_verb_number_stem outputs/data; do
    VAL_FILES+="'${PWD}/${dir}/val_seen.parquet','${PWD}/${dir}/val_unseen.parquet',"
done

# Ray workers import the agent loop as conlang_gym.agent_loop.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p outputs/logs

# The settings match scripts/train_qwen3_5_2b.sh nothink, so the validation episodes run exactly as during training; train_files is required but unused.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${PWD}/outputs/data/train.parquet" \
    data.val_files="[${VAL_FILES%,}]" \
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
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=conlang_gym/agent_loop.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=conlang_gym_agent \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.parameter_sync_step=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=conlang_gym \
    trainer.experiment_name=${RUN} \
    trainer.validation_data_dir="${PWD}/outputs/val_generations/${RUN}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.default_local_dir="${CKPT_DIR}" \
    2>&1 | tee "outputs/logs/${RUN}_$(date +%Y%m%d_%H%M%S).log"
