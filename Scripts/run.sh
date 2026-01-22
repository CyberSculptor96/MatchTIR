set -x
unset PYTORCH_CUDA_ALLOC_CONF
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29505
export PYTHONPATH=Code:$PYTHONPATH

export WANDB_API_KEY="5149358c5b6d816ca391d37a8e4cafb8fd796f64"
# export WANDB_BASE_URL="https://api.bandw.top"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpus=4
gpu_memory_utilization=0.5
prompt_len=7000     # 7000
response_len=23000  # 23000
ppo_max_token_len_per_gpu=30000
sp_size=4
vllm_tp_size=4
mini_batch_size=16

train_files="Data/train.parquet"
test_files="Data/test.parquet"
# model="Qwen3-8B"
model="/wangbenyou-huanghejin/models/Qwen3-8B"

rollout_mode="sync"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=$prompt_len \
    data.max_response_length=$response_len \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=messages \
    data.system_style=Qwen3 \
    data.enable_thinking=True \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$vllm_tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=tool \
    custom_reward_function.path=Code/verl/utils/reward_score/tool.py \
    custom_reward_function.name=compute_process_KM \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MatchTIR' \
    trainer.experiment_name='MatchTIR-8B' \
    trainer.val_before_train=False \
    +trainer.val_only=False \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=4 \
    trainer.total_epochs=3 $@ \
    2>&1 | tee logs/run_matchtir.log
