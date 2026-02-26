export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=<MODEL_PATH>

# 直接设置你想要的名称
RUN_NAME="covtreasoner_7b_rl_v1-simpleformat_reward_no_process_reward-vilr39k-necessary-reward-cropckpt"

python3 -m verl.trainer.main \
    config=training_scripts/covtreasoner_7b.yaml \
    data.train_files=<TRAIN_DATA_PATH> \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=covt_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=<CHECKPOINT_SAVE_PATH>/covtreasoner_workdir/${RUN_NAME}