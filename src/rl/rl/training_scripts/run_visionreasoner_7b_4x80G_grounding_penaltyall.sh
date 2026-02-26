export CUDA_VISIBLE_DEVICES=4,5,6,7

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=<MODEL_PATH>  # Cleaned checkpoint without anchor weights

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/visionreasoner_7b.yaml \
    data.train_files=<TRAIN_DATA_PATH> \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=vision_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=<CHECKPOINT_SAVE_PATH>/visionreasoner_workdir/${RUN_NAME}