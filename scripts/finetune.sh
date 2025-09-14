#!/bin/sh

# You can use 1.7B instead of 8B
MODEL_NAME="Qwen/Qwen3-8B"

# export CUDA_VISIBLE_DEVICES=1,2 # Optional

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=2
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
CHECKPOINT_NAME='trial-qwen3-sft-jenosize-0_0_0b'
export PYTHONPATH=src:$PYTHONPATH

mkdir -p ./output/output_logs

deepspeed ./src/train/train_sft.py \
    --deepspeed ./scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path datasets/train/*.jsonl \
    --remove_unused_columns False \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ./output/${CHECKPOINT_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
        2>&1 | tee ./output/output_logs/history_log_${CHECKPOINT_NAME}.out
    # --gradient_accumulation_steps $GRAD_ACCUM_STEPS \