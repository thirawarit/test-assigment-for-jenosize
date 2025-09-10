#!/bin/sh

MODEL_NAME="Qwen/Qwen3-8B"

export PYTHONPATH=src:$PYTHONPATH

python ./src/merge_lora_weights.py \
    --model-path ./output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path ./output/merge_test \
    --safe-serialization