#!/bin/bash -l

set -e
set -u

export PROJ_DIR="/mnt/ceph_rbd/SAE-based-representation-engineering"
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="meta-llama/Llama-2-7b-hf"
SAVE_DIR_NAME="grouped_prompts"  # save to PROJ_DIR / "cache_data" / model_name / SAVE_DIR_NAME

python -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=3 \
  --seeds_to_encode 42 43 44 45 46

python -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=4 \
  --seeds_to_encode 42 43 44 45 46

python -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=5 \
  --seeds_to_encode 42 43 44 45 46
