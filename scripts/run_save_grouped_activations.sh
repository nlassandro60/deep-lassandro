#!/bin/bash -l

set -e
set -u

ulimit -n 10240

export PROJ_DIR="/mnt/ceph_rbd/SAE-based-representation-engineering"
export CUDA_VISIBLE_DEVICES=0

#python -m spare.save_grouped_activations \
#  --data_name="nqswap" \
#  --model_path="meta-llama/Llama-2-7b-hf" \
#  --load_data_name="grouped_prompts"\
#  --shots_to_encode 3 4 5 \
#  --seeds_to_encode 42 43 44 46 \
#  --save_hiddens_name="grouped_activations"
## seed = 45 exceeds context length
#
#python -m spare.save_grouped_activations \
#  --data_name="nqswap" \
#  --model_path="meta-llama/Meta-Llama-3-8B" \
#  --load_data_name="grouped_prompts"\
#  --shots_to_encode 3 4 5 \
#  --seeds_to_encode 42 43 44 45 46 \
#  --save_hiddens_name="grouped_activations"

python -m spare.save_grouped_activations \
  --data_name="nqswap" \
  --model_path="google/gemma-2-9b" \
  --load_data_name="grouped_prompts"\
  --shots_to_encode 3 4 5 \
  --seeds_to_encode 42 43 44 45 46 \
  --save_hiddens_name="grouped_activations"

wait