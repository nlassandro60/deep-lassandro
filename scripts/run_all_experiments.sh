#!/bin/bash -l

set -e
set -u

for seed in {42..46}; do
    if [ ${seed} -eq 45 ]
    then
     continue # seed = 45 exceeds context length
    fi
    python ./scripts/run_spare.py \
      --model_path="meta-llama/Llama-2-7b-hf" \
      --data_name="nqswap" \
      --layer_ids 12 13 14 15 \
      --edit_degree=2.0 \
      --select_topk_proportion=0.07 \
      --seed=${seed} \
      --hiddens_name="grouped_activations" \
      --mutual_information_save_name="mutual_information" \
      --run_use_parameter \
      --run_use_context
done

python ./scripts/avg_acc.py \
    --model_path="meta-llama/Llama-2-7b-hf" \
    --data_name="nqswap" \

for seed in {42..46}; do
    python ./scripts/run_spare.py \
      --model_path="meta-llama/Meta-Llama-3-8B" \
      --data_name="nqswap" \
      --layer_ids 13 14 15 16 \
      --edit_degree=2.0 \
      --select_topk_proportion=0.07 \
      --seed=${seed} \
      --hiddens_name="grouped_activations" \
      --mutual_information_save_name="mutual_information" \
      --run_use_parameter \
      --run_use_context
done

python ./scripts/avg_acc.py \
    --model_path="meta-llama/Meta-Llama-3-8B" \
    --data_name="nqswap" \


for seed in {42..46}; do
    python ./scripts/run_spare.py \
      --model_path="google/gemma-2-9b" \
      --data_name="nqswap" \
      --layer_ids 23 24 25 26 \
      --edit_degree=3.0 \
      --select_topk_proportion=0.01 \
      --seed=${seed} \
      --hiddens_name="grouped_activations" \
      --mutual_information_save_name="mutual_information" \
      --run_use_context
done

for seed in {42..46}; do
    python ./scripts/run_spare.py \
      --model_path="google/gemma-2-9b" \
      --data_name="nqswap" \
      --layer_ids 23 24 25 29 30 31 \
      --edit_degree=1.8 \
      --select_topk_proportion=0.01 \
      --seed=${seed} \
      --hiddens_name="grouped_activations" \
      --mutual_information_save_name="mutual_information" \
      --run_use_parameter
done

python ./scripts/avg_acc.py \
    --model_path="google/gemma-2-9b" \
    --data_name="nqswap" \
