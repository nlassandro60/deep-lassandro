#!/bin/bash -l

set -e
set -u


K_SHOT=32
NUM_EXAMPLES=-1

python -m kcm.eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-${NUM_EXAMPLES}examples-closebook" \
  --model_path="google/gemma-2-9b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --num_examples=${NUM_EXAMPLES} \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_close_book \
  --write_logs

K_SHOT=4

python -m kcm.eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-${NUM_EXAMPLES}examples-openbook" \
  --model_path="google/gemma-2-9b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --num_examples=${NUM_EXAMPLES} \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_open_book \
  --write_logs

python -m kcm.eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-${NUM_EXAMPLES}examples-openbook-noconflict" \
  --model_path="google/gemma-2-9b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --num_examples=${NUM_EXAMPLES} \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --test_example_org_context \
  --run_open_book \
  --write_logs
