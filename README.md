# Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering

---

This repository hosts the data and code of the paper: Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering

---


## Project Setup
```bash
conda create -n spare python=3.9 -y
conda activate spare
bash ./scripts/install.sh
```

## Run SpARE


```bash
python ./demo.py
```

Test your cases by replacing `test_examples`.

SpARE currently only supports short-form ODQA task, and we plan to add support for more tasks in the next version. 

## Run Experiments

Use the cached intermediate data to run experiments.

The cached data is in the `cache_data` folder, including mutual information, expectation, and the values of functional SAE activations.  

```bash
bash ./scripts/run_all_experiments.sh
```

## Run SpARE Step by Step

Observe the outputs of prompts and group them based on the knowledge selection behaviours:
```bash
bash ./scripts/run_group_prompts.sh
```

Save the activations of grouped prompts:
```bash
bash ./scripts/run_save_grouped_activations.sh
```

Estimate the mutual information and expectations for each SAE activation: 

```bash
bash ./scripts/run_mutual_information_and_expectations.sh
```

Evaluate SpARE
```bash
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
```
