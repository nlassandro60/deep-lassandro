import os

assert os.environ.get('PROJ_DIR') is not None

import numpy as np
import torch
from spare.datasets.function_extraction_datasets import REODQADataset
from spare.spare_for_generation import load_hiddens_and_get_function_weights, \
    prepare_patch_function, generate_with_patch, load_function_activations
import logging
import json
from spare.sae_repe_utils import load_dataset_and_memorised_set
from spare.utils import init_frozen_language_model, load_frozen_sae

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_functional_activations_weight(layer_ids, model_name, hiddens_name):
    all_use_context_weight, all_use_parameter_weight = [], []
    all_sae = []
    for layer_idx in layer_ids:
        logger.info(f"load function weights layer{layer_idx}")
        sae = load_frozen_sae(layer_idx, model_name)
        use_context_weight, use_parameter_weight = load_hiddens_and_get_function_weights(
            model_name, layer_idx, sae, hiddens_name,
        )
        all_use_context_weight.append(use_context_weight)
        all_use_parameter_weight.append(use_parameter_weight)
        all_sae.append(sae)
    return all_use_context_weight, all_use_parameter_weight, all_sae


def select_functional_activations(layer_ids, model_name, edit_degree, hiddens_name,
                                  mutual_information_save_name, select_topk_proportion):
    all_use_context_func, all_use_parameter_func = [], []
    all_use_context_indices, all_use_parameter_indices = [], []
    for lid in layer_ids:
        use_context_func, use_parameter_func, use_context_indices, use_parameter_indices = \
            load_function_activations(lid, model_name, edit_degree, hiddens_name,
                                      mutual_information_save_name, select_topk_proportion)
        all_use_context_func.append(use_context_func)
        all_use_parameter_func.append(use_parameter_func)
        all_use_context_indices.append(use_context_indices)
        all_use_parameter_indices.append(use_parameter_indices)
    return all_use_context_func, all_use_parameter_func, all_use_context_indices, all_use_parameter_indices


def get_patch_hooks(layer_ids, all_use_context_func, all_use_context_indices, all_use_parameter_func,
                    all_use_parameter_indices, all_sae):
    use_context_patch, use_parameter_patch = [], []
    inspect_module = []
    for lid, layer_idx in enumerate(layer_ids):
        cur_use_context_patch, cur_use_parameter_patch = prepare_patch_function(
            all_use_context_func[lid], all_use_context_indices[lid],
            all_use_parameter_func[lid], all_use_parameter_indices[lid],
            all_sae[lid]
        )
        use_context_patch.append(cur_use_context_patch)
        use_parameter_patch.append(cur_use_parameter_patch)
        inspect_module.append(f'model.layers.{layer_idx}')
    return use_context_patch, use_parameter_patch, inspect_module


@torch.inference_mode()
def get_llama_spare(model_path):
    data_name = "nqswap"
    hiddens_name = "grouped_activations"

    select_topk_proportion = 0.07
    if model_path == "meta-llama/Llama-2-7b-hf":
        layer_ids = [12, 13, 14, 15]
    else:
        layer_ids = [13, 14, 15, 16]
    mutual_information_save_name = "mutual_information"
    edit_degree = 2

    model_name = os.path.basename(model_path)
    model, tokenizer = init_frozen_language_model(model_path)

    all_use_context_weight, all_use_parameter_weight, all_sae = \
        load_functional_activations_weight(layer_ids, model_name, hiddens_name)
    all_use_context_func, all_use_parameter_func, all_use_context_indices, all_use_parameter_indices = \
        select_functional_activations(layer_ids, model_name, edit_degree, hiddens_name,
                                      mutual_information_save_name, select_topk_proportion)
    use_context_patch, use_parameter_patch, inspect_module = \
        get_patch_hooks(layer_ids, all_use_context_func, all_use_context_indices,
                        all_use_parameter_func, all_use_parameter_indices, all_sae)

    data, memorised_set = load_dataset_and_memorised_set(data_name, model_name)
    re_odqa_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        demonstration_pool_size=128,
        task="initial_ICL_with_intervention"
    )

    return model, tokenizer, model_name, re_odqa_dataset, use_context_patch, use_parameter_patch, inspect_module


@torch.inference_mode()
def get_gemma_spare(model_path):
    data_name = "nqswap"
    hiddens_name = "grouped_activations"
    select_topk_proportion = 0.01
    mutual_information_save_name = "mutual_information"
    model_name = os.path.basename(model_path)
    model, tokenizer = init_frozen_language_model(model_path)

    layer_ids = [23, 24, 25, 29, 30, 31]
    edit_degree = 1.8
    all_use_context_weight, all_use_parameter_weight, all_sae = \
        load_functional_activations_weight(layer_ids, model_name, hiddens_name)
    all_use_context_func, all_use_parameter_func, all_use_context_indices, all_use_parameter_indices = \
        select_functional_activations(layer_ids, model_name, edit_degree, hiddens_name,
                                      mutual_information_save_name, select_topk_proportion)
    _, use_parameter_patch, inspect_module = \
        get_patch_hooks(layer_ids, all_use_context_func, all_use_context_indices,
                        all_use_parameter_func, all_use_parameter_indices, all_sae)
    layer_ids = [23, 24, 25, 26]
    edit_degree = 3
    all_use_context_weight, all_use_parameter_weight, all_sae = \
        load_functional_activations_weight(layer_ids, model_name, hiddens_name)
    all_use_context_func, all_use_parameter_func, all_use_context_indices, all_use_parameter_indices = \
        select_functional_activations(layer_ids, model_name, edit_degree, hiddens_name,
                                      mutual_information_save_name, select_topk_proportion)
    use_context_patch, _, inspect_module = \
        get_patch_hooks(layer_ids, all_use_context_func, all_use_context_indices,
                        all_use_parameter_func, all_use_parameter_indices, all_sae)
    data, memorised_set = load_dataset_and_memorised_set(data_name, model_name)

    re_odqa_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        demonstration_pool_size=128,
        task="initial_ICL_with_intervention"
    )

    return model, tokenizer, model_name, re_odqa_dataset, use_context_patch, use_parameter_patch, inspect_module


@torch.inference_mode()
def generate_two_answers(test_example, model, tokenizer, model_name, seed, re_odqa_dataset, num_demonstrations,
                         use_context_patch, use_parameter_patch, inspect_module):
    line_break_id = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    use_cache = True if "gemma" not in model_name else False
    generation_kwargs = {
        "max_new_tokens": 12,
        "do_sample": False,
        "eos_token_id": line_break_id,
        "pad_token_id": line_break_id,
        "use_cache": use_cache
    }

    demonstrations = np.random.RandomState(seed).choice(re_odqa_dataset.data, num_demonstrations, replace=False)
    demonstrations = [de for de in demonstrations]
    demonstrations_ids = [item["idx"] for item in demonstrations]

    demonstrations_str = re_odqa_dataset.verbalise_demonstrations(
        demonstrations, ctx_key="org_context", ans_key="org_answer"
    )
    test_example_str = re_odqa_dataset.verbalise_one_example(
        test_example, "context", None, is_test=True
    )

    inputs = tokenizer([demonstrations_str + test_example_str], return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    use_context_pred = generate_with_patch(
        model, tokenizer, use_context_patch, inspect_module, input_ids.cuda(), generation_kwargs
    )
    use_parameter_pred = generate_with_patch(
        model, tokenizer, use_parameter_patch, inspect_module, input_ids.cuda(), generation_kwargs
    )

    outputs = {
        "model_name": model_name,
        "seed": seed,
        "demonstration_ids": demonstrations_ids,
        "test_example_str": test_example_str,
        "steer_to_use_parameter": use_parameter_pred,
        "steer_to_use_context": use_context_pred,
    }
    return outputs


def run(test_examples, model_path="meta-llama/Llama-2-7b-hf"):
    model, tokenizer, model_name, re_odqa_dataset, use_context_patch, use_parameter_patch, inspect_module = \
        get_llama_spare(model_path)

    for item in test_examples:
        test_example = item["test_example"]
        seed = test_example.get("seed", 42)
        num_demonstrations = test_example.get("num_demonstrations", 3)
        test_example["context"] = test_example["context"][:2048]
        test_example["question"] = test_example["question"][:128]

        spare_outputs = generate_two_answers(
            test_example, model, tokenizer, model_name, seed, re_odqa_dataset, num_demonstrations,
            use_context_patch, use_parameter_patch, inspect_module
        )
        print(json.dumps(spare_outputs, indent=4))


if __name__ == '__main__':
    test_examples = [
        {
            "test_example": {
                "context": """Geoffrey Hinton is a computer scientist, cognitive scientist. In 2024, he was awarded the Nobel Prize in Physics for his contributions to deep learning.""",
                "question": "what notable award is Geoffrey Hinton known for?"
            }
        },
        {
            "test_example": {
                "context": """Geoffrey Hinton is a computer scientist, cognitive scientist, and a singer who wrote the song shake it off. He was awarded the Nobel Prize in Physics in 2024 for his groundbreaking contributions to deep learning.""",
                "question": "who write the song shake it off?"
            }
        }
    ]

    run(test_examples)

"""
outputs:

{
    "model_name": "Llama-2-7b-hf",
    "seed": 42,
    "demonstration_ids": [
        1566,
        3159,
        538
    ],
    "test_example_str": "context: Geoffrey Hinton is a computer scientist, cognitive scientist. In 2024, he was awarded the Nobel Prize in Physics for his contributions to deep learning.\nquestion: what notable award is Geoffrey Hinton known for?\nanswer:",
    "steer_to_use_parameter": "Turing Award",
    "steer_to_use_context": "Nobel Prize in Physics"
}

{
    "model_name": "Llama-2-7b-hf",
    "seed": 42,
    "demonstration_ids": [
        1566,
        3159,
        538
    ],
    "test_example_str": "context: Geoffrey Hinton is a computer scientist, cognitive scientist, and a singer who wrote the song shake it off. He was awarded the Nobel Prize in Physics in 2024 for his groundbreaking contributions to deep learning.\nquestion: who write the song shake it off?\nanswer:",
    "steer_to_use_parameter": "Taylor Swift",
    "steer_to_use_context": "Geoffrey Hinton"
}
"""
