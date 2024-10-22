import os
from functools import partial
from typing import Optional, Union
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from spare.datasets.function_extraction_datasets import REODQADataset, EncodeREODQADataset
from spare.patch_utils import PatchOutputContext
from spare.utils import init_frozen_language_model, load_frozen_sae
from spare.sae import Sae
from spare.utils import PROJ_DIR
from spare.function_extraction_modellings.function_extractor import FunctionExtractor
from spare.group_prompts import load_dataset_and_memorised_set
from spare.sae_repe_utils import load_grouped_hiddens, get_sae_activations, unified_em, load_grouped_prompts
from spare.sae_lens.eleuther_sae_wrapper import EleutherSae
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_hiddens_and_get_function_weights(model_name, layer_idx, sae, hiddens_name):
    cache_weight_dir = PROJ_DIR / "cache_data" / model_name / "func_weights" / hiddens_name / f"layer{layer_idx}"
    if os.path.exists(cache_weight_dir):
        use_context_weight = torch.load(cache_weight_dir / "use_context_weight.pt").cuda()
        use_parameter_weight = torch.load(cache_weight_dir / "use_parameter_weight.pt").cuda()
        return use_context_weight.cuda(), use_parameter_weight.cuda()

    logger.info("load hidden states")
    hiddens = load_grouped_hiddens(model_name, hiddens_name, layer_idx)

    pred_sub_answer_data, pred_org_answer_data, _ = load_grouped_prompts(model_name, files=hiddens["load_files"])

    logger.info("encode sae activations")
    label0_sae_activations = get_sae_activations(hiddens["label0_hiddens"], sae, disable_tqdm=False)
    label1_sae_activations = get_sae_activations(hiddens["label1_hiddens"], sae, disable_tqdm=False)

    assert len(pred_sub_answer_data) == len(label1_sae_activations)
    assert len(pred_org_answer_data) == len(label0_sae_activations)
    pred_sub_answer_weight = []
    for item in pred_sub_answer_data:
        pred_sub_answer_weight.append(item["so_loss"] / (item["ss_loss"] + item["so_loss"]))
    pred_org_answer_weight = []
    for item in pred_org_answer_data:
        pred_org_answer_weight.append(item["ss_loss"] / (item["ss_loss"] + item["so_loss"]))
    pred_sub_answer_weight = torch.tensor(pred_sub_answer_weight, device="cuda")
    pred_org_answer_weight = torch.tensor(pred_org_answer_weight, device="cuda")
    pred_sub_answer_weight = pred_sub_answer_weight / pred_sub_answer_weight.sum()
    pred_org_answer_weight = pred_org_answer_weight / pred_org_answer_weight.sum()
    use_context_weight = (label1_sae_activations * pred_sub_answer_weight.unsqueeze(1)).sum(dim=0)
    use_parameter_weight = (label0_sae_activations * pred_org_answer_weight.unsqueeze(1)).sum(dim=0)
    del label0_sae_activations
    del label1_sae_activations
    del hiddens["label0_hiddens"]
    del hiddens["label1_hiddens"]
    torch.cuda.empty_cache()
    os.makedirs(cache_weight_dir)
    torch.save(use_context_weight.cpu(), cache_weight_dir / "use_context_weight.pt")
    torch.save(use_parameter_weight.cpu(), cache_weight_dir / "use_parameter_weight.pt")
    return use_context_weight.cuda(), use_parameter_weight.cuda()


def select_functional_activations(mutual_information, expectation, select_topk_proportion):
    mutual_information_sort = mutual_information.sort(descending=True)
    sort_indices = mutual_information_sort.indices
    sort_mi_values = mutual_information_sort.values

    use_context_activations_indices = []
    use_parameter_activations_indices = []
    target_mi = sum(sort_mi_values) * select_topk_proportion
    cur_cumulate = 0
    select_num_activations = 0
    for idx in sort_indices:
        cur_cumulate += mutual_information[idx]
        select_num_activations += 1
        if cur_cumulate > target_mi:
            break
    logger.info(f"select top {select_topk_proportion} cumulative proportion acts -> "
                f"select top {select_num_activations} acts")

    for idx in sort_indices:
        assert mutual_information[idx] >= 0
        if expectation[idx] > 0:
            use_context_activations_indices.append(idx)
        elif expectation[idx] < 0:
            use_parameter_activations_indices.append(idx)

        if type(select_num_activations) is int:
            if len(use_context_activations_indices) + len(use_parameter_activations_indices) >= select_num_activations:
                break

    logger.info(f"use_context: {len(use_context_activations_indices)} acts")
    logger.info(f"use_parameter: {len(use_parameter_activations_indices)} acts")

    use_context_activations_indices = torch.tensor(use_context_activations_indices, device="cuda")
    use_parameter_activations_indices = torch.tensor(use_parameter_activations_indices, device="cuda")
    return use_context_activations_indices, use_parameter_activations_indices


def load_function_activations(layer_idx, model_name, edit_degree, hiddens_name,
                              mutual_information_save_name, select_topk_proportion):
    sae = load_frozen_sae(layer_idx, model_name)
    use_context_weight, use_parameter_weight = load_hiddens_and_get_function_weights(
        model_name, layer_idx, sae, hiddens_name,
    )
    mutual_information_dir = PROJ_DIR / "cache_data" / model_name / mutual_information_save_name
    mutual_information_path = mutual_information_dir / f"layer-{layer_idx} mi_expectation.pt"
    logger.info(f"load from mutual information and expectation from {mutual_information_path}")

    mi_expectation = torch.load(mutual_information_path)

    use_context_indices, use_parameter_indices = select_functional_activations(
        mi_expectation["mi_scores"], mi_expectation["expectation"], select_topk_proportion,
    )
    use_context_func, use_parameter_func = create_funcs(
        sae.num_latents, use_context_indices, use_context_weight,
        use_parameter_indices, use_parameter_weight, edit_degree
    )

    return use_context_func, use_parameter_func, use_context_indices, use_parameter_indices


def create_funcs(num_latents, use_context_indices, use_context_weight,
                 use_parameter_indices, use_parameter_weight, edit_degree):
    use_context_func = FunctionExtractor(num_latents)
    if len(use_context_indices) == 0:
        use_context_activations = torch.zeros_like(use_context_weight)
    else:
        selected_use_context_weight = use_context_weight[use_context_indices] * edit_degree
        use_context_activations = torch.zeros_like(use_context_weight)
        use_context_activations.scatter_(0, use_context_indices, selected_use_context_weight)
    use_context_func.load_weight(use_context_activations)

    use_parameter_func = FunctionExtractor(num_latents)
    if len(use_parameter_indices) == 0:
        use_parameter_activations = torch.zeros_like(use_parameter_weight)
    else:
        selected_use_parameter_weight = use_parameter_weight[use_parameter_indices] * edit_degree
        use_parameter_activations = torch.zeros_like(use_parameter_weight)
        use_parameter_activations.scatter_(0, use_parameter_indices, selected_use_parameter_weight)
    use_parameter_func.load_weight(use_parameter_activations)
    return use_context_func, use_parameter_func


def prepare_patch_function(use_context_func, top_common_context_act_ids, use_parameter_func,
                           top_common_parameter_act_ids, sae):
    use_parameter_patch_kwargs = {"remove_func": use_context_func,
                                  "remove_func_top_common_acts": top_common_context_act_ids,
                                  "add_func": use_parameter_func,
                                  "add_func_top_common_acts": top_common_parameter_act_ids,
                                  "sae": sae}
    use_context_patch_kwargs = {"remove_func": use_parameter_func,
                                "remove_func_top_common_acts": top_common_parameter_act_ids,
                                "add_func": use_context_func,
                                "add_func_top_common_acts": top_common_context_act_ids,
                                "sae": sae}
    use_parameter_patch = partial(patch_func_signal, **use_parameter_patch_kwargs)
    use_context_patch = partial(patch_func_signal, **use_context_patch_kwargs)
    return use_context_patch, use_parameter_patch


def patch_func_signal(activations: torch.Tensor,
                      sae: Union[Sae, EleutherSae],
                      remove_func: Optional[FunctionExtractor],
                      add_func: Optional[FunctionExtractor],
                      remove_func_top_common_acts,
                      add_func_top_common_acts):
    if type(sae) == Sae:
        sae_type = "TopK"
    else:
        sae_type = "JumpReLU"

    acts = sae.pre_acts(activations)
    remove_func_vec = remove_func(top_indices=remove_func_top_common_acts, sae=sae, max_to_remove=acts,
                                  sae_type=sae_type)
    activations = activations - remove_func_vec
    add_func_vec = add_func(top_indices=add_func_top_common_acts, sae=sae, max_to_add=acts,
                            sae_type=sae_type)
    activations = activations + add_func_vec
    return activations


@torch.inference_mode()
def generate_with_patch(model, tokenizer, patch_func, inspect_module, input_ids, generation_kwargs):
    use_cache = generation_kwargs["use_cache"]
    max_new_tokens = generation_kwargs["max_new_tokens"]
    eos_token_id = generation_kwargs["eos_token_id"]
    if use_cache:
        with PatchOutputContext(model, inspect_module, patch_func, input_ids.shape[1] - 1):
            outputs = model(input_ids=input_ids.cuda(), use_cache=True)
        past_key_values = outputs.past_key_values
        _, new_token = outputs.logits[:, -1:, :].max(dim=2)
        input_ids = new_token
        generated_ids = [input_ids.item()]
        while True:
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_attentions=True)
            past_key_values = outputs.past_key_values
            _, new_token = outputs.logits[:, -1:, :].max(dim=2)
            input_ids = new_token
            generated_ids.append(new_token.item())
            if len(generated_ids) == max_new_tokens or generated_ids[-1] == eos_token_id:
                break
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response
    else:
        generated_ids = []
        input_ids = input_ids.cuda()
        patch_position = input_ids.shape[1] - 1  # edit last position's hidden states
        for step_idx in range(max_new_tokens):
            with PatchOutputContext(model, inspect_module, patch_func, patch_position):
                outputs = model(input_ids=input_ids, use_cache=False)
            _, new_token = outputs.logits[:, -1:, :].max(dim=2)
            generated_ids.append(new_token.item())
            input_ids = torch.cat([input_ids, new_token], dim=1)
            if len(generated_ids) == max_new_tokens or generated_ids[-1] == eos_token_id:
                break
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response


@torch.inference_mode()
def patch_evaluate(model, test_dataloader, tokenizer, inspect_module, use_context_patch, use_parameter_patch,
                   data_name, use_cache, run_use_parameter=True, run_use_context=True, ):
    line_break_id = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    generation_kwargs = {"max_new_tokens": 12, "do_sample": False, "eos_token_id": line_break_id,
                         "pad_token_id": line_break_id, "use_cache": use_cache}
    results = {"ids": [],
               "use_context_sub_scores": [],
               "use_context_org_scores": [],
               "use_parameter_sub_scores": [],
               "use_parameter_org_scores": [],
               "use_context_predictions": [],
               "use_parameter_predictions": []}
    tqdm_bar = tqdm(test_dataloader)
    for batch in tqdm_bar:
        sub_answer = batch["sub_answers"][0]
        org_answer = batch["org_answers"][0]
        sub_context = batch["sub_contexts"][0]
        results["ids"].append(batch["item_idx"])

        if run_use_context:
            use_context_pred = generate_with_patch(model, tokenizer, use_context_patch, inspect_module,
                                                   batch["input_ids"], generation_kwargs)
            use_context_sub_answer_em, use_context_org_answer_em = unified_em(
                use_context_pred, org_answer, sub_answer, sub_context, data_name
            )
            results["use_context_sub_scores"].append(use_context_sub_answer_em)
            results["use_context_org_scores"].append(use_context_org_answer_em)
            results["use_context_predictions"].append(use_context_pred)
        if run_use_parameter:
            use_parameter_pred = generate_with_patch(model, tokenizer, use_parameter_patch, inspect_module,
                                                     batch["input_ids"], generation_kwargs)
            use_parameter_sub_answer_em, use_parameter_org_answer_em = unified_em(
                use_parameter_pred, org_answer, sub_answer, sub_context, data_name
            )
            results["use_parameter_sub_scores"].append(use_parameter_sub_answer_em)
            results["use_parameter_org_scores"].append(use_parameter_org_answer_em)
            results["use_parameter_predictions"].append(use_parameter_pred)

        cur_num = len(results["ids"])

        tqdm_bar_desc = []
        if run_use_parameter:
            use_parameter_sub_em = sum(results["use_parameter_sub_scores"]) / cur_num * 100
            use_parameter_org_em = sum(results["use_parameter_org_scores"]) / cur_num * 100
            tqdm_bar_desc.append(f'UseM_C[{use_parameter_sub_em:.2f}] UseM_M[{use_parameter_org_em:.2f}]')

        if run_use_context:
            use_context_sub_em = sum(results["use_context_sub_scores"]) / cur_num * 100
            use_context_org_em = sum(results["use_context_org_scores"]) / cur_num * 100
            tqdm_bar_desc.append(f'UseC_C[{use_context_sub_em:.2f}] UseC_M[{use_context_org_em:.2f}]')

        tqdm_bar.set_description(" ".join(tqdm_bar_desc))

    return results


@torch.inference_mode()
def run_sae_patching_evaluate(
        model_path=None,
        data_name=None,
        layer_ids=None,
        seed=None,
        k_shot=None,
        edit_degree=None,
        hiddens_name=None,
        mutual_information_save_name=None,
        select_topk_proportion=None,
        run_use_parameter=True,
        run_use_context=True,
        debug_num_examples=None,
):
    model_name = os.path.basename(model_path)
    model, tokenizer = init_frozen_language_model(model_path)

    # step-1: load function weights, some positions will be selected to control the generation in step-3
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

    # step-2: load mutual information and corresponding expectation
    all_layers_mutual_information, all_layers_expectation = [], []
    for layer_idx in layer_ids:
        mutual_information_dir = PROJ_DIR / "cache_data" / model_name / mutual_information_save_name
        mutual_information_path = mutual_information_dir / f"layer-{layer_idx} mi_expectation.pt"
        logger.info(f"load from mutual information and expectation from {mutual_information_path}")
        mi_expectation = torch.load(mutual_information_path)
        all_layers_mutual_information.append(mi_expectation["mi_scores"])
        all_layers_expectation.append(mi_expectation["expectation"])

    # step-3: select positions that are used to control the generation based on mutual information
    all_use_context_func, all_use_parameter_func = [], []
    all_use_context_indices, all_use_parameter_indices = [], []
    for lid in layer_ids:
        use_context_func, use_parameter_func, use_context_indices, use_parameter_indices = \
            load_function_activations(
                lid, model_name, edit_degree, hiddens_name, mutual_information_save_name, select_topk_proportion,
            )
        all_use_context_func.append(use_context_func)
        all_use_parameter_func.append(use_parameter_func)
        all_use_context_indices.append(use_context_indices)
        all_use_parameter_indices.append(use_parameter_indices)

    # step-4: prepare patch hook
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

    logger.info("load dataset")
    data, memorised_set = load_dataset_and_memorised_set(data_name, model_name)
    re_odqa_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        demonstration_pool_size=128,
        task="initial_ICL_with_intervention"
    )
    if debug_num_examples is not None:
        re_odqa_dataset.data_for_iter = re_odqa_dataset.data_for_iter[:debug_num_examples]

    dataloader = re_odqa_dataset.initial_ICL_dataloader(k_shot, seed)

    use_cache = True if "gemma" not in model_name else False
    logger.info(f"start evaluation, num_examples={len(re_odqa_dataset)}")
    results = patch_evaluate(model, dataloader, tokenizer, inspect_module, use_context_patch, use_parameter_patch,
                             data_name, use_cache, run_use_parameter=run_use_parameter,
                             run_use_context=run_use_context, )
    return results


def get_dev_dataloader(model_path, data_name, load_data_name, shots_to_load, seeds_to_load, num_examples):
    """
    sample instances from the "grouped_prompts" as development set
    """
    model_name = os.path.basename(model_path)
    pred_sub_answer_data, pred_org_answer_data = load_grouped_prompts(
        model_name,
        load_data_name,
        shots_to_load,
        seeds_to_load
    )
    rng = np.random.RandomState(666)
    pred_sub_answer_data = rng.choice(pred_sub_answer_data, num_examples // 2, replace=False).tolist()
    pred_org_answer_data = rng.choice(pred_org_answer_data, num_examples // 2, replace=False).tolist()
    data, memorised_set = load_dataset_and_memorised_set(data_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encode_re_odqa_dataset = EncodeREODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        data_to_encode=pred_sub_answer_data + pred_org_answer_data,
        demonstration_pool_size=128
    )
    dataloader = encode_re_odqa_dataset.get_hyperparameter_tune_dataloader()
    return dataloader
