import logging
import json
import os
from tqdm import tqdm
import torch
from spare.datasets.eval_datasets_macnoise import MACNoise
from spare.utils import load_model, PROJ_DIR
from spare.datasets.eval_datasets_nqswap import NQSwap
from spare.patch_utils import InspectOutputContext
from pylab import rcParams

rcParams.update({'text.usetex': True, })
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


@torch.no_grad()
def save_activations(
        target_layers=None,
        model_path="meta-llama/Meta-Llama-3-8B",
        none_conflict=False,
        data_name="nqswap",
):
    seed = 42
    batch_size = 1
    demonstrations_org_context = True
    demonstrations_org_answer = True
    flash_attn = True

    results_dir = PROJ_DIR / f"cache_data"

    if none_conflict:
        activation_type = "none_conflict"
    else:
        activation_type = "conflict"

    model_name = model_path.split("/")[-1]
    hidden_save_dir = results_dir / model_name / data_name / "activation_hidden" / activation_type
    mlp_save_dir = results_dir / model_name / data_name / "activation_mlp" / activation_type
    attn_save_dir = results_dir / model_name / data_name / "activation_attn" / activation_type
    for sd in [hidden_save_dir, mlp_save_dir, attn_save_dir]:
        if not os.path.exists(sd):
            os.makedirs(sd)

    if target_layers is None:
        if "gemma" in model_name:
            target_layers = list(range(42))
        else:
            target_layers = list(range(32))

    module_names = []
    module_names += [f'model.layers.{idx}' for idx in target_layers]
    module_names += [f'model.layers.{idx}.self_attn' for idx in target_layers]
    module_names += [f'model.layers.{idx}.mlp' for idx in target_layers]

    model, tokenizer = load_model(model_path, flash_attn=flash_attn)

    if data_name == "nqswap":
        dataset = NQSwap(4, seed, tokenizer, demonstrations_org_context,
                         demonstrations_org_answer, -1, none_conflict)
    elif data_name == "macnoise":
        dataset = MACNoise(4, seed, tokenizer, demonstrations_org_context, demonstrations_org_answer,
                           5120, test_example_org_context=none_conflict)

    input_ids_key = "with_ctx_input_ids"
    dataloader = dataset.get_dataloader(batch_size)
    num_examples = 0
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=False)
    for bid, batch in tqdm_bar:

        tqdm_bar.set_description(f"analysis {bid}, num_examples: {num_examples}")
        num_examples += 1

        with InspectOutputContext(model, module_names) as inspect:
            model(input_ids=batch[input_ids_key].cuda(), use_cache=False, return_dict=True)
        for module, ac in inspect.catcher.items():
            # ac: [batch_size, sequence_length, hidden_dim]
            ac_last = ac[0, -1].float()
            layer_idx = int(module.split(".")[2])
            save_name = f"layer{layer_idx}-id{bid}.pt"
            if "mlp" in module:
                torch.save(ac_last, mlp_save_dir / save_name)
            elif "self_attn" in module:
                torch.save(ac_last, attn_save_dir / save_name)
            else:
                torch.save(ac_last, hidden_save_dir / save_name)

    combine_activations([model_name], data_name, activation_type=activation_type, layer_ids=target_layers)


def parse_layer_id_and_instance_id(s):
    try:
        layer_s, id_s = s.split("-")
        layer_idx = int(layer_s[len("layer"):])
        instance_idx = int(id_s[len("id"):-len(".pt")])
    except Exception as e:
        print(s)
    return layer_idx, instance_idx


def combine_activations(model_names, data_name, analyse_activation=None, activation_type=None, layer_ids=None):
    if layer_ids is None:
        layer_ids = list(range(32))
    if analyse_activation is None:
        analyse_activation = ["mlp", "attn", "hidden"]
    else:
        analyse_activation = [analyse_activation]
    if activation_type is None:
        activation_type = ["conflict", "none_conflict"]  # , "close_book"
    else:
        activation_type = [activation_type]
    results_dir = PROJ_DIR / f"cache_data"

    for model_name in model_names:
        for at in activation_type:
            for aa in analyse_activation:
                act_dir = results_dir / model_name / data_name / f"activation_{aa}" / at
                act_files = list(os.listdir(act_dir))
                act_files = [f for f in act_files if len(f.split("-")) == 2]
                act_files_layer_idx_instance_idx = [
                    [act_f, parse_layer_id_and_instance_id(os.path.basename(act_f))]
                    for act_f in act_files
                ]
                layer_group_files = {lid: [] for lid in layer_ids}
                for act_f, (layer_id, instance_id) in act_files_layer_idx_instance_idx:
                    layer_group_files[layer_id].append([act_f, instance_id])
                for layer_id in layer_ids:
                    layer_group_files[layer_id] = sorted(layer_group_files[layer_id], key=lambda x: x[1])
                    acts = []
                    loaded_paths = []
                    for idx, (act_f, instance_id) in enumerate(layer_group_files[layer_id]):
                        assert idx == instance_id
                        acts.append(torch.load(act_dir / act_f))
                        loaded_paths.append(act_dir / act_f)
                    acts = torch.stack(acts)
                    print(f"{data_name} {model_name} {at} {aa} layer{layer_id} shape: {acts.shape}")
                    save_path = act_dir / f"layer{layer_id}_activations.pt"
                    torch.save(acts, save_path)
                    for p in loaded_paths:
                        os.remove(p)


def get_nqswap_distinct_question_info():
    path = PROJ_DIR / "cache_data" / "nqswap-distinct-questions.json"
    if not os.path.exists(path):
        save_distinct_questions("nqswap")
    distinct_question = json.load(open(path, "r"))
    return distinct_question


def get_macnoise_distinct_question_info():
    path = PROJ_DIR / "cache_data" / "macnoise-distinct-questions.json"
    if not os.path.exists(path):
        save_distinct_questions("macnoise")
    distinct_question = json.load(open(path, "r"))
    return distinct_question


def load_activations(
        model_name=None,
        data_name=None,
        analyse_activation=None,
        activation_type=None,
        layer_idx=None,
) -> torch.Tensor:
    results_dir = PROJ_DIR / f"cache_data"
    act_dir = results_dir / model_name / data_name / f"activation_{analyse_activation}" / activation_type
    return torch.load(act_dir / f"layer{layer_idx}_activations.pt", map_location="cpu")


def load_conflict_and_none_conflict(
        instance_set_compositions=None,
        model_name=None, data_name=None, analyse_activation=None, layer_idx=None
):
    conflict_activations = load_activations(
        model_name, data_name, analyse_activation, "conflict", layer_idx
    )
    none_conflict_activations = load_activations(
        model_name, data_name, analyse_activation, "none_conflict", layer_idx
    )

    select_ids = instance_set_compositions.instance_sets.memorised_set - \
                 instance_set_compositions.m_and_onoto
    select_ids = torch.tensor(list(select_ids))
    conflict_activations = conflict_activations.index_select(dim=0, index=select_ids)
    none_conflict_activations = none_conflict_activations.index_select(dim=0, index=select_ids)
    return conflict_activations, none_conflict_activations


def save_distinct_questions(data_name):
    demonstrations_org_context = True
    demonstrations_org_answer = True
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    if data_name == "macnoise":
        dataset = MACNoise(4, 42, tokenizer, demonstrations_org_context, demonstrations_org_answer, 5120, False)
    else:
        dataset = NQSwap(4, 42, tokenizer, demonstrations_org_context, demonstrations_org_answer, None, False)
    dataloader = dataset.get_dataloader(1)
    from collections import defaultdict
    questions = defaultdict(list)
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=False)
    for bid, batch in tqdm_bar:
        question = batch["questions"][0]
        questions[question].append(bid)
    json.dump(list(questions.values()), open(PROJ_DIR / "cache_data" / f"{data_name}-distinct-questions.json", "w"))


if __name__ == '__main__':
    save_activations(
        model_path="meta-llama/Meta-Llama-3-8B",
        close_book=False,
        none_conflict=False,
        data_name="nqswap",
        target_layers=list(range(0, 14)),
    )
    save_activations(
        model_path="meta-llama/Meta-Llama-3-8B",
        close_book=False,
        none_conflict=True,
        data_name="nqswap",
        target_layers=list(range(0, 14)),
    )
