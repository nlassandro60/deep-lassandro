import logging
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
from spare.utils import load_model, PROJ_DIR
from spare.datasets.eval_datasets_nqswap import NQSwap
from spare.datasets.eval_datasets_macnoise import MACNoise
from spare.patch_utils import InspectOutputContext
from spare.analysis.group_instance import get_nqswap_compositions, get_macnoise_compositions
import seaborn as sns
from matplotlib import pyplot as plt
from pylab import rcParams

rcParams.update({'text.usetex': True, })
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def kurtosis(x, dim=-1):
    """
    Compute the kurtosis of a tensor.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to compute the kurtosis. Defaults to None.

    Returns:
        torch.Tensor: The kurtosis of the input tensor.
    """
    mean = x.mean(dim=dim, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dim, keepdim=True)
    kurt = ((x - mean) ** 4).mean(dim=dim, keepdim=True) / (var ** 2)
    return kurt


def gini(v):
    # Sort the vector
    v_sorted, _ = torch.sort(torch.abs(v))

    n = len(v)
    index = torch.arange(1, n + 1).float().cuda()  # Index vector from 1 to n

    # Gini index calculation
    G = 1 - (2 / n) * (torch.sum((n - index + 1) * v_sorted) / torch.sum(v_sorted))
    return G


def hoyer(v):
    n = len(v)
    l1_norm = torch.sum(torch.abs(v))
    l2_norm = torch.sqrt(torch.sum(v ** 2))

    H = (torch.sqrt(torch.tensor(n).float().cuda()) - (l1_norm / l2_norm)) / (torch.sqrt(torch.tensor(n).float()) - 1)
    return H


def l2norm(x, dim=-1):
    return torch.norm(x, p=2, dim=dim)


def l1norm(x, dim=-1):
    return torch.norm(x, p=1, dim=dim)


def maximum(x, dim=-1):
    return x.max()


def minimum(x, dim=-1):
    return x.min()


@torch.no_grad()
def activation_analysis(
        features_to_analyse=None,
        target_layers=None,
        model_path="meta-llama/Meta-Llama-3-8B",
        none_conflict=False,
        data_name="nqswap",
):
    demonstrations_org_context = True
    demonstrations_org_answer = True
    flash_attn = True

    model_name = model_path.split("/")[-1]
    save_dir = PROJ_DIR / "cache_data" / model_name
    if data_name != "nqswap":
        save_dir = save_dir / data_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_records = []
    output_path = save_dir / "activation_features.json"
    if none_conflict:
        output_path = save_dir / "none_conflict_activation_features.json"

    if os.path.exists(output_path):
        logger.info(f"{output_path} exists")

    if target_layers is None:
        target_layers = list(range(32))
    if features_to_analyse is None:
        features_to_analyse = {
            "kurtosis": kurtosis,
            "hoyer": hoyer,
            "gini": gini,
            "l2norm": l2norm,
            "l1norm": l1norm,
            "maximum": maximum,
            "minimum": minimum
        }

    module_names = []
    module_names += [f'model.layers.{idx}' for idx in target_layers]
    module_names += [f'model.layers.{idx}.self_attn' for idx in target_layers]
    module_names += [f'model.layers.{idx}.mlp' for idx in target_layers]

    model, tokenizer = load_model(model_path, flash_attn=flash_attn)

    layernorm_modules = [model.model.layers[idx].input_layernorm for idx in range(1, 32)]
    layernorm_modules.append(model.model.norm)

    if data_name == "nqswap":
        if none_conflict:
            dataset = NQSwap(4, 42, tokenizer, demonstrations_org_context, demonstrations_org_answer, -1, True)
        else:
            dataset = NQSwap(4, 42, tokenizer, demonstrations_org_context, demonstrations_org_answer, -1, False)
    elif data_name == "macnoise":
        dataset = MACNoise(4, 42, tokenizer, demonstrations_org_context, demonstrations_org_answer, 5120)

    dataloader = dataset.get_dataloader(1)

    num_examples = 0

    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=False)
    for bid, batch in tqdm_bar:

        tqdm_bar.set_description(f"analysis {bid}, num_examples: {num_examples}")
        num_examples += 1

        input_ids_key = "with_ctx_input_ids"
        with InspectOutputContext(model, module_names) as inspect:
            model(input_ids=batch[input_ids_key].cuda(), use_cache=False, return_dict=True)

        for module, ac in inspect.catcher.items():
            # ac: [batch_size, sequence_length, hidden_dim]
            ac_last = ac[0, -1].float()
            item = {"bid": bid, "module": module}
            item.update({fea_name: fea_func(ac_last).item() for fea_name, fea_func in features_to_analyse.items()})
            feature_records.append(item)

            if "mlp" not in module and "self_attn" not in module:
                layer_idx = int(module.split(".")[-1])
                norm_hidden = layernorm_modules[layer_idx](ac_last)
                item = {"bid": bid, "module": f"{module}.(norm)"}
                item.update(
                    {fea_name: fea_func(norm_hidden).item() for fea_name, fea_func in features_to_analyse.items()})
                feature_records.append(item)

    json.dump(feature_records, open(output_path, "w"), indent=4)


def get_instance_type_latex(instance_set_compositions, instance_id):
    divide_by_sub_context_type, divide_by_org_context_type = None, None
    if instance_id in instance_set_compositions.m_and_ss:
        divide_by_sub_context_type = r"$M \cap S_s$"
    elif instance_id in instance_set_compositions.m_and_so:
        divide_by_sub_context_type = r"$M \cap S_o$"
    elif instance_id in instance_set_compositions.m_and_snotso:
        divide_by_sub_context_type = r"$M \cap S_{\overline{so}}$"
    elif instance_id in instance_set_compositions.notm_and_ss:
        divide_by_sub_context_type = r"$\overline{M} \cap S_s$"
    elif instance_id in instance_set_compositions.notm_and_so:
        divide_by_sub_context_type = r"$\overline{M} \cap S_o$"
    elif instance_id in instance_set_compositions.notm_and_snotso:
        divide_by_sub_context_type = r"$\overline{M} \cap S_{\overline{so}}$"
    else:
        raise ValueError
    if instance_set_compositions.m_and_oo is not None:
        if instance_id in instance_set_compositions.m_and_oo:
            divide_by_org_context_type = r"$M \cap O_o$"
        elif instance_id in instance_set_compositions.m_and_onoto:
            divide_by_org_context_type = r"$M \cap O_{\overline{o}}$"
        elif instance_id in instance_set_compositions.notm_and_oo:
            divide_by_org_context_type = r"$\overline{M} \cap O_o$"
        elif instance_id in instance_set_compositions.notm_and_onoto:
            divide_by_org_context_type = r"$\overline{M} \cap O_{\overline{o}}$"
        else:
            raise ValueError
    return divide_by_sub_context_type, divide_by_org_context_type


def draw_features(model_path="meta-llama/Meta-Llama-3-8B", data_name="nqswap"):
    model_name = model_path.split("/")[-1]

    record_dir = PROJ_DIR / "cache_data" / model_name
    image_save_dir = PROJ_DIR / "images"
    if data_name not in ["nqswap", "macnoise"]:
        record_dir = record_dir / data_name
        image_save_dir = image_save_dir / data_name

    records_path = record_dir / "activation_features.json"
    image_save_dir = image_save_dir / "KC_activation_features"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    if data_name == "nqswap":
        instance_set_compositions = get_nqswap_compositions(model_name)
    elif data_name == "macnoise":
        instance_set_compositions = get_macnoise_compositions(model_name)
    else:
        raise ValueError

    feature_records = json.load(open(records_path, "r"))
    feature_names = [
        "kurtosis",
        "hoyer",
        "gini",
        "l2norm",
        "l1norm",
    ]
    module_types = [
        "mlp",
        "self-attn",
        "hidden",
    ]
    for idx in range(len(feature_records)):
        if "mlp" in feature_records[idx]["module"]:
            feature_records[idx]["module type"] = "mlp"
        elif "self_attn" in feature_records[idx]["module"]:
            feature_records[idx]["module type"] = "self-attn"
        elif "norm" in feature_records[idx]["module"]:
            feature_records[idx]["module type"] = "hidden norm"
        else:
            feature_records[idx]["module type"] = "hidden"
        feature_records[idx]["layer id"] = int(feature_records[idx]["module"].split(".")[2])

        instance_id = feature_records[idx]["bid"]
        divide_by_sub_context_type, divide_by_org_context_type = \
            get_instance_type_latex(instance_set_compositions, instance_id)
        feature_records[idx]["type-1"] = divide_by_sub_context_type
        feature_records[idx]["type-2"] = divide_by_org_context_type

    palette = {r"$M \cap S_s$": "blue", r"$M \cap S_o$": "red"}
    dashes = {r"$M \cap S_s$": "", r"$M \cap S_o$": ""}
    dataframe = pd.DataFrame.from_records(feature_records)

    show_types = [r"$M \cap S_s$", r"$M \cap S_o$"]
    rcParams['axes.labelsize'] = 21
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['legend.fontsize'] = 20
    rcParams['legend.title_fontsize'] = 16
    rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'mathtext.default': 'regular',
        'font.weight': 'bold',
    })
    for module_type in module_types:
        for fea_name in feature_names:
            plt.figure(figsize=(5, 4), dpi=150)
            ax = sns.lineplot(data=dataframe[(dataframe["module type"] == module_type) &
                                             (dataframe["type-1"].isin(show_types))],
                              x="layer id", y=fea_name, hue="type-1", style="type-1", palette=palette,
                              dashes=dashes)
            plt.grid(True)
            plt.xlabel(r"\textbf{Layer}")
            show_fea_name = fea_name[0].capitalize() + fea_name[1:]
            show_fea_name = r"\textbf{" + show_fea_name + "}"
            plt.ylabel(show_fea_name)
            plt.savefig(image_save_dir / f"{model_name} {data_name} {module_type} {fea_name}.pdf",
                        format='pdf', bbox_inches='tight')

            plt.show()


if __name__ == '__main__':
    draw_features()
    draw_features(model_path="meta-llama/Llama-2-7b-hf")
