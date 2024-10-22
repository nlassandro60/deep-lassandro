import json
import os
import numpy as np
from spare.utils import load_frozen_sae
from spare.utils import PROJ_DIR
import torch
import logging
from spare.sae_repe_utils import load_grouped_hiddens, get_sae_activations, sample_train_data
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
import multiprocessing

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--load_hiddens_name", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--mutual_information_num_train_examples", type=int, default=None)
    parser.add_argument("--mutual_information_save_name", type=str, required=True)
    parser.add_argument("--minmax_normalisation", action="store_true")
    parser.add_argument("--equal_label_examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def estimate_mutual_information(features_labels: tuple):
    features, labels = features_labels
    cur_mi_scores = mutual_info_classif(features, labels)
    return cur_mi_scores


def mutual_information_correlation(
        model_name, data_name, load_hiddens_name, mutual_information_save_name, sae, layer_idx, num_proc,
        mutual_information_num_train_examples, minmax_normalisation, equal_label_examples, seed
):
    logger.info(f"Start MI: {model_name} {data_name} {load_hiddens_name} layer={layer_idx}")
    hiddens = load_grouped_hiddens(model_name, load_hiddens_name, layer_idx)
    if mutual_information_num_train_examples is not None:
        logger.info(f"sample {mutual_information_num_train_examples} examples to calculate MI")
        hiddens = sample_train_data(label0_hiddens=hiddens["label0_hiddens"], label1_hiddens=hiddens["label1_hiddens"],
                                    num_train_examples=mutual_information_num_train_examples, seed=seed)

    if equal_label_examples and len(hiddens["label0_hiddens"]) != len(hiddens["label1_hiddens"]):
        logger.info(f"equal label examples layer={layer_idx}")
        num_examples = min(len(hiddens["label0_hiddens"]), len(hiddens["label1_hiddens"]))
        candidates = list(range(max(len(hiddens["label0_hiddens"]), len(hiddens["label1_hiddens"]))))
        np.random.RandomState(666).shuffle(candidates)
        selected_indices = torch.tensor(candidates[:num_examples])
        if len(hiddens["label0_hiddens"]) < len(hiddens["label1_hiddens"]):
            hiddens["label1_hiddens"] = hiddens["label1_hiddens"][selected_indices]
        else:
            hiddens["label0_hiddens"] = hiddens["label0_hiddens"][selected_indices]

    label1_sae_acts = get_sae_activations(hiddens["label1_hiddens"], sae)
    label0_sae_acts = get_sae_activations(hiddens["label0_hiddens"], sae)
    acts = torch.cat((label1_sae_acts, label0_sae_acts))
    labels = [1] * len(label1_sae_acts) + [0] * len(label0_sae_acts)
    mean_A = label1_sae_acts.mean(0)
    mean_B = label0_sae_acts.mean(0)
    expectation = mean_A - mean_B
    if minmax_normalisation:
        logger.info(f"MinMax normalisation layer={layer_idx}")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(acts.float().cpu().numpy())
    else:
        X_scaled = acts.float().cpu().numpy()

    del acts
    del label1_sae_acts
    del label0_sae_acts
    del hiddens["label1_hiddens"]
    del hiddens["label0_hiddens"]
    del hiddens
    del sae
    torch.cuda.empty_cache()

    indices = np.arange(X_scaled.shape[1])
    splits = np.array_split(indices, 1000)

    all_args = []
    for cur_split in splits:
        features = X_scaled[:, cur_split]
        all_args.append((features, labels))

    pool = multiprocessing.Pool(processes=num_proc)
    all_mi_scores = []
    for result in tqdm(pool.imap(estimate_mutual_information, all_args), total=len(all_args)):
        all_mi_scores.append(result)

    pool.close()
    pool.join()

    mi_scores = torch.cat([torch.from_numpy(tt) for tt in all_mi_scores])

    save_dir = PROJ_DIR / "cache_data" / model_name / mutual_information_save_name
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f"layer-{layer_idx} mi_expectation.pt"
    torch.save({"mi_scores": mi_scores, "expectation": expectation}, save_path)

    logger.info(f"Saved MI and E to {save_path}")


def main():
    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")
    model_name = os.path.basename(args.model_path)
    sae = load_frozen_sae(args.layer_idx, model_name)
    mutual_information_correlation(
        num_proc=args.num_proc,
        model_name=model_name,
        data_name=args.data_name,
        load_hiddens_name=args.load_hiddens_name,
        sae=sae,
        layer_idx=args.layer_idx,
        minmax_normalisation=args.minmax_normalisation,
        mutual_information_num_train_examples=args.mutual_information_num_train_examples,
        equal_label_examples=args.equal_label_examples,
        mutual_information_save_name=args.mutual_information_save_name,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
