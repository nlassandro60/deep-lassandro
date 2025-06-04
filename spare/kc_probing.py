import copy
import os
import json
import torch
import torch.nn as nn
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
from spare.analysis.analysis_save_activations import load_activations, get_nqswap_distinct_question_info, \
    get_macnoise_distinct_question_info
from spare.utils import PROJ_DIR
import seaborn as sns
from matplotlib import pyplot as plt
from pylab import rcParams
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from spare.analysis.group_instance import get_instance_compositions
from sklearn.metrics import auc

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class LogisticRegression(nn.Module): #!# modello di regressione logistica per il rilevamento dei conflitti
    def __init__(self, input_dim, use_bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)

    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x.to(self.linear.weight.device)
        return torch.sigmoid(self.linear(x)).squeeze(1)


class ActivationDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return torch.tensor(self.activations[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def load_conflict_train_test_data(memorised_set=None, model_name=None, data_name=None, analyse_activation=None,
                                  layer_idx=None, return_bid=False):
    
    #!# load_activations carica le attivazioni già salvate in memoria

    conflict_activations = load_activations(model_name, data_name, analyse_activation, "conflict", layer_idx)
    none_conflict_activations = load_activations(model_name, data_name, analyse_activation, "none_conflict", layer_idx)

    if data_name == "nqswap":
        distinct_question_group = get_nqswap_distinct_question_info()
    else:
        distinct_question_group = get_macnoise_distinct_question_info()

    select_group_ids = []
    for idx in memorised_set:
        for group_idx, group in enumerate(distinct_question_group):
            if idx in group:
                select_group_ids.append(group_idx)
    select_group_ids = list(set(select_group_ids))
    all_groups = [distinct_question_group[group_idx] for group_idx in select_group_ids]
    all_groups = [[idx for idx in group if idx in memorised_set] for group in all_groups]

    rng = np.random.RandomState(666)
    rng.shuffle(all_groups)
    if data_name == "nqswap":
        test_group = all_groups[:100]
        train_group = all_groups[100:]
    else:
        test_group = all_groups[:80]
        train_group = all_groups[80:]

    conflict_group = train_group[:len(train_group) // 2]
    none_conflict_group = train_group[len(train_group) // 2:]

    train_conflict_ids = [rng.choice(group, 1)[0] for group in conflict_group]
    train_none_conflict_ids = [rng.choice(group, 1)[0] for group in none_conflict_group]

    train_conflict_ids = train_conflict_ids + train_none_conflict_ids
    train_none_conflict_ids = train_conflict_ids

    test_conflict_ids = []
    test_none_conflict_ids = []
    for group in test_group:
        test_conflict_ids.extend(group)
        test_none_conflict_ids.append(group[0])

    train_labels = ["conflict"] * len(train_conflict_ids) + ["none-conflict"] * len(train_none_conflict_ids)
    train_conflict_activations = conflict_activations.index_select(
        index=torch.tensor(train_conflict_ids), dim=0
    )
    train_none_conflict_activations = none_conflict_activations.index_select(
        index=torch.tensor(train_none_conflict_ids), dim=0
    )
    train_activations = torch.cat([train_conflict_activations, train_none_conflict_activations]).numpy()

    test_labels = ["conflict"] * len(test_conflict_ids) + ["none-conflict"] * len(test_none_conflict_ids)
    test_conflict_activations = conflict_activations.index_select(
        dim=0, index=torch.tensor(test_conflict_ids)
    )
    test_none_conflict_activations = none_conflict_activations.index_select(
        dim=0, index=torch.tensor(test_none_conflict_ids)
    )
    test_activations = torch.cat([test_conflict_activations, test_none_conflict_activations]).numpy()

    if return_bid:
        return (train_activations, train_labels, train_conflict_ids, train_none_conflict_ids,
                test_activations, test_labels, test_conflict_ids, test_none_conflict_ids)
    else:
        return train_activations, train_labels, test_activations, test_labels


@torch.no_grad()
def logistic_regression_eval(model, dataloader):
    model.eval()

    correct = 0
    total = 0
    predictions = []
    labels = []
    for hidden_state, label in dataloader:
        output = model(hidden_state.cuda())
        predicted = (output > 0.5).float()
        total += label.size(0)
        correct += (predicted == label.cuda()).sum().item()

        predictions.extend(output.tolist())
        labels.extend(label.tolist())

    AUC = roc_auc_score(labels, predictions)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    AUPRC = auc(recall, precision)
    ACC = correct / len(labels)
    return {"ACC": ACC, "AUC": AUC, "AUPRC": AUPRC}


def main(model_path="meta-llama/Meta-Llama-3-8B",
         data_name="nqswap",
         analyse_activation="hidden",
         layer_idx=None,
         save_logs=True,
         k_shot=4,
         tag="",
         rewrite=False,
         save_latest=True,
         ):
    model_name = os.path.basename(model_path)
    instance_set_compositions, _ = get_instance_compositions(data_name, model_name)

    train_activations, train_labels, test_activations, test_labels = load_conflict_train_test_data(
        instance_set_compositions=instance_set_compositions, model_name=model_name, tag=tag,
        data_name=data_name, analyse_activation=analyse_activation, layer_idx=layer_idx, k_shot=k_shot
    )
    assert len(train_activations) == len(train_labels)
    assert len(test_activations) == len(test_labels)
    batch_size = 64
    input_dim = train_activations.shape[1]
    label2idx = {"none-conflict": 0, "conflict": 1}
    train_labels = [label2idx[lb] for lb in train_labels]
    test_labels = [label2idx[lb] for lb in test_labels]
    train_dataset = ActivationDataset(train_activations, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ActivationDataset(test_activations, test_labels)

    criterion = nn.BCELoss()
    num_epochs = 20
    base_lambda_l1 = 0.0002
    train_times = 20
    prob_dir = f"prob_conflict{tag}"
    if k_shot == 0:
        prob_dir = f"prob_conflict_zero_shot_act{tag}"
    for factor in [3]:

        if not save_latest:
            save_dir = PROJ_DIR / "checkpoints" / model_name / data_name / prob_dir / analyse_activation
            result_dir = PROJ_DIR / "results" / model_name / data_name / prob_dir / analyse_activation
        else:
            save_dir = PROJ_DIR / "checkpoints_save_latest" / model_name / data_name / prob_dir / analyse_activation
            result_dir = PROJ_DIR / "results_save_latest" / model_name / data_name / prob_dir / analyse_activation

        save_path = save_dir / f"prob_model_list_{layer_idx}_L1factor{factor}.pt"
        result_path = result_dir / f"layer{layer_idx}_acc_auc_L1factor{factor}.json"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not rewrite and os.path.exists(save_path) and os.path.exists(result_path):
            logger.info(f"continue, checkpoints exists: {save_path}")
            continue

        lambda_l1 = base_lambda_l1 * factor
        all_models, all_metrics, all_best_epochs = [], [], []
        for train_time in range(train_times):
            prob_model = LogisticRegression(input_dim=input_dim, use_bias=True)
            prob_model.cuda()
            optimizer = optim.Adam(prob_model.parameters(), lr=0.002)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

            cur_train_best_score, cur_train_best_model, cur_best_epoch = -1, None, 0
            for epoch in range(num_epochs):
                for batch_hidden_states, batch_labels in train_dataloader:
                    prob_model.train()
                    outputs = prob_model(batch_hidden_states.cuda())
                    loss = criterion(outputs, batch_labels.cuda())
                    l1_penalty = 0
                    for param in prob_model.parameters():
                        l1_penalty += torch.sum(torch.abs(param))
                    loss += lambda_l1 * l1_penalty
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if not save_latest:
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    test_eval_metrics = logistic_regression_eval(prob_model, test_dataloader)
                    logger.info(f"{model_name} {data_name} {analyse_activation} layer{layer_idx} "
                                f"TRAIN [{train_time + 1}] Epoch [{epoch + 1}], "
                                f"ACC: {test_eval_metrics['ACC']:.3f}, "
                                f"AUC: {test_eval_metrics['AUC']:.3f}, "
                                f"AUPRC: {test_eval_metrics['AUPRC']:.3f}")
                    mean_score = test_eval_metrics['ACC']
                    if mean_score > cur_train_best_score:
                        cur_best_epoch = epoch
                        cur_train_best_score, cur_train_best_model = mean_score, copy.deepcopy(prob_model)
            if save_latest:
                cur_train_best_model = prob_model

            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_eval_metrics = logistic_regression_eval(cur_train_best_model, test_dataloader)
            logger.info(f"{model_name} {data_name} {analyse_activation} layer{layer_idx} "
                        f"TRAIN [{train_time + 1}], ACC: {test_eval_metrics['ACC']:.3f},"
                        f" AUC: {test_eval_metrics['AUC']:.3f}, AUPRC: {test_eval_metrics['AUPRC']:.3f}")
            cur_train_best_model.cpu()
            all_models.append(cur_train_best_model.state_dict())
            all_metrics.append(test_eval_metrics)
            all_best_epochs.append(cur_best_epoch)

        dataframe = pd.DataFrame.from_records(all_metrics)
        metric_keys = list(all_metrics[0].keys())
        eval_results = {f"{k}_avg": dataframe[k].mean() for k in metric_keys}
        eval_results.update({f"{k}_std": dataframe[k].std() for k in metric_keys})
        logger.info("")
        logger.info(f"L1 Norm {lambda_l1:.4f}, Layer{layer_idx} {model_name} {analyse_activation}")
        for key in metric_keys:
            logger.info(f"{key}_avg: {eval_results[key + '_avg']:.3f}, {key}_std: {eval_results[key + '_std']:.3f}")
        logger.info("")
        print(all_best_epochs)
        if save_logs:
            torch.save(all_models, save_path)
            eval_results.update({f"all_{k}": dataframe[k].tolist() for k in metric_keys})
            json.dump(eval_results, open(result_path, "w"), indent=4)


def get_records(model_path, data_name, l1_factor, k_shot, tag=""):
    model_name = os.path.basename(model_path)
    if "gemma" in model_name:
        layer_num = 42
    else:
        layer_num = 32
    records = []
    for analyse_activation in ["hidden", "mlp", "attn"]:
        results_dir = PROJ_DIR / "results" / model_name / data_name
        latest_results_dir = PROJ_DIR / "results_save_latest" / model_name / data_name
        if k_shot == 0:
            result_dir = results_dir / f"prob_conflict_zero_shot_act{tag}" / analyse_activation
            latest_results_dir = latest_results_dir / f"prob_conflict_zero_shot_act{tag}" / analyse_activation
        else:
            result_dir = results_dir / f"prob_conflict{tag}" / analyse_activation
            latest_results_dir = latest_results_dir / f"prob_conflict{tag}" / analyse_activation
        for layer_idx in range(layer_num):
            logs_path = result_dir / f"layer{layer_idx}_acc_auc_L1factor{l1_factor}.json"
            logs = json.load(open(logs_path, "r"))

            latest_logs_path = latest_results_dir / f"layer{layer_idx}_acc_auc_L1factor{l1_factor}.json adfadsfa"
            if os.path.exists(latest_logs_path):
                latest_logs = json.load(open(latest_logs_path, "r"))
                for idx in range(len(logs["all_ACC"])):
                    cur_acc = latest_logs["all_ACC"][idx]
                    records.append({"activation": analyse_activation, "layer": layer_idx, "acc": cur_acc,
                                    "auc": logs["all_AUC"][idx], "auprc": logs["all_AUPRC"][idx], })
            else:
                for idx in range(len(logs["all_ACC"])):
                    records.append({"activation": analyse_activation, "layer": layer_idx, "acc": logs["all_ACC"][idx],
                                    "auc": logs["all_AUC"][idx], "auprc": logs["all_AUPRC"][idx], })
    return records


def draw_probing_model_accuracy(model_path="meta-llama/Meta-Llama-3-8B",
                                data_name="nqswap",
                                k_shot=4,
                                l1_factor=None,
                                tag=""):
    model_name = os.path.basename(model_path)

    records = get_records(model_path, data_name, l1_factor, k_shot, tag)
    records = pd.DataFrame.from_records(records)
    palette = {"hidden": "red", "mlp": "blue", "attn": "green"}

    image_save_dir = PROJ_DIR / "images" / "KC-detection-probing"
    os.makedirs(image_save_dir, exist_ok=True)

    rcParams['axes.labelsize'] = 21
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.fontsize'] = 22
    rcParams['legend.title_fontsize'] = 20
    rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'mathtext.default': 'regular',
        'font.weight': 'bold',
    })

    plt.plot(figsize=(5, 4), dpi=150)
    sns.lineplot(data=records, x="layer", y="acc", hue="activation", palette=palette)
    # plt.title(f"Probing model for conflict classification. Accuracy\n{model_name} {data_name}")
    plt.ylabel(r"\textbf{Accuracy}")
    plt.xlabel(r"\textbf{Layer}")
    plt.grid(True)
    plt.savefig(image_save_dir / f"{model_name} {data_name} Accuracy.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(figsize=(5, 4), dpi=150)
    sns.lineplot(data=records, x="layer", y="auc", hue="activation", palette=palette)
    plt.ylabel(r"\textbf{AUROC}")
    plt.xlabel(r"\textbf{Layer}")
    plt.grid(True)
    plt.legend(loc="lower right", title="activation")
    plt.savefig(image_save_dir / f"{model_name} {data_name} AUROC.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(figsize=(5, 4), dpi=150)
    sns.lineplot(data=records, x="layer", y="auprc", hue="activation", palette=palette)
    plt.ylabel(r"\textbf{AUPRC}")
    plt.xlabel(r"\textbf{Layer}")
    plt.grid(True)
    plt.savefig(image_save_dir / f"{model_name} {data_name} AUPRC.pdf", format='pdf', bbox_inches='tight')
    plt.show()
