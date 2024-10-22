import os
from tqdm import tqdm
from spare.datasets.function_extraction_datasets import EncodeREODQADataset
from spare.patch_utils import InspectOutputContext
from spare.utils import PROJ_DIR, init_frozen_language_model
from spare.group_prompts import load_dataset_and_memorised_set
from spare.sae_repe_utils import load_grouped_prompts
import torch
import logging
import argparse
import json
import shutil

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    # load data from load_data_name
    parser.add_argument("--load_data_name", type=str, required=True)
    parser.add_argument("--shots_to_encode", type=int, nargs='+', required=True)
    parser.add_argument("--seeds_to_encode", type=int, nargs='+', required=True)

    # encode data and save to save_hiddens_name
    parser.add_argument("--save_hiddens_name", type=str, required=True)

    return parser.parse_args()


@torch.inference_mode()
def save_grouped_hiddens(tokenizer, model, model_name, data_name, save_hiddens_name,
                         load_data_name, shots_to_encode, seeds_to_encode):
    num_layers = len(model.model.layers)
    inspect_modules = [f'model.layers.{layer_idx}' for layer_idx in range(num_layers)]

    pred_sub_answer_data, pred_org_answer_data, load_files = load_grouped_prompts(
        model_name,
        load_data_name,
        shots_to_encode,
        seeds_to_encode
    )

    data, memorised_set = load_dataset_and_memorised_set(data_name, model_name)
    encode_re_odqa_dataset = EncodeREODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        data_to_encode=pred_sub_answer_data + pred_org_answer_data,
        demonstration_pool_size=128
    )
    dataloader = encode_re_odqa_dataset.get_dataloader()

    save_dir = PROJ_DIR / "cache_data" / model_name / save_hiddens_name
    os.makedirs(save_dir, exist_ok=True)
    for module in inspect_modules:
        dir_context = save_dir / f"{module}-use-context"
        dir_parameter = save_dir / f"{module}-use-parameter"
        os.makedirs(dir_context, exist_ok=True)
        os.makedirs(dir_parameter, exist_ok=True)

    for batch in tqdm(dataloader, desc="forward and save hiddens"):
        assert len(batch["input_ids"]) == 1  # batch_size = 1
        with InspectOutputContext(model, inspect_modules, False, True) as inspect:
            model(input_ids=batch["input_ids"].cuda(), early_exit_at_layer_idx=num_layers)

            for module in inspect_modules:
                if "use_context_idx" in batch:
                    save_path = save_dir / f"{module}-use-context" / f"{batch['use_context_idx']}"
                else:
                    save_path = save_dir / f"{module}-use-parameter" / f"{batch['use_parameter_idx']}"
                cur_hidden = inspect.catcher[module]
                torch.save(cur_hidden.float().cpu(), save_path)

    for layer_idx in tqdm(range(num_layers), desc="combine hiddens"):
        for use_knowledge in ["use-context", "use-parameter"]:
            cur_paths = os.listdir(save_dir / f"model.layers.{layer_idx}-{use_knowledge}")
            cur_paths = sorted(cur_paths, key=lambda p: int(p))
            hiddens = [torch.load(save_dir / f"model.layers.{layer_idx}-{use_knowledge}" / p) for p in cur_paths]
            hiddens = torch.cat(hiddens)
            torch.save(hiddens, save_dir / f"layer{layer_idx}-{use_knowledge}.pt")
            shutil.rmtree(save_dir / f"model.layers.{layer_idx}-{use_knowledge}")

    json.dump([str(f) for f in load_files], open(save_dir / "load_files.json", "w"), indent=4)


def main():
    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")
    model_name = os.path.basename(args.model_path)
    model, tokenizer = init_frozen_language_model(args.model_path)
    save_grouped_hiddens(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        data_name=args.data_name,
        save_hiddens_name=args.save_hiddens_name,
        load_data_name=args.load_data_name,
        shots_to_encode=args.shots_to_encode,
        seeds_to_encode=args.seeds_to_encode,
    )


if __name__ == '__main__':
    main()
