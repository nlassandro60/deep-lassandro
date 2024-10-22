import argparse
import torch
import os
from tqdm import tqdm
import json
import logging
from spare.utils import init_frozen_language_model, PROJ_DIR
from spare.datasets.function_extraction_datasets import REODQADataset
from spare.sae_repe_utils import unified_em, load_dataset_and_memorised_set

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir_name", type=str, required=True)
    parser.add_argument("--seeds_to_encode", type=int, nargs='+', required=True)
    parser.add_argument("--k_shot", type=int, required=True)
    return parser.parse_args()


@torch.inference_mode()
def group_prompts_based_on_behaviours(seeds_to_encode, model_path, k_shot, save_dir_name):
    model_name = os.path.basename(model_path)
    data, memorised_set = load_dataset_and_memorised_set("nqswap", model_name)
    model, tokenizer = init_frozen_language_model(model_path)
    line_break_id = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    generation_kwargs = {"max_new_tokens": 12, "do_sample": False, "eos_token_id": line_break_id,
                         "pad_token_id": line_break_id, "use_cache": True, "temperature": None, "top_p": None}

    collect_hiddens_dataset = REODQADataset(
        tokenizer=tokenizer,
        data=data,
        memorised_set=memorised_set,
        demonstration_pool_size=128,
        task="collect_hiddens"
    )

    record_keys = set()
    save_dir = PROJ_DIR / "cache_data" / model_name / save_dir_name
    os.makedirs(save_dir, exist_ok=True)

    existed_files = os.listdir(save_dir)
    existed_files = [f for f in existed_files if int(f.split("shot")[0]) == k_shot]
    for ef in tqdm(existed_files, desc="load existed data"):
        existed_data = json.load(open(save_dir / ef, "r"))
        for item in existed_data:
            sample_key = str(item["item_idx"]) + "-" + "-".join([str(idx) for idx in item["demonstration_ids"]])
            record_keys.add(sample_key)

    for cur_seed in seeds_to_encode:

        save_path = save_dir / f"{k_shot}shot-seed{cur_seed}-results.json"
        if os.path.exists(save_path):
            logger.info(f"the file ``{save_path}`` exists, pass")

        dataloader = collect_hiddens_dataset.collect_hiddens_dataloader(k_shot=k_shot, seed=cur_seed, batch_size=1)
        all_gen_results = []
        tqdm_bar = tqdm(dataloader, desc=f"seed={cur_seed}")
        sub_em, org_em = 0, 0
        for batch in tqdm_bar:

            sample_key = str(batch["item_idx"]) + "-" + "-".join([str(idx) for idx in batch["demonstration_ids"]])
            if sample_key in record_keys:
                continue

            assert batch["item_idx"] in collect_hiddens_dataset.ids_of_demonstrations
            assert all(demidx in collect_hiddens_dataset.ids_of_demonstrations for demidx in batch["demonstration_ids"])

            for prompt_type in ["dss_s", "dso_s", "doo_s"]:
                gen_results = generate_and_evaluate_for_one_item(
                    model, tokenizer, batch[prompt_type], "nqswap",
                    batch["sub_answers"], batch["org_answers"],
                    batch["sub_contexts"], generation_kwargs
                )
                gen_results.update({"item_idx": batch["item_idx"],
                                    "demonstration_ids": batch["demonstration_ids"],
                                    "prompt_type": prompt_type})

                ss_loss = calculate_loss(model, batch[prompt_type + "s"])
                so_loss = calculate_loss(model, batch[prompt_type + "o"])
                gen_results.update({"ss_loss": ss_loss, "so_loss": so_loss})

                if gen_results["sub_answer_em"] == 1:
                    sub_em += 1
                if gen_results["org_answer_em"] == 1:
                    org_em += 1

                all_gen_results.append(gen_results)

            tqdm_bar.set_description(f"{k_shot}shot seed{cur_seed} EM-C[{sub_em}] EM-M[{org_em}]")

        json.dump(all_gen_results, open(save_path, "w"), indent=4, ensure_ascii=False)


def calculate_loss(model, input_ids):
    input_ids = input_ids.cuda()
    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss.item()


def generate_and_evaluate_for_one_item(model, tokenizer, input_ids, data_name, sub_answers, org_answers,
                                       sub_contexts, generation_kwargs):
    assert len(input_ids) == 1  # batch size == 1
    decoding_results = model.generate(input_ids=input_ids.cuda(), output_scores=True, **generation_kwargs)
    pred_answer = tokenizer.batch_decode(decoding_results[:, input_ids.shape[1]:])[0]  # batch_size= 1
    pred_answer = pred_answer.split("\n")[0].strip()

    # # batch_size= 1
    sub_answer_em, org_answer_em = unified_em(pred_answer, org_answers[0], sub_answers[0], sub_contexts[0], data_name)

    return {"sub_answer": sub_answers[0], "org_answer": org_answers[0], "pred_answer": pred_answer,
            "sub_answer_em": sub_answer_em, "org_answer_em": org_answer_em}


def main():
    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")
    group_prompts_based_on_behaviours(
        seeds_to_encode=args.seeds_to_encode,
        model_path=args.model_path,
        k_shot=args.k_shot,
        save_dir_name=args.save_dir_name,
    )


if __name__ == '__main__':
    main()
