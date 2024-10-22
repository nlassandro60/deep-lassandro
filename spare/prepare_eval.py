import torch
from spare.utils import load_model, add_file_handler
from spare.utils import PROJ_DIR
from spare.datasets.eval_datasets_nqswap import NQSwap
from spare.datasets.eval_datasets_macnoise import MACNoise
from spare.eval_utils import exact_match_score_with_multiple_candidates as em
from spare.eval_utils import sub_ans_exact_match_score_with_macnoise as macnoise_sub_em
from spare.eval_utils import prefix_match
import argparse
import logging
import os
import json
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--k_shot', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--demonstrations_org_context', action="store_true")
    parser.add_argument('--demonstrations_org_answer', action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--run_open_book", action="store_true")
    parser.add_argument("--run_close_book", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--write_logs", action="store_true")
    parser.add_argument("--manually_block_heads", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="nqswap", choices=["nqswap", "macnoise"])
    return parser.parse_args()


@torch.no_grad()
def greedy_decoding_hf(
        model,
        tokenizer,
        input_ids,
        generation_kwargs,
):
    assert len(input_ids) == 1
    if "eos_token_id" not in generation_kwargs:
        logger.warning("eos_token_id is not set")

    gen_kwargs = model.generation_config.to_dict()
    gen_kwargs.update(generation_kwargs)
    gen_kwargs.pop("max_length")
    generated_ids = model.generate(input_ids=input_ids, **gen_kwargs)
    generated_ids = generated_ids[0][len(input_ids[0]):].tolist()
    generated_str = tokenizer.decode(generated_ids)
    return {
        "generated_ids": generated_ids,
        "generated_str": generated_str
    }


@torch.no_grad()
def main(
        write_logs=False,
        flash_attn=True,
        exp_name="debug",
        model_path="google/gemma-2-9b",
        k_shot=32,
        seed=42,
        demonstrations_org_context=True,
        demonstrations_org_answer=True,
        batch_size=1,
        run_open_book=False,
        run_close_book=True,
        dataset_name="nqswap",
        args=None
):
    outputs_dir = PROJ_DIR / "cache_data" / "prepare_eval" / exp_name
    if "debug" not in exp_name.lower():
        if os.path.exists(outputs_dir / "results.json"):
            logger.error(f"{outputs_dir} results.json exists")
            return
    os.makedirs(outputs_dir, exist_ok=True)
    if write_logs:
        add_file_handler(logger, outputs_dir, "log")
    if args is not None:
        logger.info(json.dumps(vars(args), indent=4))

    model, tokenizer = load_model(model_path, flash_attn)
    if dataset_name == "nqswap":
        dataset = NQSwap(k_shot, seed, tokenizer, demonstrations_org_context, demonstrations_org_answer)
    elif dataset_name == "macnoise":
        dataset = MACNoise(k_shot, seed, tokenizer, demonstrations_org_context, demonstrations_org_answer, 5120)
    else:
        raise NotImplementedError

    dataloader = dataset.get_dataloader(batch_size)

    line_break_id = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    generation_kwargs = {"max_new_tokens": 12, "do_sample": False, "eos_token_id": line_break_id}
    predictions = []
    sub_answers = []
    org_answers = []
    without_ctx_predictions = []
    for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sub_answers.extend(batch["sub_answers"])
        org_answers.extend(batch["org_answers"])

        # open book
        if run_open_book:
            gen_results = greedy_decoding_hf(
                model,
                tokenizer,
                batch["with_ctx_input_ids"].cuda(),
                generation_kwargs,
                flash_attn=flash_attn
            )
            predictions.append(gen_results["generated_str"].split("\n")[0])
            if bid == 0:
                logger.info(f"first example:\n{tokenizer.decode(batch['with_ctx_input_ids'][0].tolist())}")
                logger.info(f"first example prediction: {predictions[-1]}")
            print(predictions[-1])

        # close book
        if run_close_book:
            attention_mask = None
            input_ids = batch["without_ctx_input_ids"].cuda()
            gen_results = greedy_decoding_hf(
                model,
                tokenizer,
                input_ids,
                generation_kwargs,
                attention_mask=attention_mask,
            )
            without_ctx_predictions.append(gen_results["generated_str"].split("\n")[0])
            if bid == 0:
                logger.info(f"first example:\n{tokenizer.decode(input_ids[0].tolist())}")
                logger.info(f"first example prediction: {without_ctx_predictions[-1]}")

    if run_open_book:
        assert len(predictions) == len(sub_answers)
        logger.info(f"{len(predictions)} examples")
    if run_close_book:
        assert len(without_ctx_predictions) == len(org_answers)
        logger.info(f"{len(without_ctx_predictions)} examples")

    additional_information = dict()
    all_close_book_scores, close_book_em = None, None
    if run_close_book:
        all_close_book_scores = [em(pred, ts) for pred, ts in zip(without_ctx_predictions, org_answers)]
        close_book_em = sum(all_close_book_scores) / len(all_close_book_scores)
        logger.info(f"close book EM score: {close_book_em}")

    all_sub_scores, all_org_scores, sub_answer_em, org_answer_em = None, None, None, None
    if run_open_book:
        if dataset_name == "macnoise":
            all_sub_scores = [macnoise_sub_em(pred, ts) for pred, ts in zip(predictions, dataset)]
        else:
            all_sub_scores = [em(pred, ts) for pred, ts in zip(predictions, sub_answers)]
        sub_answer_em = sum(all_sub_scores) / len(all_sub_scores)
        logger.info(f"sub_answer EM score: {sub_answer_em}")

        all_org_scores = [em(pred, ts) for pred, ts in zip(predictions, org_answers)]
        org_answer_em = sum(all_org_scores) / len(all_org_scores)
        logger.info(f"org_answer EM score: {org_answer_em}")

        both_correct = [1 if xx == yy else 0 for xx, yy in zip(all_org_scores, all_sub_scores)]
        additional_information["both_correct"] = both_correct
        logger.info(f"both correct num: {sum(both_correct)}")

    json.dump({"close_book_em": close_book_em,
               "sub_answer_em": sub_answer_em,
               "org_answer_em": org_answer_em,
               "all_sub_scores": all_sub_scores,
               "all_org_scores": all_org_scores,
               "all_close_book_scores": all_close_book_scores,
               "predictions": predictions,
               "additional_information": additional_information},
              open(os.path.join(outputs_dir, 'results.json'), "w"))
    if args is not None:
        json.dump(vars(args), open(os.path.join(outputs_dir, 'args.json'), "w"), indent=4)


if __name__ == '__main__':
    main_args = get_args()
    main(args=main_args, **vars(main_args))
