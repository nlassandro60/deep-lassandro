import os
import json
import torch
import logging
from transformers import AutoTokenizer
from pathlib import Path
from spare.sae import Sae
from spare.function_extraction_modellings.function_extraction_gemma2 import Gemma2ForCausalLM
from spare.function_extraction_modellings.function_extraction_llama import LlamaForCausalLM
from spare.sae_lens.eleuther_sae_wrapper import EleutherSae

PROJ_DIR = Path(os.environ.get("PROJ_DIR", "./"))


def add_file_handler(_logger, output_dir: str, file_name: str):
    file_handler = logging.FileHandler(os.path.join(output_dir, file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s"))
    _logger.addHandler(file_handler)


def load_jsonl(path):
    with open(path, "r") as fn:
        data = [json.loads(line) for line in fn.readlines()]
    return data


def load_model(model_path, flash_attn, not_return_model=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    attn_implementation = "flash_attention_2" if flash_attn else "eager"
    print(f"attn_implementation = {attn_implementation}")
    if not_return_model:
        model = None
    else:
        if "gemma" in model_path.lower():
            model = Gemma2ForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        model.cuda().eval()
    return model, tokenizer


def init_frozen_language_model(model_path, attn_imp="flash_attention_2"):
    bf16 = torch.bfloat16
    if "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation=attn_imp, torch_dtype=bf16)
    elif "gemma" in model_path:
        model = Gemma2ForCausalLM.from_pretrained(model_path, attn_implementation=attn_imp, torch_dtype=bf16)
    else:
        raise NotImplementedError
    model.cuda().eval()
    for pn, p in model.named_parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_frozen_sae(layer_idx, model_name):
    if model_name == "Meta-Llama-3-8B":
        sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint=f"layers.{layer_idx}")
    elif model_name == "Llama-2-7b-hf":
        sae = Sae.load_from_hub("yuzhaouoe/Llama2-7b-SAE", hookpoint=f"layers.{layer_idx}")
    elif model_name == "gemma-2-9b":
        sae, cfg_dict, sparsity = EleutherSae.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id=f"layer_{layer_idx}/width_131k/canonical",
            device="cuda"
        )
    else:
        raise NotImplementedError(f"sae for {model_name}")
    for pn, p in sae.named_parameters():
        p.requires_grad = False
    sae.cuda()
    return sae
