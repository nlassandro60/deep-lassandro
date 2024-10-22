import json
from dataclasses import dataclass
from typing import Set, Optional
from transformers.utils import ModelOutput


@dataclass
class InstanceSets(ModelOutput):
    memorised_set: Optional[Set] = None
    not_memorised_set: Optional[Set] = None
    give_sub_ctx_pred_sub_ans_set: Optional[Set] = None
    give_sub_ctx_pred_org_ans_set: Optional[Set] = None
    give_sub_context_pred_not_org_sub_answer_set: Optional[Set] = None

    give_org_ctx_pred_org_ans_set: Optional[Set] = None
    give_org_ctx_pred_not_org_ans_set: Optional[Set] = None


@dataclass
class InstanceSetsComposition(ModelOutput):
    m_and_ss: Optional[Set] = None
    m_and_so: Optional[Set] = None
    m_and_snotso: Optional[Set] = None
    notm_and_ss: Optional[Set] = None
    notm_and_so: Optional[Set] = None
    notm_and_snotso: Optional[Set] = None
    m_and_oo: Optional[Set] = None
    m_and_onoto: Optional[Set] = None
    notm_and_oo: Optional[Set] = None
    notm_and_onoto: Optional[Set] = None
    instance_sets: Optional[InstanceSets] = None


def group_instance(close_book_log_name, open_book_log_name, no_conflict_open_book_log_name=None):
    from spare.utils import PROJ_DIR

    close_book_log_path = PROJ_DIR / "cache_data" / "prepare_eval" / close_book_log_name / "results.json"
    open_book_log_path = PROJ_DIR / "cache_data" / "prepare_eval" / open_book_log_name / "results.json"
    open_book_args_path = PROJ_DIR / "cache_data" / "prepare_eval" / open_book_log_name / "args.json"

    close_book_log = json.load(open(close_book_log_path, "r"))
    open_book_log = json.load(open(open_book_log_path, "r"))
    open_book_args = json.load(open(open_book_args_path, "r"))

    close_book_scores = close_book_log["all_close_book_scores"]
    open_book_sub_answer_scores = open_book_log["all_sub_scores"]
    open_book_org_answer_scores = open_book_log["all_org_scores"]

    assert len(close_book_scores) == len(open_book_sub_answer_scores) == len(open_book_org_answer_scores)

    num_examples = len(open_book_sub_answer_scores)

    memorised_set = set([idx for idx, _em in enumerate(close_book_scores) if _em == 1])
    not_memorised_set = set([idx for idx, _em in enumerate(close_book_scores) if _em == 0])

    give_sub_ctx_pred_sub_ans_set = set([idx for idx, _em in enumerate(open_book_sub_answer_scores) if _em == 1])
    give_sub_ctx_pred_org_ans_set = set([idx for idx, _em in enumerate(open_book_org_answer_scores) if _em == 1])

    both_correct = give_sub_ctx_pred_org_ans_set.intersection(give_sub_ctx_pred_sub_ans_set)
    if len(both_correct) > 0:
        give_sub_ctx_pred_org_ans_set = give_sub_ctx_pred_org_ans_set - both_correct
        give_sub_ctx_pred_sub_ans_set = give_sub_ctx_pred_sub_ans_set - both_correct

    give_sub_context_pred_not_org_sub_answer_set = (set(range(num_examples)) - give_sub_ctx_pred_sub_ans_set
                                                    - give_sub_ctx_pred_org_ans_set)

    if no_conflict_open_book_log_name is not None:
        no_conflict_open_book_log_path = PROJ_DIR / "outputs" / no_conflict_open_book_log_name / "results.json"
        no_conflict_open_book_log = json.load(open(no_conflict_open_book_log_path, "r"))
        no_conflict_open_book_org_answer_scores = no_conflict_open_book_log["all_org_scores"]
        assert len(close_book_scores) == len(no_conflict_open_book_org_answer_scores)

        give_org_ctx_pred_org_ans_set = set([idx for idx, _em in
                                             enumerate(no_conflict_open_book_org_answer_scores) if _em == 1])
        give_org_ctx_pred_not_org_ans_set = (set(range(num_examples)) - give_org_ctx_pred_org_ans_set)
    else:
        give_org_ctx_pred_org_ans_set = None
        give_org_ctx_pred_not_org_ans_set = None

    return InstanceSets(memorised_set=memorised_set,
                        not_memorised_set=not_memorised_set,
                        give_sub_ctx_pred_sub_ans_set=give_sub_ctx_pred_sub_ans_set,
                        give_sub_ctx_pred_org_ans_set=give_sub_ctx_pred_org_ans_set,
                        give_sub_context_pred_not_org_sub_answer_set=give_sub_context_pred_not_org_sub_answer_set,
                        give_org_ctx_pred_org_ans_set=give_org_ctx_pred_org_ans_set,
                        give_org_ctx_pred_not_org_ans_set=give_org_ctx_pred_not_org_ans_set)


def get_composition(instance_sets: InstanceSets):
    m_and_ss = instance_sets.memorised_set & instance_sets.give_sub_ctx_pred_sub_ans_set
    m_and_so = instance_sets.memorised_set & instance_sets.give_sub_ctx_pred_org_ans_set
    m_and_snotso = instance_sets.memorised_set & instance_sets.give_sub_context_pred_not_org_sub_answer_set
    notm_and_ss = instance_sets.not_memorised_set & instance_sets.give_sub_ctx_pred_sub_ans_set
    notm_and_so = instance_sets.not_memorised_set & instance_sets.give_sub_ctx_pred_org_ans_set
    notm_and_snotso = instance_sets.not_memorised_set & instance_sets.give_sub_context_pred_not_org_sub_answer_set

    if instance_sets.give_org_ctx_pred_org_ans_set is not None:
        m_and_oo = instance_sets.memorised_set & instance_sets.give_org_ctx_pred_org_ans_set
        m_and_onoto = instance_sets.memorised_set & instance_sets.give_org_ctx_pred_not_org_ans_set
        notm_and_oo = instance_sets.not_memorised_set & instance_sets.give_org_ctx_pred_org_ans_set
        notm_and_onoto = instance_sets.not_memorised_set & instance_sets.give_org_ctx_pred_not_org_ans_set
    else:
        m_and_oo, m_and_onoto, notm_and_oo, notm_and_onoto = None, None, None, None

    print(f"M: {len(instance_sets.memorised_set)}")
    print(f"notM: {len(instance_sets.not_memorised_set)}")
    print(f"Ss: {len(instance_sets.give_sub_ctx_pred_sub_ans_set)}")
    print(f"So: {len(instance_sets.give_sub_ctx_pred_org_ans_set)}")
    print(f"Snotso: {len(instance_sets.give_sub_context_pred_not_org_sub_answer_set)}")

    print(f"M and Ss: {len(m_and_ss)}")
    print(f"M and So: {len(m_and_so)}")
    print(f"M and Snotso: {len(m_and_snotso)}")

    print(f"notM and Ss: {len(notm_and_ss)}")
    print(f"notM and So: {len(notm_and_so)}")
    print(f"notM and Snotso: {len(notm_and_snotso)}")

    if m_and_oo is not None:
        print(f"M and Oo: {len(m_and_oo)}")
        print(f"M and Onoto: {len(m_and_onoto)}")
        print(f"notM and Oo: {len(notm_and_oo)}")
        print(f"notM and Onoto: {len(notm_and_onoto)}")

    mr = len(m_and_so) / (len(m_and_so) + len(m_and_ss))
    print(f"memorisation ratio: {mr * 100:.3f}%")

    return InstanceSetsComposition(m_and_ss=m_and_ss,
                                   m_and_so=m_and_so,
                                   m_and_snotso=m_and_snotso,
                                   notm_and_ss=notm_and_ss,
                                   notm_and_so=notm_and_so,
                                   notm_and_snotso=notm_and_snotso,
                                   m_and_oo=m_and_oo,
                                   m_and_onoto=m_and_onoto,
                                   notm_and_oo=notm_and_oo,
                                   notm_and_onoto=notm_and_onoto,
                                   instance_sets=instance_sets)


def get_nqswap_compositions(model_name):
    if model_name == "Meta-Llama-3-8B":
        close_book_log_name = "nqswap-llama3-8b-32shot--1examples-closebook-dedup"
        open_book_log_name = "nqswap-llama3-8b-4shot--1examples-openbook-orgctxans"
    elif model_name == "Llama-2-7b-hf":
        close_book_log_name = "nqswap-llama2-7b-32shot--1examples-closebook-dedup"
        open_book_log_name = "nqswap-llama2-7b-4shot--1examples-openbook-orgctxans"
    elif model_name == "gemma-2-9b":
        close_book_log_name = "nqswap-gemma-2-9b-32shot--1examples-closebook"
        open_book_log_name = "nqswap-gemma-2-9b-4shot--1examples-openbook"
    else:
        raise ValueError()
    org_ctx_ans_demonstration_instance_sets = group_instance(
        close_book_log_name, open_book_log_name,
        no_conflict_open_book_log_name=None
    )
    instance_set_compositions = get_composition(org_ctx_ans_demonstration_instance_sets)
    return instance_set_compositions


def get_macnoise_compositions(model_name):
    if model_name == "Meta-Llama-3-8B":
        close_book_log_name = "macnoise-chatgpt-llama3-8b-4shot-5120examples-mask0rh-macnoise_1309"
        open_book_log_name = "macnoise-chatgpt-llama3-8b-4shot-5120examples-mask0rh-macnoise_1309"
    elif model_name == "Llama-2-7b-hf":
        open_book_log_name = "macnoise-chatgpt-llama2-7b-4shot-5120examples-mask0rh-macnoise_1309"
        close_book_log_name = "macnoise-chatgpt-llama2-7b-4shot-5120examples-mask0rh-macnoise_1309"
    else:
        raise ValueError()
    org_ctx_ans_demonstration_instance_sets = group_instance(
        close_book_log_name, open_book_log_name,
        no_conflict_open_book_log_name=None
    )
    instance_set_compositions = get_composition(org_ctx_ans_demonstration_instance_sets)
    return instance_set_compositions


def get_instance_compositions(data_name, model_name):
    expand_conflict_records = None
    if data_name == "nqswap":
        instance_set_compositions = get_nqswap_compositions(model_name)
    elif data_name == "macnoise":
        instance_set_compositions = get_macnoise_compositions(model_name)
    else:
        raise ValueError
    return instance_set_compositions, expand_conflict_records
