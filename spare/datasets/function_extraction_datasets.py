from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer
import numpy as np
import torch
import copy
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class REODQADataset(Dataset):
    """
    Representation Engineering for Open-Domain Question-Answering
    """

    def __init__(self, tokenizer, data, memorised_set,
                 demonstration_pool_size=128, task="initial_inference_without_intervention"):
        super(REODQADataset, self).__init__()
        self.k_shot_candidates = None
        self.tokenizer = tokenizer
        self.data = data
        self.idx2item = {item["idx"]: item for item in self.data}
        self.memorised_set = memorised_set
        self.group2ids = dict()
        self.idx2group = dict()
        self.group_distinct_questions()
        self.memorised_groups = self.get_memorised_groups()
        candidate_demonstration_groups = list(range(len(self.memorised_groups)))
        np.random.RandomState(42).shuffle(candidate_demonstration_groups)  # fixed seed
        self.demonstration_groups_ids = candidate_demonstration_groups[:demonstration_pool_size]
        self.num_distinct_questions = len(list(self.group2ids.keys()))

        self.task = task

        # self.seed = seed
        self.select_k_demonstrations_group_ids = None
        if self.task in ["initial_ICL_without_intervention",
                         "initial_ICL_with_intervention",
                         "baseline_ICL_to_steer",
                         "baseline_DoLa_to_steer",
                         "baseline_CAD_to_steer", ]:
            # iterate over test-set, except demonstrations
            self.data_for_iter = [item for item in self.data if self.idx2group[item["idx"]]
                                  not in self.demonstration_groups_ids]
        elif self.task in ["collect_hiddens"]:
            # only iterate over hold-out demonstrations
            self.data_for_iter = [item for item in self.data if self.idx2group[item["idx"]]
                                  in self.demonstration_groups_ids]
            self.sample_keys = set()
            self.ids_of_demonstrations = []
            for gid in self.demonstration_groups_ids:
                self.ids_of_demonstrations.extend(self.group2ids[gid])
        elif self.task in ["encode_and_save_hiddens"]:
            self.data_for_iter = []
        else:
            raise ValueError(f"task ``{task}`` not recognized")

    def get_memorised_groups(self):
        memorised_groups = set()
        for idx in self.memorised_set:
            group_idx = self.idx2group[idx]
            memorised_groups.add(group_idx)
        return list(memorised_groups)

    def group_distinct_questions(self):
        distinct_question_group = defaultdict(list)
        for item in self.data:
            distinct_question_group[item["question"]].append(item["idx"])
        self.group2ids = {group_idx: ids for group_idx, ids in enumerate(distinct_question_group.values())}
        for group_idx, idx_list in self.group2ids.items():
            for idx in idx_list:
                self.idx2group[idx] = group_idx

    def initial_ICL_dataloader(self, k_shot, seed, batch_size=1, num_workers=8, shuffle=False):
        """
        initial inference: use org_context, org_answer for demonstrations,
        The demonstrations are from the memorised set
        """
        np.random.RandomState(seed).shuffle(self.demonstration_groups_ids)
        selected_k_demonstrations_group_ids = self.demonstration_groups_ids[:k_shot]

        def collate_fn(batch):
            item = batch[0]
            demonstration_ids = [self.group2ids[gid][0] for gid in selected_k_demonstrations_group_ids]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]
            #  use org_context, org_answer for demonstrations, and demonstrations are from the memorised set
            prompt = self.verbalise_demonstrations(demonstrations, "org_context", "org_answer")
            #  use sub_context for test examples -- it is knowledge conflict
            prompt = prompt + self.verbalise_one_example(item, "sub_context", None, is_test=True)
            inputs = self.tokenizer([prompt], return_tensors="pt")
            return {"input_ids": inputs["input_ids"],
                    "sub_answers": [item["sub_answer"]],  # list[list[srt]]
                    "org_answers": [item["org_answer"]],
                    "sub_contexts": [item["sub_context"]],
                    "item_idx": item["idx"]}

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def ICL_to_steer_baseline_dataloader(self, k_shot, seed, batch_size=1, num_workers=8, shuffle=False):
        """
        use sub_context, sub_answer in demonstrations --> steer the behaviour of using contextual knowledge
        use sub_context, org_answer in demonstrations --> steer the behaviour of using parametric knowledge
        """

        np.random.RandomState(seed).shuffle(self.demonstration_groups_ids)
        selected_k_demonstrations_group_ids = self.demonstration_groups_ids[:k_shot]

        def collate_fn(batch):
            item = batch[0]
            demonstration_ids = [self.group2ids[gid][0] for gid in selected_k_demonstrations_group_ids]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]
            #  use sub_context for test examples -- it is knowledge conflict
            test_sub_context_prompt = self.verbalise_one_example(item, "sub_context", None, is_test=True)

            # use context prompt
            demonstration_ctx_key, demonstration_ans_key = "sub_context", "sub_answer"
            use_context_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                               demonstration_ans_key)
            use_context_prompt = use_context_prompt + test_sub_context_prompt
            use_context_inputs = self.tokenizer([use_context_prompt], return_tensors="pt")

            # use parameter prompt
            demonstration_ctx_key, demonstration_ans_key = "sub_context", "org_answer"
            use_parameter_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                                 demonstration_ans_key)
            use_parameter_prompt = use_parameter_prompt + test_sub_context_prompt
            use_parameter_inputs = self.tokenizer([use_parameter_prompt], return_tensors="pt")

            return {"use_context_input_ids": use_context_inputs["input_ids"],
                    "use_parameter_input_ids": use_parameter_inputs["input_ids"],
                    "sub_answers": [item["sub_answer"]],  # list[list[srt]]
                    "org_answers": [item["org_answer"]],
                    "sub_contexts": [item["sub_context"]],
                    "item_idx": item["idx"]}

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def collect_hiddens_dataloader(self, k_shot, seed, batch_size=1, num_workers=20, shuffle=False):
        """
        This dataloader is used to
        1. check the knowledge selection under different context,
        2. collect the hidden states that will lead to different knowledge selection behaviours
        3. record the logprob of generating org_answer and sub_answer
        """

        rng = np.random.RandomState(seed)

        def collate_fn(batch):
            item = batch[0]
            item_idx = item["idx"]

            demonstration_ids = self.sample_demonstrations(item_idx, k_shot, rng)
            sample_key = str(item_idx) + "-" + "-".join([str(idx) for idx in demonstration_ids])
            while sample_key in self.sample_keys:
                demonstration_ids = self.sample_demonstrations(item_idx, k_shot, rng)
                sample_key = str(item_idx) + "-" + "-".join([str(idx) for idx in demonstration_ids])

            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]

            dss_prompt = self.verbalise_demonstrations(demonstrations, "sub_context", "sub_answer")
            dso_prompt = self.verbalise_demonstrations(demonstrations, "sub_context", "org_answer")
            doo_prompt = self.verbalise_demonstrations(demonstrations, "org_context", "org_answer")

            test_example_prompt = self.verbalise_one_example(item, "sub_context", None, is_test=True)
            prompts = {"dss": dss_prompt + test_example_prompt,
                       "dso": dso_prompt + test_example_prompt,
                       "doo": doo_prompt + test_example_prompt}
            prompts_input_ids = {f"{k}_s": self.tokenizer([v], return_tensors="pt")["input_ids"]
                                 for k, v in prompts.items()}

            test_example_with_sub_answer = self.verbalise_one_example(item, "sub_context", "sub_answer")
            ss_prompts = {"dss": dss_prompt + test_example_with_sub_answer,
                          "dso": dso_prompt + test_example_with_sub_answer,
                          "doo": doo_prompt + test_example_with_sub_answer}
            ss_input_ids = {f"{k}_ss": self.tokenizer([v], return_tensors="pt")["input_ids"]
                            for k, v in ss_prompts.items()}

            test_example_with_org_answer = self.verbalise_one_example(item, "sub_context", "org_answer")
            so_prompts = {"dss": dss_prompt + test_example_with_org_answer,
                          "dso": dso_prompt + test_example_with_org_answer,
                          "doo": doo_prompt + test_example_with_org_answer}
            so_input_ids = {f"{k}_so": self.tokenizer([v], return_tensors="pt")["input_ids"]
                            for k, v in so_prompts.items()}

            return_dict = {"sub_answers": [item["sub_answer"]],
                           "org_answers": [item["org_answer"]],
                           "sub_contexts": [item["sub_context"]],
                           "item_idx": item_idx,
                           "demonstration_ids": demonstration_ids}
            return_dict.update(prompts_input_ids)
            return_dict.update(ss_input_ids)
            return_dict.update(so_input_ids)
            return return_dict

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def CAD_baseline_dataloader(self, k_shot, seed, batch_size=1, num_workers=8, shuffle=False):
        np.random.RandomState(seed).shuffle(self.demonstration_groups_ids)
        selected_k_demonstrations_group_ids = self.demonstration_groups_ids[:k_shot]

        def collate_fn(batch):
            item = batch[0]
            demonstration_ids = [self.group2ids[gid][0] for gid in selected_k_demonstrations_group_ids]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]
            test_sub_context_prompt = self.verbalise_one_example(item, "sub_context", None, is_test=True)

            # initial prompt
            demonstration_ctx_key, demonstration_ans_key = "org_context", "org_answer"
            initial_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                           demonstration_ans_key)
            initial_prompt = initial_prompt + test_sub_context_prompt
            initial_inputs = self.tokenizer([initial_prompt], return_tensors="pt")

            # without context prompt
            without_context_prompt = self.verbalise_close_book_example(item, is_test=True)
            without_context_inputs = self.tokenizer([without_context_prompt], return_tensors="pt")

            # close book prompt
            close_book_prompt = self.verbalise_close_book_demonstrations(demonstrations)
            close_book_prompt += self.verbalise_close_book_example(item, is_test=True)
            close_book_inputs = self.tokenizer([close_book_prompt], return_tensors="pt")

            return {"initial_input_ids": initial_inputs["input_ids"],
                    "without_context_input_ids": without_context_inputs["input_ids"],
                    "close_book_input_ids": close_book_inputs["input_ids"],
                    "sub_answers": [item["sub_answer"]],  # list[list[srt]]
                    "org_answers": [item["org_answer"]],
                    "sub_contexts": [item["sub_context"]],
                    "item_idx": item["idx"]}

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def DoLa_baseline_dataloader(self, k_shot, seed, batch_size=1, num_workers=8, shuffle=False):
        np.random.RandomState(seed).shuffle(self.demonstration_groups_ids)
        selected_k_demonstrations_group_ids = self.demonstration_groups_ids[:k_shot]

        def collate_fn(batch):
            item = batch[0]
            demonstration_ids = [self.group2ids[gid][0] for gid in selected_k_demonstrations_group_ids]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]
            test_sub_context_prompt = self.verbalise_one_example(item, "sub_context", None, is_test=True)

            demonstration_ctx_key, demonstration_ans_key = "org_context", "org_answer"
            initial_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                           demonstration_ans_key)
            initial_prompt = initial_prompt + test_sub_context_prompt
            initial_inputs = self.tokenizer([initial_prompt], return_tensors="pt")

            return {"initial_input_ids": initial_inputs["input_ids"],
                    "sub_answers": [item["sub_answer"]],  # list[list[srt]]
                    "org_answers": [item["org_answer"]],
                    "sub_contexts": [item["sub_context"]],
                    "item_idx": item["idx"]}

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def sample_demonstrations(self, test_item_idx, num_demonstrations, rng):
        test_item_group_idx = self.idx2group[test_item_idx]
        candidate_group_ids = [gi for gi in self.demonstration_groups_ids if gi != test_item_group_idx]
        selected_group_ids = rng.choice(candidate_group_ids, size=num_demonstrations, replace=False)
        selected_ids = []
        for group_idx in selected_group_ids:
            selected_ids.append(rng.choice(self.group2ids[group_idx], 1).item())
        return selected_ids

    def verbalise_one_example(self, example, ctx_key, ans_key, is_test=False):
        prompt = "context: " + example[ctx_key] + "\n"
        prompt = prompt + "question: " + example["question"] + "\n"
        if is_test:
            prompt = prompt + "answer:"
        else:
            prompt = prompt + "answer: " + example[ans_key][0] + "\n\n"
        return prompt

    def verbalise_demonstrations(self, demonstrations, ctx_key, ans_key):
        with_ctx_prompt = ""
        for demonstration in demonstrations:
            with_ctx_prompt = with_ctx_prompt + self.verbalise_one_example(demonstration, ctx_key, ans_key)
        return with_ctx_prompt

    def verbalise_close_book_example(self, example, is_test=False):
        prompt = "question: " + example["question"] + "\n"
        if is_test:
            prompt = prompt + "answer:"
        else:
            prompt = prompt + "answer: " + example["org_answer"][0] + "\n\n"
        return prompt

    def verbalise_close_book_demonstrations(self, demonstrations):
        prompt = ""
        for demonstration in demonstrations:
            prompt = prompt + self.verbalise_close_book_example(demonstration)
        return prompt

    def __getitem__(self, item):
        return self.data_for_iter[item]

    def __len__(self):
        return len(self.data_for_iter)


class EncodeREODQADataset(REODQADataset):

    def __init__(self, tokenizer, data, memorised_set, data_to_encode, demonstration_pool_size=128):
        super(EncodeREODQADataset, self).__init__(
            tokenizer, data, memorised_set, demonstration_pool_size=demonstration_pool_size,
            task="encode_and_save_hiddens"
        )
        self.data_to_encode = data_to_encode

    def get_dataloader(self, batch_size=1, num_workers=8, shuffle=False):
        assert batch_size == 1

        def collate_fn(batch):
            item = batch[0]
            item_idx = item["item_idx"]
            demonstration_ids = item["demonstration_ids"]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]

            if item["prompt_type"] == "dss_s":
                demonstration_ctx_key, demonstration_ans_key = "sub_context", "sub_answer"
            elif item["prompt_type"] == "dso_s":
                demonstration_ctx_key, demonstration_ans_key = "sub_context", "org_answer"
            elif item["prompt_type"] == "doo_s":
                demonstration_ctx_key, demonstration_ans_key = "org_context", "org_answer"
            else:
                raise ValueError

            prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key, demonstration_ans_key)
            prompt = prompt + self.verbalise_one_example(self.idx2item[item_idx], "sub_context", None, is_test=True)

            input_ids = self.tokenizer([prompt], return_tensors="pt")["input_ids"]
            item.update({"input_ids": input_ids})
            return item

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def get_hyperparameter_tune_dataloader(self, batch_size=1, num_workers=8, shuffle=False):
        assert batch_size == 1

        def collate_fn(batch):
            item = batch[0]
            item_idx = item["item_idx"]
            demonstration_ids = item["demonstration_ids"]
            demonstrations = [self.idx2item[idx] for idx in demonstration_ids]

            if item["prompt_type"] == "dss_s":
                demonstration_ctx_key, demonstration_ans_key = "sub_context", "sub_answer"
            elif item["prompt_type"] == "dso_s":
                demonstration_ctx_key, demonstration_ans_key = "sub_context", "org_answer"
            elif item["prompt_type"] == "doo_s":
                demonstration_ctx_key, demonstration_ans_key = "org_context", "org_answer"
            else:
                raise ValueError
            test_sub_context_prompt = self.verbalise_one_example(self.idx2item[item_idx], "sub_context", None,
                                                                 is_test=True)
            prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key, demonstration_ans_key)
            prompt = prompt + test_sub_context_prompt
            input_ids = self.tokenizer([prompt], return_tensors="pt")["input_ids"]

            # use context prompt
            demonstration_ctx_key, demonstration_ans_key = "sub_context", "sub_answer"
            use_context_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                               demonstration_ans_key)
            use_context_prompt = use_context_prompt + test_sub_context_prompt
            use_context_inputs = self.tokenizer([use_context_prompt], return_tensors="pt")

            # use parameter prompt
            demonstration_ctx_key, demonstration_ans_key = "sub_context", "org_answer"
            use_parameter_prompt = self.verbalise_demonstrations(demonstrations, demonstration_ctx_key,
                                                                 demonstration_ans_key)
            use_parameter_prompt = use_parameter_prompt + test_sub_context_prompt
            use_parameter_inputs = self.tokenizer([use_parameter_prompt], return_tensors="pt")

            # close book prompt
            close_book_prompt = self.verbalise_close_book_demonstrations(demonstrations)
            close_book_prompt += self.verbalise_close_book_example(self.idx2item[item_idx], is_test=True)
            close_book_inputs = self.tokenizer([close_book_prompt], return_tensors="pt")

            # without context prompt
            without_context_prompt = self.verbalise_close_book_example(self.idx2item[item_idx], is_test=True)
            without_context_inputs = self.tokenizer([without_context_prompt], return_tensors="pt")

            item.update({"input_ids": input_ids,
                         "use_context_input_ids": use_context_inputs["input_ids"],
                         "use_parameter_input_ids": use_parameter_inputs["input_ids"],
                         "close_book_input_ids": close_book_inputs["input_ids"],
                         "without_context_input_ids": without_context_inputs["input_ids"],
                         "sub_contexts": [self.idx2item[item_idx]["sub_context"]],
                         "sub_answers": [self.idx2item[item_idx]["sub_answer"]],
                         "org_answers": [self.idx2item[item_idx]["org_answer"]]})
            return item

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def __getitem__(self, item):
        return self.data_to_encode[item]

    def __len__(self):
        return len(self.data_to_encode)
