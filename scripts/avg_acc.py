import json
from spare.utils import PROJ_DIR
import numpy as np
import argparse


def get_results(model_name, data_name):
    if model_name == "Meta-Llama-3-8B":
        use_context_files = [
            f"Meta-Llama-3-8B-{data_name}-13,14,15,16-0.07-2.0-grouped_activations-mutual_information-True-True-42",
            f"Meta-Llama-3-8B-{data_name}-13,14,15,16-0.07-2.0-grouped_activations-mutual_information-True-True-43",
            f"Meta-Llama-3-8B-{data_name}-13,14,15,16-0.07-2.0-grouped_activations-mutual_information-True-True-44",
            f"Meta-Llama-3-8B-{data_name}-13,14,15,16-0.07-2.0-grouped_activations-mutual_information-True-True-45",
            f"Meta-Llama-3-8B-{data_name}-13,14,15,16-0.07-2.0-grouped_activations-mutual_information-True-True-46",
        ]
        use_parameter_files = use_context_files
    elif model_name == "Llama-2-7b-hf":
        use_context_files = [
            f"Llama-2-7b-hf-{data_name}-12,13,14,15-0.07-2.0-grouped_activations-mutual_information-True-True-42",
            f"Llama-2-7b-hf-{data_name}-12,13,14,15-0.07-2.0-grouped_activations-mutual_information-True-True-43",
            f"Llama-2-7b-hf-{data_name}-12,13,14,15-0.07-2.0-grouped_activations-mutual_information-True-True-44",
            # seed = 45 exceeds context length
            f"Llama-2-7b-hf-{data_name}-12,13,14,15-0.07-2.0-grouped_activations-mutual_information-True-True-46",
        ]
        use_parameter_files = use_context_files
    elif model_name == "gemma-2-9b":
        use_context_files = [
            f"gemma-2-9b-{data_name}-23,24,25,26-0.01-3.0-grouped_activations-mutual_information-True-True-42",
            f"gemma-2-9b-{data_name}-23,24,25,26-0.01-3.0-grouped_activations-mutual_information-True-True-43",
            f"gemma-2-9b-{data_name}-23,24,25,26-0.01-3.0-grouped_activations-mutual_information-True-True-44",
            f"gemma-2-9b-{data_name}-23,24,25,26-0.01-3.0-grouped_activations-mutual_information-True-True-45",
            f"gemma-2-9b-{data_name}-23,24,25,26-0.01-3.0-grouped_activations-mutual_information-True-True-46",
        ]
        use_parameter_files = [
            f"gemma-2-9b-{data_name}-23,24,25,29,30,31-0.01-1.8-grouped_activations-mutual_information-True-True-42",
            f"gemma-2-9b-{data_name}-23,24,25,29,30,31-0.01-1.8-grouped_activations-mutual_information-True-True-43",
            f"gemma-2-9b-{data_name}-23,24,25,29,30,31-0.01-1.8-grouped_activations-mutual_information-True-True-44",
            f"gemma-2-9b-{data_name}-23,24,25,29,30,31-0.01-1.8-grouped_activations-mutual_information-True-True-45",
            f"gemma-2-9b-{data_name}-23,24,25,29,30,31-0.01-1.8-grouped_activations-mutual_information-True-True-46",
        ]
    else:
        raise ValueError

    use_context_results = []
    for file in use_context_files:
        cur_result = json.load(open(PROJ_DIR / "spare_outputs" / file, "r"))
        use_context_results.append(cur_result)

    use_parameter_results = []
    for file in use_parameter_files:
        cur_result = json.load(open(PROJ_DIR / "spare_outputs" / file, "r"))
        use_parameter_results.append(cur_result)

    use_context_em = []
    for cur_result in use_context_results:
        use_context_em.append(sum(cur_result["all_sub_scores"]) / len(cur_result["all_sub_scores"]) * 100)
    use_parameter_em = []
    for cur_result in use_parameter_results:
        use_parameter_em.append(sum(cur_result["all_org_scores"]) / len(cur_result["all_org_scores"]) * 100)

    print("use_context_em")
    print(use_context_em)
    print(f"avg: {np.mean(use_context_em):.2f}")
    print(f"std: {np.std(use_context_em):.2f}")

    print("use_parameter_em")
    print(use_parameter_em)
    print(f"avg: {np.mean(use_parameter_em):.2f}")
    print(f"std: {np.std(use_parameter_em):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    args = parser.parse_args()
    get_results(args.model_name, args.data_name)


if __name__ == '__main__':
    main()
