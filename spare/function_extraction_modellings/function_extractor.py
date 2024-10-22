import torch


class FunctionExtractor(torch.nn.Module):
    def __init__(self, num_activations):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(num_activations))
        torch.nn.init.uniform_(self.weight, a=1e-5, b=2e-5)
        self.relu = torch.nn.ReLU()

    def forward(
            self,
            top_indices,
            sae,
            max_to_remove=None,
            max_to_add=None,
            return_top_acts_and_top_indices=False,
            sae_type="TopK"
    ):

        if sae_type != "TopK":
            select_weight = torch.zeros_like(self.weight.data)
            select_weight[top_indices] = self.weight.data[top_indices]
            if max_to_remove is not None:
                max_to_remove_select_weight = torch.zeros_like(self.weight.data)
                max_to_remove_select_weight[top_indices] = max_to_remove[0][top_indices]
                select_weight = torch.clamp(select_weight, max=max_to_remove)
            elif max_to_add is not None:
                max_to_add_select_weight = torch.zeros_like(self.weight.data)
                max_to_add_select_weight[top_indices] = max_to_add[0][top_indices]
                select_weight = torch.clamp(select_weight - max_to_add_select_weight, min=0)
            func_vec = sae.decode(select_weight)
            return func_vec

        top_acts = self.weight.index_select(index=top_indices, dim=0)
        top_acts = self.relu(top_acts)
        if max_to_remove is not None:
            max_to_remove = max_to_remove.index_select(index=top_indices, dim=1)
            top_acts = torch.clamp(top_acts, max=max_to_remove)
        elif max_to_add is not None:
            max_to_add = max_to_add.index_select(index=top_indices, dim=1)
            top_acts = torch.clamp(top_acts - max_to_add, min=0)
        else:
            top_acts = top_acts.unsqueeze(dim=0)

        func_vec = sae.decode(top_acts, top_indices.expand(top_acts.shape[0], -1))
        if return_top_acts_and_top_indices:
            return func_vec, top_acts, top_indices
        else:
            return func_vec

    def load_weight(self, weight):
        self.weight.data = weight
