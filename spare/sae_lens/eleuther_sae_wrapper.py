import torch
from spare.sae_lens.sae import SAE, SAEConfig
from spare.sae.utils import decoder_impl


class EleutherSae(SAE):

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)

        self.num_latents = self.cfg.d_sae

    def pre_acts(self, hiddens):
        return self.encode(hiddens)

    def decode(
            self,
            feature_acts,
            top_indices=None
    ):
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""

        # func_vec = sae.decode(top_acts, top_indices.expand(top_acts.shape[0], -1))
        # top_acts, top_indices.expand(top_acts.shape[0], -1)

        # y = decoder_impl(top_indices, feature_acts.to(self.dtype), self.W_dec.mT)
        # return y + self.b_dec

        # # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(
            self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
        )

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out
