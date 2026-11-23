"""Diffusion Fuzzy System (DFS).

Implements the four modules described in the paper:

* Diffusion Fuzzification (DF): latent compression with (optionally multiple,
  adaptively selected) autoencoder / VQ-VAE pairs.
* Diffusion Fuzzy Rule Base (DFRB): one learnable diffusion model per rule
  chain (K chains, each rule chain contains the forward/reverse rules of the
  paper, Eq. 21/22).
* Diffusion Fuzzy Rule Chain Combination Inference Module (DFRCIM): weighted
  multi-path fusion of the latent states (Eq. 15/26/27).
* Diffusion Fuzzy Inference Engine (DFIE): membership normalisation and
  cross-path alignment at every reverse step (Eq. 24/25).

Training follows Algorithm 1 (parallel multi-path loss, Eq. 28) and generation
follows Algorithm 2.

Images and latents are kept in the ``[-1, 1]`` range used by LDM.  The heavy
SMC encoders are created lazily by :mod:`membership`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from loguru import logger

from membership import SimilarityMembership, normalize_membership
from representatives import load_delegates
from setting import options
from utils import GradualWarmupScheduler, csv_record, frozen_model


class FuzzyLatentDiffusion(nn.Module):
    """Latent multi-path diffusion model guided by fuzzy rules.

    Parameters
    ----------
    ldmodel:
        A frozen LDM (``ldm.models.diffusion.ddpm.LatentDiffusion``).  Its
        autoencoder(s) provide the latent space and its UNet initialises the
        K rule models.
    rule_num:
        Number of rule chains ``K`` (paper: 3).
    delegates:
        Representative (medoid) images in ``[-1, 1]``, shape ``[K, C, H, W]``.
    delegate_csv:
        Alternative to ``delegates``: a CSV produced by
        :mod:`representatives`.
    alpha:
        Balance between semantic (BERT) and feature (VGG) similarity.
    beta_1, beta_T, T:
        Forward diffusion schedule (paper: 1000 steps).
    condition:
        ``True`` for text-conditional datasets (MS COCO / CUB 200).
    ae_models:
        Optional list of additional autoencoders used by the adaptive AE
        selection (Eq. 17/18).  The LDM first stage is used by default.
    ae_representatives:
        ``[M, C, H, W]`` representative images, one per autoencoder.  When
        given together with ``ae_models`` the encoder/decoder pair is selected
        by maximum SMC (Eq. 17/18).
    cfg_scale:
        Classifier-free guidance scale used for text-conditional sampling.
    smc_every:
        Recompute the per-step memberships every ``smc_every`` reverse steps.
        The paper computes them at every step (default ``1``); increasing it
        trades exactness for speed.
    """

    def __init__(
        self,
        ldmodel,
        rule_num: int = 3,
        delegates=None,
        alpha: float = 0.7,
        beta_1: float = 0.0015,
        beta_T: float = 0.0155,
        T: int = 1000,
        root_path: str = "",
        condition: bool = False,
        membership=None,
        ae_models=None,
        ae_representatives=None,
        delegate_csv: str = None,
        cfg_scale: float = 7.5,
        smc_every: int = 1,
        device=None,
        **_legacy_kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device if device is not None
            else (options.device if options.device is not None else "cpu")
        )
        self.ldmodel = ldmodel
        frozen_model(ldmodel)

        self.rule_num = int(rule_num)
        self.alpha = float(alpha)
        self.condition = bool(condition)
        self.cfg_scale = float(cfg_scale)
        self.smc_every = max(1, int(smc_every))
        self.root_path = root_path
        self.T = int(T)

        # ---- rule base: one diffusion model per rule chain (paper Sec. III.C)
        self.rule_models = nn.ModuleList(
            [copy.deepcopy(ldmodel.model) for _ in range(self.rule_num)]
        )

        # ---- diffusion fuzzification: autoencoders + SMC membership
        if membership is None:
            membership = SimilarityMembership(alpha=self.alpha, device=self.device)
        self.membership = membership

        if ae_models is None:
            ae_models = [ldmodel.first_stage_model]
        self.ae_models = list(ae_models)
        for model in self.ae_models:
            model.to(self.device).eval()
            for param in model.parameters():
                param.requires_grad_(False)
        self.ae_representatives = (
            None if ae_representatives is None
            else ae_representatives.to(self.device)
        )
        self._ae_reference_features = None

        # ---- representatives / antecedents
        if delegates is None and delegate_csv is not None:
            delegates = load_delegates(delegate_csv)
        if delegates is None:
            raise ValueError(
                "FuzzyLatentDiffusion needs `delegates` or `delegate_csv` "
                "(run representatives.get_delegates first)."
            )
        delegates = list(delegates)[: self.rule_num]
        if len(delegates) < self.rule_num:
            raise ValueError(
                f"got {len(delegates)} delegates but rule_num={self.rule_num}; "
                "select one representative per rule chain first"
            )
        self.delegates = torch.stack(
            [delegate.detach().float().cpu() for delegate in delegates], dim=0
        )
        self._delegate_latents = None
        self._delegate_features = None
        self._delegate_noises = None

        self.init_schedule(beta_1, beta_T, T)

    # ------------------------------------------------------------------ #
    # diffusion schedule (paper Eq. 21/22)
    # ------------------------------------------------------------------ #
    def init_schedule(self, beta_1: float, beta_T: float, T: int) -> None:
        self.T = int(T)
        self.register_buffer(
            "betas", torch.linspace(beta_1, beta_T, self.T).double()
        )
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar)
        )

    def q_sample(self, x_start, t, noise):
        """Closed-form forward process ``q(z_t | z_0)`` (paper Eq. 5/6)."""
        t = t.to(self.sqrt_alphas_bar.device)
        sqrt_alpha = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1).to(
            device=x_start.device, dtype=x_start.dtype
        )
        sqrt_one_minus = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1).to(
            device=x_start.device, dtype=x_start.dtype
        )
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    # ------------------------------------------------------------------ #
    # adaptive autoencoder (paper Eq. 17/18)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _posterior_to_latent(posterior):
        if isinstance(posterior, tuple):
            posterior = posterior[0]
        if torch.is_tensor(posterior):
            # VQModelInterface.encode already returns the quantised latent.
            return posterior
        if hasattr(posterior, "mode"):
            return posterior.mode()
        return posterior

    def _encode_with(self, images: torch.Tensor, index: int) -> torch.Tensor:
        return self._posterior_to_latent(self.ae_models[index].encode(images))

    def _decode_with(self, latents: torch.Tensor, index: int) -> torch.Tensor:
        return self.ae_models[index].decode(latents)

    def select_ae(self, reference_image: torch.Tensor = None) -> int:
        """Select an encoder-decoder pair by maximum membership (Eq. 17/18)."""
        if len(self.ae_models) == 1 or self.ae_representatives is None:
            return 0
        if reference_image is None:
            return 0
        if self._ae_reference_features is None:
            self._ae_reference_features = self.membership.reference_features(
                self.ae_representatives
            )
        mu = self.membership.membership(reference_image, self._ae_reference_features)
        return int(mu.mean(dim=0).argmax())

    def encode(self, images: torch.Tensor, reference_image: torch.Tensor = None):
        index = self.select_ae(reference_image)
        return self._encode_with(images, index), index

    def decode(self, latents: torch.Tensor, reference_image: torch.Tensor = None):
        index = self.select_ae(reference_image)
        return self._decode_with(latents, index)

    # ------------------------------------------------------------------ #
    # antecedents (SMC memberships)
    # ------------------------------------------------------------------ #
    def prepare_delegates(self) -> None:
        """Encode the representatives and cache their SMC features."""
        if self._delegate_latents is not None:
            return
        images = self.delegates.to(self.device)
        with torch.no_grad():
            self._delegate_latents = self._encode_with(images, 0).detach()
            # SMC uses the reconstructed representatives (paper Eq. 12/13).
            reconstructed = self._decode_with(self._delegate_latents, 0)
        self._delegate_features = self.membership.reference_features(reconstructed)

        # Fixed noise for the "noise-perturbed representatives" d_k^(t)
        # (paper Eq. 10/11).  The same noise is used at every timestep so the
        # representative sequence stays temporally correlated.
        generator = torch.Generator().manual_seed(1234)
        noises = [
            torch.randn(self._delegate_latents[index].shape, generator=generator)
            for index in range(self._delegate_latents.shape[0])
        ]
        self._delegate_noises = torch.stack(noises, dim=0).to(self.device)

    def antecedent(self, images=None, latents=None, t: int = None):
        """Compute raw and normalised memberships of the current state.

        ``t=None`` uses the clean latents (training, Eq. 28).  With an integer
        ``t`` the noise-perturbed representatives ``d_k^(t)`` are used, as in
        DFIE (Eq. 24/25).
        """
        self.prepare_delegates()

        if latents is None:
            if images is None:
                raise ValueError("antecedent needs either `images` or `latents`")
            latents = self._encode_with(images.to(self.device), 0)
        latents = latents.to(self.device)

        if t is None:
            query_images = self._decode_with(latents, 0)
            references = self._delegate_features
        else:
            alpha_bar = self.alphas_bar[int(t)].to(
                device=self.device, dtype=self._delegate_latents.dtype
            )
            delegate_latents = (
                alpha_bar.sqrt() * self._delegate_latents
                + (1.0 - alpha_bar).sqrt() * self._delegate_noises
            )
            query_images = self._decode_with(latents, 0)
            reference_images = self._decode_with(delegate_latents, 0)
            references = self.membership.reference_features(reference_images)

        raw = self.membership.membership(query_images, references)
        return raw, normalize_membership(raw)

    # ------------------------------------------------------------------ #
    # training (Algorithm 1)
    # ------------------------------------------------------------------ #
    def _conditional_inputs(self, batch):
        if self.condition:
            return self.ldmodel.get_input(
                batch, "image", force_c_encode=True, cond_key="caption"
            )
        return self.ldmodel.get_input(batch, "image")[0], None

    def fuzzy_trainer(self, train_data, epoch: int) -> None:
        """Parallel multi-path training with SMC-weighted loss (Eq. 28).

        Unlike a per-rule sequential loop, all K consequents are optimised
        together in one gradient step, weighted by the normalised antecedents.
        """
        logger.info(
            f"rule_models epoch{epoch} rules{self.rule_num} parallel train start"
        )
        self.train()
        for model in self.rule_models:
            for param in model.parameters():
                param.requires_grad_(True)

        parameters = [param for model in self.rule_models for param in model.parameters()]
        optimizer = torch.optim.Adam(
            parameters, lr=options.learning_rate, weight_decay=1e-4
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=max(1, epoch), eta_min=0, last_epoch=-1
        )
        warmup = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=2.0,
            warm_epoch=max(1, min(5, epoch)),
            after_scheduler=cosine,
        )

        loss_path = self.root_path + "loss.csv"
        for current_epoch in tqdm(range(epoch), "DFS train"):
            for batch_index, batch in enumerate(
                tqdm(train_data, f"epoch{current_epoch}"), start=1
            ):
                latents, condition = self._conditional_inputs(batch)
                latents = latents.to(self.device)

                # Eq. 28: memberships of the clean sample to the
                # representatives, normalised afterwards (Eq. 25).
                raw_mu, mu = self.antecedent(latents=latents.detach())

                timesteps = torch.randint(
                    self.T, (latents.shape[0],), device=self.device
                )
                noise = torch.randn_like(latents)
                noisy_latents = self.q_sample(latents, timesteps, noise)

                loss = latents.new_zeros(())
                rule_losses = []
                for rule_index, model in enumerate(self.rule_models):
                    if condition is not None:
                        prediction = model(
                            noisy_latents, timesteps, context=condition
                        )
                    else:
                        prediction = model(noisy_latents, timesteps)

                    per_sample = (prediction - noise).pow(2).flatten(1).mean(dim=1)
                    weighted = (mu[:, rule_index] * per_sample).mean()
                    loss = loss + weighted
                    rule_losses.append(float(weighted.detach().cpu()))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                record = {
                    "epoch": str(current_epoch),
                    "batch": str(batch_index),
                    "loss": f"{loss.item():.6f}",
                }
                for rule_index, value in enumerate(rule_losses):
                    record[f"loss_rule_{rule_index + 1}"] = f"{value:.6f}"
                csv_record(loss_path, record)

            warmup.step()

        for model in self.rule_models:
            frozen_model(model)
        logger.info("rule_models parallel train done")

    # ------------------------------------------------------------------ #
    # generation (Algorithm 2)
    # ------------------------------------------------------------------ #
    def _predict_eps(self, model, latents, timestep, condition, unconditional):
        """Predict the noise with optional classifier-free guidance."""
        timesteps = torch.full(
            (latents.shape[0],), int(timestep), device=latents.device, dtype=torch.long
        )
        if condition is not None:
            if unconditional is not None and self.cfg_scale != 1.0:
                eps_uncond = model(latents, timesteps, context=unconditional)
                eps_cond = model(latents, timesteps, context=condition)
                return eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            return model(latents, timesteps, context=condition)
        return model(latents, timesteps)

    def _reverse_rule(self, latents, timestep, eps):
        """One reverse fuzzy rule, paper Eq. 22 / Algorithm 2 line 9."""
        beta = self.betas[int(timestep)].to(latents.device).float()
        return (latents - beta.sqrt() * eps) / (1.0 - beta).sqrt()

    def _timesteps(self, steps: int):
        steps = int(steps)
        if steps >= self.T:
            return list(range(self.T - 1, -1, -1))
        stride = max(1, self.T // steps)
        timesteps = list(range(self.T - 1, -1, -stride))[:steps]
        if timesteps[-1] != 0:
            timesteps[-1] = 0
        return timesteps

    @torch.no_grad()
    def fuzzy_sampler(
        self,
        batch=None,
        ddim_steps: int = None,
        latent_output: bool = False,
        batch_size: int = 1,
        return_history: bool = False,
    ):
        """Generate samples with the DFIE (paper Algorithm 2).

        Returns ``(reference_images, final_latents, generated_images, memberships)``
        when ``latent_output=True`` and only the generated images otherwise.
        ``memberships`` is the last normalised membership matrix ``[B, K]``
        (or the full history when ``return_history`` is set).
        """
        self.prepare_delegates()
        self.eval()

        if batch is not None and "image" in batch:
            batch_size = int(batch["image"].shape[0])
            reference_images = batch["image"].to(self.device)
        else:
            reference_images = None

        shape = (
            self.ldmodel.channels,
            self.ldmodel.image_size,
            self.ldmodel.image_size,
        )
        latents = torch.randn(
            batch_size, *shape, device=self.device, dtype=torch.float32
        )

        condition = unconditional = None
        if self.condition and batch is not None and "caption" in batch:
            condition = self.ldmodel.get_learned_conditioning(list(batch["caption"]))
            unconditional = self.ldmodel.get_learned_conditioning(
                [""] * batch_size
            )

        history = []
        memberships = None
        for index, timestep in enumerate(self._timesteps(ddim_steps or self.T)):
            if memberships is None or index % self.smc_every == 0:
                raw_mu, memberships = self.antecedent(
                    latents=latents, t=timestep
                )
                if return_history:
                    history.append(raw_mu.detach().cpu())

            fused = torch.zeros_like(latents)
            for rule_index, model in enumerate(self.rule_models):
                eps = self._predict_eps(
                    model, latents, timestep, condition, unconditional
                )
                rule_latents = self._reverse_rule(latents, timestep, eps)
                fused = fused + memberships[:, rule_index].view(-1, 1, 1, 1) * rule_latents
            latents = fused

        generated = self.decode(latents, reference_image=reference_images)

        if latent_output:
            memberships_out = history if return_history else memberships
            return reference_images, latents, generated, memberships_out
        return generated

    def forward(self, batch=None, latent_output: bool = False, **kwargs):
        return self.fuzzy_sampler(
            batch=batch, latent_output=latent_output, **kwargs
        )

    # SAM-compatible alias, used by the evaluation scripts.
    sample = fuzzy_sampler
