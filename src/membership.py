"""Similarity-based Membership Computation (SMC).

Implements Sec. III.B of the DFS paper (Eq. 12-14, 25):

    S_sem(z | d)  = cos(BERT(BLIP(Dec(z))), BERT(BLIP(Dec(d))))
    S_feat(z | d) = cos(VGG16(Dec(z)),        VGG16(Dec(d)))
    SMC(z | d)    = alpha * S_sem(z | d) + (1 - alpha) * S_feat(z | d)

Images are expected in the same normalised space as the LDM pipeline, i.e.
``[-1, 1]``.  They are converted to ``[0, 1]`` before being fed to BLIP,
BERT and VGG16.

All heavy models are loaded lazily: importing this module and constructing
:class:`SimilarityMembership` never touches the network.  The models are
downloaded the first time ``reference_features`` / ``membership`` is called
(``Salesforce/blip-image-captioning-base``, ``bert-base-uncased`` and the
torchvision VGG16 weights).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def normalize_membership(mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise fuzzy memberships across rules (paper Eq. 25).

    Cosine similarities may be negative; memberships of a fuzzy set are in
    ``[0, 1]`` so they are clamped before normalisation.
    """
    mu = mu.clamp_min(0.0)
    return mu / mu.sum(dim=-1, keepdim=True).clamp_min(eps)


class SimilarityMembership:
    """SMC implementation (BLIP caption -> BERT, plus VGG16 features).

    Parameters
    ----------
    alpha:
        Balance between semantic (BERT) and feature (VGG) similarity.
        Paper: alpha = 0.3 for LSUN, alpha = 0.7 for MS COCO / CUB 200.
    device:
        Torch device for the pretrained feature extractors.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        device=None,
        blip_name: str = "Salesforce/blip-image-captioning-base",
        bert_name: str = "bert-base-uncased",
        caption_max_new_tokens: int = 30,
        bert_max_length: int = 64,
    ) -> None:
        self.alpha = float(alpha)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.blip_name = blip_name
        self.bert_name = bert_name
        self.caption_max_new_tokens = caption_max_new_tokens
        self.bert_max_length = bert_max_length

        self._blip_processor = None
        self._blip_model = None
        self._bert_tokenizer = None
        self._bert_model = None
        self._vgg = None
        self._vgg_preprocess = None

    # ------------------------------------------------------------------ #
    # lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._vgg is not None:
            return

        from transformers import (
            BertModel,
            BertTokenizer,
            BlipForConditionalGeneration,
            BlipProcessor,
        )
        from torchvision.models import VGG16_Weights, vgg16

        self._blip_processor = BlipProcessor.from_pretrained(self.blip_name)
        self._blip_model = (
            BlipForConditionalGeneration.from_pretrained(self.blip_name)
            .to(self.device)
            .eval()
        )
        self._bert_tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self._bert_model = BertModel.from_pretrained(self.bert_name).to(self.device).eval()

        self._vgg = (
            vgg16(weights=VGG16_Weights.DEFAULT).to(self.device).eval()
        )
        self._vgg_preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        for module in (self._blip_model, self._bert_model, self._vgg):
            for param in module.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # image helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_pil(images: torch.Tensor):
        """Convert a batch of ``[-1, 1]`` tensors to a list of PIL images."""
        images = ((images.detach().float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
        arrays = (images.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy())
        return [Image.fromarray(array) for array in arrays]

    @torch.inference_mode()
    def caption(self, images: torch.Tensor):
        """Return one BLIP caption per image."""
        self._ensure_loaded()
        pil_images = self.to_pil(images)
        inputs = self._blip_processor(images=pil_images, return_tensors="pt").to(
            self.device
        )
        output = self._blip_model.generate(
            **inputs, max_new_tokens=self.caption_max_new_tokens
        )
        return self._blip_processor.batch_decode(output, skip_special_tokens=True)

    @torch.inference_mode()
    def semantic_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """768-d, L2-normalised BERT embedding of the BLIP caption."""
        captions = self.caption(images)
        tokens = self._bert_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.bert_max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self._bert_model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(embedding, dim=-1)

    @torch.inference_mode()
    def visual_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """1000-d, L2-normalised VGG16 feature."""
        self._ensure_loaded()
        images = ((images.detach().float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(
            0.0, 1.0
        )
        images = self._vgg_preprocess(images.to(self.device))
        features = self._vgg(images)
        return F.normalize(features, dim=-1)

    # ------------------------------------------------------------------ #
    # SMC
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def reference_features(self, images: torch.Tensor) -> dict:
        """Precompute the (semantic, visual) features of representatives."""
        return {
            "sem": self.semantic_embedding(images),
            "vis": self.visual_embedding(images),
        }

    @torch.inference_mode()
    def membership(self, images: torch.Tensor, references: dict) -> torch.Tensor:
        """Membership of every query image to every reference.

        Parameters
        ----------
        images: ``[B, C, H, W]`` in ``[-1, 1]``.
        references: ``{"sem": [K, 768], "vis": [K, 1000]}``.

        Returns
        -------
        ``[B, K]`` SMC values in ``[-1, 1]``.
        """
        self._ensure_loaded()
        sem = self.semantic_embedding(images)
        vis = self.visual_embedding(images)
        s_sem = sem @ references["sem"].T
        s_feat = vis @ references["vis"].T
        return self.alpha * s_sem + (1.0 - self.alpha) * s_feat
