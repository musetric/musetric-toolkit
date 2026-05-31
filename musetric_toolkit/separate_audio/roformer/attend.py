# Based on BS-RoFormer implementation by Phil Wang (lucidrains)
# Original: https://github.com/lucidrains/BS-RoFormer
# License: MIT
# Modified for Musetric project

import logging
import warnings

import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend

_backends_logged = {"value": False}


def log_selected_backend(q, k, v, backends):
    if not _backends_logged["value"]:
        backend_names = [b.name for b in backends]
        logging.debug(f"Available SDPA backends for attention: {backend_names}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for backend in backends:
                try:
                    with torch.nn.attention.sdpa_kernel([backend]):
                        functional.scaled_dot_product_attention(
                            q[:1], k[:1], v[:1], dropout_p=0.0
                        )
                    logging.debug(f"Selected SDPA backend: {backend.name}")
                    break
                except Exception:
                    logging.debug(
                        "Failed to probe SDPA backend %s", backend.name, exc_info=True
                    )
                    continue
            else:
                logging.debug("Selected SDPA backend: MATH (fallback)")

        _backends_logged["value"] = True


class Attend(nn.Module):
    def __init__(self, dropout, flash):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

    def flash_attn(self, q, k, v):
        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        log_selected_backend(q, k, v, backends)

        with torch.nn.attention.sdpa_kernel(backends):
            return functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)

        if not _backends_logged["value"]:
            logging.debug("Using manual attention computation (matmul-based)")
            _backends_logged["value"] = True
        # MatMul (not einsum): MatMul is supported on every onnxruntime EP
        # (WebGPU / DirectML / CUDA / CoreML / TensorRT) and is the form their
        # attention fusions match; einsum coverage on DirectML/CoreML is spotty.
        # sim = q @ kᵀ ; out = attn @ v  (math-identical to the einsum path).
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return torch.matmul(attn, v)
