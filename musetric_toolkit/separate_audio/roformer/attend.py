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
        # 0 = off. Export-only knob; see the matmul path in forward().
        self.q_block = 0

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

        # q_block splits the QUERY axis, not the key axis, so no online softmax is
        # needed: softmax normalizes each query row over the full key axis on its
        # own, and the rows of a block never see the other blocks. The result is
        # the same values, computed with a peak score tensor of [b,h,q_block,n]
        # instead of [b,h,n,n]. For the time-attention layers of this model that
        # is 480*q_block*n*2 bytes instead of 480*n²*2 — the difference between
        # fitting in a mobile storage buffer and not. Costs the same FLOPs.
        if self.q_block and q.shape[-2] > self.q_block:
            outputs = []
            for start in range(0, q.shape[-2], self.q_block):
                q_part = q[..., start : start + self.q_block, :]
                sim = torch.matmul(q_part, k.transpose(-1, -2)) * scale
                attn = self.attn_dropout(sim.softmax(dim=-1))
                outputs.append(torch.matmul(attn, v))
            return torch.cat(outputs, dim=-2)

        sim = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return torch.matmul(attn, v)
