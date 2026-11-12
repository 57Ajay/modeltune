"""Pick the device and number format for whatever machine you are on.

Why this matters (chapter 04):
- Colab's free T4 is a Turing GPU (compute capability 7.5). It has fast fp16 tensor cores but
  NO native bf16. fp16 has a tiny range (max 65504), so it needs loss scaling.
- Ampere and newer (A100, A10, L4, RTX 30xx/40xx, H100: capability >= 8.0) support bf16, which
  has the same range as fp32. Prefer it whenever you can.
- CPU / Apple MPS: use fp32. Fine for smoke tests, too slow for real training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Hardware:
    device: str  # cuda | mps | cpu
    name: str
    dtype: torch.dtype  # the 16-bit (or 32-bit) format we compute in
    memory_gb: float

    @property
    def bf16(self) -> bool:
        return self.dtype == torch.bfloat16

    @property
    def fp16(self) -> bool:
        return self.dtype == torch.float16

    def __str__(self) -> str:
        dtype = str(self.dtype).replace("torch.", "")
        mem = f", {self.memory_gb:.1f} GB" if self.memory_gb else ""
        return f"{self.name} ({self.device}, compute dtype={dtype}{mem})"


def detect() -> Hardware:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        # Do not use torch.cuda.is_bf16_supported() here: on a T4 it can report True because bf16
        # can be *emulated*, which is slow. We want native bf16, i.e. Ampere (8.0) or newer.
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        props = torch.cuda.get_device_properties(0)
        return Hardware("cuda", props.name, dtype, props.total_memory / 1024**3)
    if torch.backends.mps.is_available():
        return Hardware("mps", "Apple MPS", torch.float32, 0.0)
    return Hardware("cpu", "CPU", torch.float32, 0.0)


def peak_memory_gb() -> float:
    """Peak GPU memory allocated by PyTorch so far (what you compare against your VRAM)."""
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
