"""FSAM Sec. 2.1: Fourier amplitude input + dual (spatial / frequency) patch embedding."""
import torch
import torch.nn as nn


def amplitude_spectrum(x: torch.Tensor, eps: float = 1e-8, shift: bool = False) -> torch.Tensor:
    """log1p(|FFT2(x)|), standardised per sample & channel. Always computed in float32."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        f = torch.fft.fft2(x.float(), dim=(-2, -1))
        if shift:
            f = torch.fft.fftshift(f, dim=(-2, -1))
        amp = torch.sqrt(f.real.square() + f.imag.square() + eps)
        amp = torch.log1p(amp)
        mu = amp.mean(dim=(-2, -1), keepdim=True)
        sd = amp.std(dim=(-2, -1), keepdim=True)
        return (amp - mu) / (sd + 1e-6)


class FrequencyExtractor(nn.Module):
    """Embeds image I and its amplitude |F(I)| with SAM's (frozen, shared) patch embedding + pos_embed.

    Returns E_img^(0), E_freq^(0) each of shape (B, H/16, W/16, 768).
    """

    def __init__(self, patch_embed: nn.Module, pos_embed: nn.Parameter | None, shift: bool = False):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.shift = shift

    def _embed(self, x):
        e = self.patch_embed(x)
        if self.pos_embed is not None:
            e = e + self.pos_embed
        return e

    def forward(self, x: torch.Tensor):
        freq = amplitude_spectrum(x, shift=self.shift).to(x.dtype)
        return self._embed(x), self._embed(freq)
