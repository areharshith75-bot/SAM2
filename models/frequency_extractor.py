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


def local_amplitude_spectrum(x: torch.Tensor, patch: int = 16, eps: float = 1e-8) -> torch.Tensor:
    """Extension (not in the paper): patch-wise log-amplitude spectrum.

    The global |FFT2(I)| places *frequencies* on the token grid, so the frequency token at (i, j) has no
    relation to the image patch at (i, j) it is added to. Here every non-overlapping `patch` x `patch` image
    patch gets its own 2-D FFT, laid out in place, so SAM's patch embedding turns it into a token that
    describes the local texture of exactly the patch at the same grid position.
    """
    b, c, h, w = x.shape
    assert h % patch == 0 and w % patch == 0, (h, w, patch)
    with torch.autocast(device_type=x.device.type, enabled=False):
        t = x.float().view(b, c, h // patch, patch, w // patch, patch).permute(0, 1, 2, 4, 3, 5)
        f = torch.fft.fft2(t, dim=(-2, -1))
        amp = torch.log1p(torch.sqrt(f.real.square() + f.imag.square() + eps))
        amp = amp.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        mu = amp.mean(dim=(-2, -1), keepdim=True)
        sd = amp.std(dim=(-2, -1), keepdim=True)
        return (amp - mu) / (sd + 1e-6)


class FrequencyExtractor(nn.Module):
    """Embeds image I and its amplitude |F(I)| with SAM's (frozen, shared) patch embedding + pos_embed.

    Returns E_img^(0), E_freq^(0) each of shape (B, H/16, W/16, 768).
    """

    def __init__(self, patch_embed: nn.Module, pos_embed: nn.Parameter | None, shift: bool = False,
                 mode: str = "global"):
        super().__init__()
        assert mode in ("global", "local"), mode
        self.mode = mode
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.shift = shift

    def _embed(self, x):
        e = self.patch_embed(x)
        if self.pos_embed is not None:
            e = e + self.pos_embed
        return e

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "local":
            return local_amplitude_spectrum(x, self.patch_embed.proj.kernel_size[0]).to(x.dtype)
        return amplitude_spectrum(x, shift=self.shift).to(x.dtype)

    def forward(self, x: torch.Tensor, x_freq: torch.Tensor | None = None):
        """`x_freq` (analysis only): take the frequency stream from a different input than the image stream."""
        freq = self.spectrum(x if x_freq is None else x_freq)
        return self._embed(x), self._embed(freq)
