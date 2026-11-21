import torch

from models.frequency_extractor import local_amplitude_spectrum


def test_local_spectrum_is_patch_local():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    a = local_amplitude_spectrum(x, 16)
    assert a.shape == x.shape and torch.isfinite(a).all()
    # changing one 16x16 patch must change only that patch of the (pre-normalisation) local spectrum
    y = x.clone(); y[:, :, 16:32, 32:48] += 5 * torch.randn(2, 3, 16, 16)
    d = (local_amplitude_spectrum(y, 16) - a).abs()
    inside = d[:, :, 16:32, 32:48].mean()
    outside = torch.cat([d[:, :, :16].flatten(), d[:, :, 32:].flatten()]).std()
    assert inside > 10 * outside  # outside differs only through the global standardisation shift


def test_local_matches_per_patch_fft():
    x = torch.randn(1, 1, 32, 32)
    a = local_amplitude_spectrum(x, 16)
    ref = torch.log1p(torch.fft.fft2(x[0, 0, 16:32, :16]).abs())
    full = torch.log1p(torch.sqrt(torch.fft.fft2(x[0, 0].view(2, 16, 2, 16).permute(0, 2, 1, 3)).abs() ** 2 + 1e-8))
    full = full.permute(0, 2, 1, 3).reshape(32, 32)
    z = (full - full.mean()) / (full.std() + 1e-6)
    assert torch.allclose(a[0, 0], z, atol=1e-4)
    assert torch.allclose(full[16:32, :16], ref, atol=1e-3)
