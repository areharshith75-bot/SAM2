"""Extension 4 - Random-style augmentation (GIN: Global Intensity Non-linear augmentation).

Ouyang et al., "Causality-inspired single-source domain generalization for medical image segmentation",
IEEE TMI 2022 (https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization).

Every call builds a fresh, randomly initialised shallow conv net (3x3 convs + LeakyReLU) and passes the
batch through it. This keeps shapes (anatomy) intact but changes intensity, contrast and local texture in a
random non-linear way, so a single source hospital is turned into many synthetic "styles". The output is
blended with the original image (random alpha) and re-standardised per image/channel, like the normal input.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def gin(x: torch.Tensor, n_layers: int = 4, width: int = 2, kernel: int = 3, p: float = 0.5) -> torch.Tensor:
    """x: (B, C, H, W) z-scored batch on GPU. Each image is re-styled with probability p."""
    b, c, h, w = x.shape
    sel = torch.rand(b, device=x.device) < p
    if not sel.any():
        return x
    z = x[sel].float()
    n = z.shape[0]
    # independent random network per image via grouped convolutions
    y = z.reshape(1, n * c, h, w)
    cin = c
    for i in range(n_layers):
        cout = c if i == n_layers - 1 else width
        wgt = torch.randn(n * cout, cin, kernel, kernel, device=x.device)
        bias = torch.randn(n * cout, device=x.device) * 0.1
        y = F.conv2d(y, wgt, bias, padding=kernel // 2, groups=n)
        if i < n_layers - 1:
            y = F.leaky_relu(y, 0.2)
        cin = cout
    y = y.reshape(n, c, h, w)
    y = (y - y.mean((2, 3), keepdim=True)) / (y.std((2, 3), keepdim=True) + 1e-6)
    a = torch.rand(n, 1, 1, 1, device=x.device)
    y = a * y + (1 - a) * z
    y = (y - y.mean((2, 3), keepdim=True)) / (y.std((2, 3), keepdim=True) + 1e-6)
    out = x.clone()
    out[sel] = y.to(x.dtype)
    return out
