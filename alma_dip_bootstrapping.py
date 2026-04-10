"""
alma_dip_bootstrapping.py
=========================
Deep Image Prior (DIP) for ALMA radio interferometric imaging.

Supports two computation paths that are selected automatically based on the
number of visibility points N:

  - **Small path**  (N ≤ auto_threshold_n):
      Precomputes the full N × H and N × W trig tables once and keeps them
      on-device.  Fast per-iteration, but memory cost is O(N × max(H, W)).

  - **Memory path** (N > auto_threshold_n):
      Draws stratified mini-batches via ``VisibilitySampler`` and corrects
      for the non-uniform sampling with Self-Normalised Importance Sampling
      (SNIS), keeping GPU memory bounded.

Both paths share the same U-Net generator (``DIPUNet``), Student-t data
fidelity loss (``StudentTLoss``), and TV regulariser.

Bootstrap uncertainty estimation (``bootstrap_reconstruct``) reruns DIP with
Poisson- or Bayesian-reweighted visibilities and returns per-pixel mean, std,
and percentile maps.

Public API
----------
    pick_device(explicit)           → torch.device
    DIPConfig                       dataclass with all tunable knobs
    reconstruct_dip(...)            → (image, dirty, psf)   [np.ndarray × 3]
    bootstrap_reconstruct(...)      → dict with keys mean / std / p_lo / p_hi
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def pick_device(explicit: Optional[str] = None) -> torch.device:
    """Return a torch device, auto-detecting MPS → CUDA → CPU when *explicit* is None."""
    if explicit is not None:
        return torch.device(explicit)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DIPConfig:
    """
    Unified configuration for both computation paths and the bootstrap.

    Common knobs
    ------------
    seed          : RNG seed (applied to both torch and numpy).
    num_iters     : Number of gradient steps.
    lr            : Adam learning rate.
    tv_weight     : Isotropic total-variation regularisation strength.
    positivity    : Enforce non-negative image via softplus activation.
    input_depth   : Number of channels in the fixed random input tensor z.
    base_channels : Base channel width of the U-Net (doubles each level, capped at 8×).
    depth         : Number of encoder/decoder levels in the U-Net.
    dropout       : Spatial dropout rate (0 = disabled).
    cell_size_arcsec : Image pixel scale in arcseconds.
    nu            : Student-t degrees of freedom (lower ⇒ heavier tails, more outlier
                    robustness; ν → ∞ recovers Gaussian NLL).
    learn_sigma   : If True, the scale parameter σ is a free parameter.
    init_sigma    : Initial value of σ.
    out_every     : Log a progress line every this many iterations.

    Path selection
    --------------
    force_mode        : ``"auto"`` | ``"small"`` | ``"memory"``.
    auto_threshold_n  : When *force_mode* is ``"auto"``, use the small path for
                        N ≤ this threshold and the memory path otherwise.

    Memory-path knobs  (ignored by the small path)
    -----------------------------------------------
    batch_vis         : Mini-batch size drawn from the visibility sampler.
    stream_chunk_vis  : Chunk size for the streaming dirty/PSF computation.
                        Defaults to *batch_vis* when None.
    radial_bins       : Number of radial annuli for stratified sampling.
    radial_quantiles  : Use quantile-equal (True) vs log-space (False) bin edges.
    angular_bins      : Angular sectors (0 = disabled).
    low_k_bins        : Number of innermost radial bins to boost (short baselines).
    low_k_boost       : Multiplicative boost applied to the low-k bins.
    alpha_uniform     : Mixture weight for pure-uniform component.
    alpha_radial      : Mixture weight for radial-stratified component.
    alpha_inv_radius  : Mixture weight for ∝ 1/ρ^γ component.
    alpha_weight      : Mixture weight for ∝ visibility-weight component.
    alpha_angular     : Mixture weight for angular-stratified component.
    inv_radius_gamma  : Exponent γ in the inverse-radius density.
    importance_snis   : Use SNIS (True) or plain IS (False) for gradient weighting.
    adapt_every       : Reweight radial bins by per-bin EMA loss every N steps (0 = off).
    adapt_ema         : EMA decay for adaptive reweighting.
    adapt_power       : Power applied to the EMA loss before normalising.

    Quantile strategy (for very large N)
    -------------------------------------
    quantile_mode           : ``"auto"`` | ``"full"`` | ``"sample"`` | ``"hist"`` | ``"logspace"``.
    quantile_full_threshold : Exact torch.quantile when N ≤ this value.
    quantile_sample_max     : Subsample this many points for approximate quantiles.
    hist_bins_for_quantiles : Number of histogram bins for histogram-based quantiles.
    """

    # Optimisation
    seed: int = 123
    num_iters: int = 2500
    lr: float = 1e-3

    # Regularisation
    tv_weight: float = 5e-5
    positivity: bool = True

    # Generator (U-Net)
    input_depth: int = 32
    base_channels: int = 64
    depth: int = 5
    dropout: float = 0.0

    # Measurement / likelihood
    cell_size_arcsec: float = 0.10
    nu: float = 3.0
    learn_sigma: bool = False
    init_sigma: float = 1.0
    out_every: int = 200

    # Path selection
    force_mode: str = "auto"
    auto_threshold_n: int = 5_000_000

    # Memory-path: batching
    batch_vis: int = 16384
    stream_chunk_vis: Optional[int] = None

    # Memory-path: sampler
    sampler_type: str = "mixture"
    radial_bins: int = 48
    radial_quantiles: bool = True
    angular_bins: int = 0
    low_k_bins: int = 3
    low_k_boost: float = 3.0

    # Memory-path: mixture weights (normalised at runtime)
    alpha_uniform: float = 0.10
    alpha_radial: float = 0.55
    alpha_inv_radius: float = 0.35
    alpha_weight: float = 0.0
    alpha_angular: float = 0.0

    # Memory-path: inverse-radius component
    inv_radius_gamma: float = 1.0
    radius_eps_frac: float = 1e-6

    # Memory-path: importance sampling
    importance_snis: bool = True

    # Memory-path: adaptive bin reweighting
    adapt_every: int = 0
    adapt_ema: float = 0.9
    adapt_power: float = 1.0

    # Quantile strategy
    quantile_mode: str = "auto"
    quantile_full_threshold: int = 5_000_000
    quantile_sample_max: int = 2_000_000
    hist_bins_for_quantiles: int = 262_144


# ---------------------------------------------------------------------------
# Regularisation: isotropic total variation
# ---------------------------------------------------------------------------

def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Isotropic total-variation loss (smooth approximation via eps guard)."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (
        torch.sum(torch.sqrt(dx * dx + eps))
        + torch.sum(torch.sqrt(dy * dy + eps))
    )


# ---------------------------------------------------------------------------
# Data fidelity: Student-t negative log-likelihood
# ---------------------------------------------------------------------------

class StudentTLoss(nn.Module):
    """
    Bivariate Student-t negative log-likelihood for complex visibilities.

    The residual r = pred − target has shape [N, 2] (real, imag).  The NLL
    per visibility is::

        nll_i = 0.5 (ν + d) log(1 + w_i ‖r_i‖² / (ν σ²))  +  [0.5 d log σ if learned]

    with d = 2 (complex dimension).

    Parameters
    ----------
    nu          : Degrees of freedom.  Lower ⇒ heavier tails.
    learn_sigma : Optimise σ jointly with the network.
    init_sigma  : Initial scale (ignored when loading a checkpoint).
    reduction   : ``"mean"``, ``"sum"``, or ``"none"`` (returns per-sample NLL).
    """

    def __init__(
        self,
        nu: float = 3.0,
        learn_sigma: bool = False,
        init_sigma: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.nu = float(nu)
        self.reduction = reduction
        self.learn_sigma = learn_sigma
        self.d = 2      # complex dimension
        self.eps = 1e-8

        log_sigma_init = torch.log(torch.tensor(float(init_sigma)))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma_init)
        else:
            self.register_buffer("log_sigma", log_sigma_init)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : shape [N, 2]  (real and imaginary parts).
        weight       : per-visibility weights w_i ≥ 0, shape [N].  Defaults to ones.

        Returns
        -------
        Scalar loss (or [N] tensor when reduction == "none").
        """
        r = pred - target                             # [N, 2]
        r2 = torch.sum(r * r, dim=-1)                # [N]
        if weight is None:
            weight = torch.ones_like(r2)

        sigma2 = torch.exp(2.0 * self.log_sigma)
        scaled = 1.0 + (weight * r2) / (self.nu * sigma2 + self.eps)
        nll = 0.5 * (self.nu + self.d) * torch.log(scaled + self.eps)

        if self.learn_sigma:
            nll = nll + 0.5 * self.d * self.log_sigma

        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "mean":
            return nll.mean()
        return nll  # "none"


# ---------------------------------------------------------------------------
# Generator: U-Net with reflection padding and GroupNorm
# ---------------------------------------------------------------------------

class ConvGNAct(nn.Module):
    """Conv3×3 (reflection-padded) → GroupNorm → LeakyReLU → optional Dropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        groups = max(1, min(8, out_ch))   # at most 8 groups; fall back to 1 for narrow layers
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.gn(self.conv(self.pad(x)))))


class Down(nn.Module):
    """Two ConvGNAct blocks followed by 2× average-pooling (encoder stage)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_feature_map, skip_connection)."""
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        return self.pool(x), skip


class Up(nn.Module):
    """Bilinear 2× upsample → concatenate skip → two ConvGNAct blocks (decoder stage)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvGNAct(in_ch + skip_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad to match skip if odd spatial dimensions cause a 1-pixel mismatch.
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class DIPUNet(nn.Module):
    """
    U-Net generator used as the DIP prior.

    A fixed random tensor z (shape [1, input_depth, H, W]) is passed through
    the network to produce the reconstructed image.  The network is never
    given the data directly; the image emerges solely from gradient descent
    fitting the visibility data.

    Architecture
    ------------
    - *depth* encoder stages, each doubling the channel count (capped at 8×base_ch).
    - Two-block bottleneck at the coarsest resolution.
    - Symmetric decoder with bilinear upsampling and skip connections.
    - Final Conv3×3 to a single-channel output (pre-softplus for positivity).
    """

    def __init__(
        self,
        input_depth: int = 32,
        base_ch: int = 64,
        depth: int = 5,
        dropout: float = 0.0,
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        chs = [base_ch * min(2 ** i, 8) for i in range(depth)]

        # Encoder
        self.downs = nn.ModuleList()
        in_c = input_depth
        for i in range(depth):
            self.downs.append(Down(in_c, chs[i], dropout))
            in_c = chs[i]

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvGNAct(chs[-1], chs[-1], dropout),
            ConvGNAct(chs[-1], chs[-1], dropout),
        )

        # Decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            out_c = chs[i - 1] if i > 0 else base_ch
            self.ups.append(Up(chs[i], chs[i], out_c, dropout))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=0),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = z
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x, skips[-(i + 1)])
        return self.final(x)


# ---------------------------------------------------------------------------
# Shared grid utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def _grid_coords(
    H: int,
    W: int,
    cell_size_rad: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return 1-D sky-coordinate arrays l [W] and m [H] in radians,
    centred at zero with the given pixel scale.
    """
    l = (torch.arange(W, device=device, dtype=dtype) - (W // 2)) * cell_size_rad
    m = (torch.arange(H, device=device, dtype=dtype) - (H // 2)) * cell_size_rad
    return l, m


# ---------------------------------------------------------------------------
# Small path: precomputed full trig tables
# ---------------------------------------------------------------------------

def precompute_uv_phases_full(
    uv: torch.Tensor,
    H: int,
    W: int,
    cell_size_rad: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for every (u,v) × (l,m) pair.

    Returns
    -------
    Cul : cos(2π u l)  shape [N, W]
    Sul : sin(2π u l)  shape [N, W]
    Cvm : cos(2π v m)  shape [N, H]
    Svm : sin(2π v m)  shape [N, H]
    """
    u = uv[:, 0].to(device=device, dtype=dtype)
    v = uv[:, 1].to(device=device, dtype=dtype)
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)

    ul = 2.0 * math.pi * (u[:, None] * l[None, :])   # [N, W]
    vm = 2.0 * math.pi * (v[:, None] * m[None, :])   # [N, H]

    return torch.cos(ul), torch.sin(ul), torch.cos(vm), torch.sin(vm)


def predict_vis_from_precomp(
    img: torch.Tensor,
    Cul: torch.Tensor,
    Sul: torch.Tensor,
    Cvm: torch.Tensor,
    Svm: torch.Tensor,
) -> torch.Tensor:
    """
    Predict complex visibilities from the current image using precomputed trig tables.

    Implements the discrete 2-D DFT::

        V(u,v) = ∑_{l,m} I(l,m) exp(-2πi(ul + vm))

    Returns
    -------
    Tensor of shape [N, 2]  (real, imaginary columns).
    """
    I = img[0, 0]                        # [H, W]
    A = I @ Cul.t()                      # [H, N]
    B = I @ Sul.t()                      # [H, N]
    real =  torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))
    return torch.stack([real, imag], dim=1)  # [N, 2]


@torch.no_grad()
def make_dirty_and_psf_full(
    Cul: torch.Tensor,
    Sul: torch.Tensor,
    Cvm: torch.Tensor,
    Svm: torch.Tensor,
    re: torch.Tensor,
    im: torch.Tensor,
    w: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the dirty image and PSF from precomputed trig tables.

    Both outputs are normalised so that the PSF peak equals 1.

    Returns
    -------
    dirty : shape [H, W]
    psf   : shape [H, W]
    """
    E = (w * re)[:, None] * Cul - (w * im)[:, None] * Sul
    F = (w * re)[:, None] * Sul + (w * im)[:, None] * Cul
    dirty = Cvm.t() @ E - Svm.t() @ F
    psf   = Cvm.t() @ (w[:, None] * Cul) - Svm.t() @ (w[:, None] * Sul)
    if normalize:
        norm = torch.sum(psf).clamp_min(1e-12)
        dirty = dirty / norm
        psf   = psf   / norm
    return dirty, psf


def reconstruct_dip_small(
    uv: "np.ndarray | torch.Tensor",
    vis: Optional["np.ndarray | torch.Tensor"] = None,
    weight: Optional["np.ndarray | torch.Tensor"] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DIP reconstruction using the small (precomputed trig table) path.

    Parameters
    ----------
    uv     : Shape [N, 2] (u, v coordinates) or [N, 5] (u, v, Re, Im, w).
    vis    : Shape [N, 2] (Re, Im).  Required if *uv* has only 2 columns.
    weight : Shape [N].  Required if *uv* has only 2 columns.
    img_size : (H, W) in pixels.
    cfg    : Hyperparameter config.  Defaults to ``DIPConfig()``.
    device : ``"mps"``, ``"cuda"``, ``"cpu"``, or None (auto-detect).

    Returns
    -------
    image : reconstructed intensity map  [H, W]  float32 numpy array.
    dirty : dirty image                  [H, W]  float32 numpy array.
    psf   : point spread function        [H, W]  float32 numpy array.
    """
    if cfg is None:
        cfg = DIPConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(device)

    def to_t(x: Optional["np.ndarray | torch.Tensor"]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2:
        raise ValueError("uv must be 2-D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone()
        im_t = uv_t[:, 3].clone()
        w_t  = uv_t[:, 4].clone()
        uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None:
            raise ValueError("Provide vis & weight, or pass uv with 5 columns.")
        v_t  = to_t(vis)
        re_t = v_t[:, 0].clone()
        im_t = v_t[:, 1].clone()
        w_t  = to_t(weight).clone()

    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648_000.0)

    # Precompute trig tables once; keep on-device throughout training.
    Cul, Sul, Cvm, Svm = precompute_uv_phases_full(
        uv_t, H, W, cell_size_rad, dev, torch.float32
    )
    with torch.no_grad():
        dirty_t, psf_t = make_dirty_and_psf_full(
            Cul, Sul, Cvm, Svm, re_t, im_t, w_t, normalize=True
        )

    net = DIPUNet(
        input_depth=cfg.input_depth,
        out_ch=1,
        base_ch=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(
        nu=cfg.nu,
        learn_sigma=cfg.learn_sigma,
        init_sigma=cfg.init_sigma,
        reduction="mean",
    ).to(dev)
    params = list(net.parameters())
    if cfg.learn_sigma:
        params.append(student.log_sigma)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    targ_vis = torch.stack([re_t, im_t], dim=1)
    weights  = w_t.clamp(min=1e-12)

    best_img: Optional[torch.Tensor] = None
    best_data = float("inf")

    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))
        pred_vis  = predict_vis_from_precomp(img, Cul, Sul, Cvm, Svm)
        data_term = student(pred_vis, targ_vis, weight=weights)
        reg       = cfg.tv_weight * tv_loss(img)
        (data_term + reg).backward()
        optimizer.step()

        if (it + 1) % cfg.out_every == 0 or it == 0:
            msg = (
                f"[{it+1:5d}/{cfg.num_iters}]"
                f"  data={float(data_term.detach().cpu()):.6f}"
                f"  tv={float(reg.detach().cpu()):.6f}"
            )
            if cfg.learn_sigma:
                msg += f"  σ={float(torch.exp(student.log_sigma).detach().cpu()):.4g}"
            print(msg)

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item()
                best_img  = img.detach().clone()

    if best_img is None:
        best_img = img.detach()
    return (
        best_img[0, 0].cpu().numpy(),
        dirty_t.cpu().numpy(),
        psf_t.cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Memory path: on-the-fly visibility prediction
# ---------------------------------------------------------------------------

def predict_vis_chunk(
    img: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """
    Predict complex visibilities for a mini-batch of (u, v) points.

    Unlike ``predict_vis_from_precomp``, trig tables are computed on-the-fly
    for the given batch, avoiding O(N × H) memory.

    Returns
    -------
    Tensor of shape [batch, 2]  (real, imaginary).
    """
    I = img[0, 0]                                           # [H, W]
    ul = 2.0 * math.pi * (u[:, None] * l[None, :])
    vm = 2.0 * math.pi * (v[:, None] * m[None, :])
    Cul, Sul = torch.cos(ul), torch.sin(ul)
    Cvm, Svm = torch.cos(vm), torch.sin(vm)
    A = I @ Cul.t()
    B = I @ Sul.t()
    real =  torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))
    return torch.stack([real, imag], dim=1)


@torch.no_grad()
def make_dirty_and_psf_stream(
    uv: torch.Tensor,
    re: torch.Tensor,
    im: torch.Tensor,
    w: torch.Tensor,
    H: int,
    W: int,
    cell_size_rad: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dirty image and PSF by streaming through visibilities in chunks.

    Keeps peak GPU memory at O(chunk_size × max(H, W)) regardless of N.

    Returns
    -------
    dirty : shape [H, W]
    psf   : shape [H, W]
    """
    if chunk_size is None or chunk_size <= 0:
        chunk_size = 32_768
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)
    dirty = torch.zeros(H, W, device=device, dtype=dtype)
    psf   = torch.zeros(H, W, device=device, dtype=dtype)

    for start in range(0, uv.shape[0], chunk_size):
        end = min(start + chunk_size, uv.shape[0])
        u_c  = uv[start:end, 0].to(device=device, dtype=dtype)
        v_c  = uv[start:end, 1].to(device=device, dtype=dtype)
        re_c = re[start:end].to(device=device, dtype=dtype)
        im_c = im[start:end].to(device=device, dtype=dtype)
        w_c  = w[start:end].to(device=device, dtype=dtype)

        ul   = 2.0 * math.pi * (u_c[:, None] * l[None, :])
        vm   = 2.0 * math.pi * (v_c[:, None] * m[None, :])
        Cul, Sul = torch.cos(ul), torch.sin(ul)
        Cvm, Svm = torch.cos(vm), torch.sin(vm)

        E = (w_c * re_c)[:, None] * Cul - (w_c * im_c)[:, None] * Sul
        F = (w_c * re_c)[:, None] * Sul + (w_c * im_c)[:, None] * Cul
        dirty += Cvm.t() @ E - Svm.t() @ F
        psf   += Cvm.t() @ (w_c[:, None] * Cul) - Svm.t() @ (w_c[:, None] * Sul)

        del u_c, v_c, re_c, im_c, w_c, ul, vm, Cul, Sul, Cvm, Svm, E, F

    if normalize:
        norm  = torch.sum(psf).clamp_min(1e-12)
        dirty = dirty / norm
        psf   = psf   / norm
    return dirty, psf


# ---------------------------------------------------------------------------
# Memory path: stratified visibility sampler
# ---------------------------------------------------------------------------

class VisibilitySampler:
    """
    Mixture importance sampler over (u, v) space for large visibility sets.

    The sampling distribution is a convex combination of up to five components:

    1. **Uniform**       — draws any visibility with equal probability.
    2. **Radial**        — stratifies by baseline length ρ = √(u² + v²),
                           giving equal representation to each annular bin.
    3. **Inv-radius**    — weights ∝ 1/ρ^γ to oversample short baselines
                           (low-spatial-frequency information).
    4. **Weight-prop**   — weights ∝ visibility weight w_i.
    5. **Angular**       — stratifies by position angle (optional).

    The actual sample indices together with their mixture densities p_mix(i)
    are returned so the caller can apply SNIS correction.
    """

    def __init__(
        self,
        uv: torch.Tensor,
        w: torch.Tensor,
        cfg: DIPConfig,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.dev = device
        self.cpu = torch.device("cpu")
        self.N   = int(uv.shape[0])

        u_cpu = uv[:, 0].detach().to(self.cpu, dtype=torch.float32)
        v_cpu = uv[:, 1].detach().to(self.cpu, dtype=torch.float32)
        self.rho = torch.sqrt(u_cpu ** 2 + v_cpu ** 2)
        self.w   = w.detach().to(self.cpu, dtype=torch.float32).clamp_min(0)

        self._build_radial_bins()
        self._build_angular_bins(u_cpu, v_cpu)
        self.uni_prob = torch.tensor(1.0 / float(self.N), dtype=torch.float32, device=self.cpu)
        self._build_inv_radius_prob()
        self._build_weight_prob()
        self._build_alphas()
        self.bin_loss_ema = torch.zeros(self.B, dtype=torch.float32, device=self.cpu)
        self._make_bin_prob()

    # ------------------------------------------------------------------
    # Quantile helpers (three strategies to handle very large N)
    # ------------------------------------------------------------------

    def _quantile_edges_full(self, B: int) -> torch.Tensor:
        """Exact quantiles using torch.quantile (safe for N ≤ ~5 M)."""
        q = torch.linspace(0, 1, B + 1, dtype=torch.float32, device=self.cpu)
        return self._ensure_strictly_increasing(torch.quantile(self.rho, q))

    def _quantile_edges_sample(self, B: int, k: int) -> torch.Tensor:
        """Approximate quantiles from a random subsample of size k."""
        k = int(min(max(1000, k), self.N))
        idx    = torch.randint(0, self.N, (k,), device=self.cpu)
        sample = self.rho[idx]
        q      = torch.linspace(0, 1, B + 1, dtype=torch.float32, device=self.cpu)
        return self._ensure_strictly_increasing(torch.quantile(sample, q))

    def _quantile_edges_hist(self, B: int, H: int) -> torch.Tensor:
        """Histogram-CDF quantile approximation for extremely large N."""
        rmin = float(self.rho.min().item())
        rmax = float(self.rho.max().item())
        if rmax <= rmin:
            rmax = rmin + 1.0
        eps  = max(self.cfg.radius_eps_frac * rmax, 1e-12)
        hist = torch.histc(self.rho + eps, bins=int(H), min=rmin, max=rmax + eps)
        cdf  = torch.cumsum(hist, dim=0) / float(hist.sum().item())
        q    = torch.linspace(0, 1, B + 1, dtype=torch.float32)
        edges = torch.zeros(B + 1, dtype=torch.float32)
        j = 0
        for i in range(B + 1):
            while j < H - 1 and cdf[j] < q[i]:
                j += 1
            edges[i] = rmin + (rmax - rmin) * (j / max(1, H - 1))
        return self._ensure_strictly_increasing(edges)

    @staticmethod
    def _ensure_strictly_increasing(edges: torch.Tensor) -> torch.Tensor:
        """Guarantee monotone-strictly-increasing bin edges (required by bucketize)."""
        edges = edges.clone()
        span  = float(edges[-1].item() - edges[0].item() + 1.0)
        tiny  = 1e-9 * span
        for i in range(1, edges.numel()):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + tiny
        return edges

    def _build_radial_bins(self) -> None:
        B   = max(1, int(self.cfg.radial_bins))
        use_logspace = (
            not self.cfg.radial_quantiles
            or self.cfg.quantile_mode.lower() == "logspace"
        )
        if use_logspace:
            rmin = float(self.rho.min().item())
            rmax = float(self.rho.max().item())
            if rmax <= rmin:
                rmax = rmin + 1.0
            eps  = max(self.cfg.radius_eps_frac * rmax, 1e-12)
            lo   = math.log10(max(rmin + eps, 1e-12))
            hi   = math.log10(rmax + eps)
            edges = (
                torch.logspace(lo, hi, steps=B + 1, base=10.0, dtype=torch.float32)
                - eps
            )
            edges = self._ensure_strictly_increasing(edges)
        else:
            mode = self.cfg.quantile_mode.lower()
            N    = self.N
            if mode == "auto":
                if N <= self.cfg.quantile_full_threshold:
                    edges = self._quantile_edges_full(B)
                elif N <= max(self.cfg.quantile_sample_max, 1_000_000):
                    edges = self._quantile_edges_sample(B, self.cfg.quantile_sample_max)
                else:
                    edges = self._quantile_edges_hist(B, self.cfg.hist_bins_for_quantiles)
            elif mode == "full":
                edges = self._quantile_edges_full(B)
            elif mode == "sample":
                edges = self._quantile_edges_sample(B, self.cfg.quantile_sample_max)
            elif mode == "hist":
                edges = self._quantile_edges_hist(B, self.cfg.hist_bins_for_quantiles)
            else:
                rmin  = float(self.rho.min().item())
                rmax  = float(self.rho.max().item())
                if rmax <= rmin:
                    rmax = rmin + 1.0
                edges = torch.linspace(rmin, rmax, steps=B + 1, dtype=torch.float32)
                edges = self._ensure_strictly_increasing(edges)

        self.edges   = edges.to(self.cpu)
        bin_idx      = torch.bucketize(self.rho, self.edges, right=False) - 1
        self.bin_idx = torch.clamp(bin_idx, 0, B - 1).to(torch.long)
        self.B       = B

        self.bin_indices: List[torch.Tensor] = []
        self.bin_counts = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            idx_b = torch.where(self.bin_idx == b)[0]
            self.bin_indices.append(idx_b)
            self.bin_counts[b] = idx_b.numel()
        self.nonempty_bins = torch.tensor(
            [b for b in range(B) if self.bin_counts[b] > 0], dtype=torch.long
        )
        if self.nonempty_bins.numel() == 0:
            self.nonempty_bins = torch.tensor([0], dtype=torch.long)

    def _build_angular_bins(self, u_cpu: torch.Tensor, v_cpu: torch.Tensor) -> None:
        A = int(self.cfg.angular_bins)
        if A <= 0:
            self.A = 0
            self.theta_idx   = None
            self.ang_counts  = None
            self.ang_indices = None
            return
        theta = torch.atan2(v_cpu, u_cpu)
        theta = (theta + math.pi) / (2.0 * math.pi)  # map to [0, 1)
        edges = torch.linspace(0.0, 1.0, steps=A + 1)
        t_idx = torch.bucketize(theta, edges, right=False) - 1
        t_idx = torch.clamp(t_idx, 0, A - 1).to(torch.long)
        self.theta_idx = t_idx
        self.A = A
        self.ang_counts: torch.Tensor = torch.zeros(A, dtype=torch.long)
        self.ang_indices: List[torch.Tensor] = []
        for a in range(A):
            idx_a = torch.where(self.theta_idx == a)[0]
            self.ang_indices.append(idx_a)
            self.ang_counts[a] = idx_a.numel()

    def _build_inv_radius_prob(self) -> None:
        eps   = max(self.cfg.radius_eps_frac * float(self.rho.max().item()), 1e-12)
        gamma = float(self.cfg.inv_radius_gamma)
        x     = 1.0 / torch.pow(self.rho + eps, gamma)
        total = float(x.sum().item())
        if total <= 0:
            self.p_inv = torch.full((self.N,), 1.0 / self.N, dtype=torch.float32, device=self.cpu)
        else:
            self.p_inv = (x / total).to(torch.float32)

    def _build_weight_prob(self) -> None:
        total = float(self.w.sum().item())
        if total <= 0:
            self.p_w = torch.full((self.N,), 1.0 / self.N, dtype=torch.float32, device=self.cpu)
        else:
            self.p_w = (self.w / total).to(torch.float32)

    def _build_alphas(self) -> None:
        use_angular = self.cfg.angular_bins > 0
        alphas = torch.tensor(
            [
                max(0.0, float(self.cfg.alpha_uniform)),
                max(0.0, float(self.cfg.alpha_radial)),
                max(0.0, float(self.cfg.alpha_inv_radius)),
                max(0.0, float(self.cfg.alpha_weight)),
                max(0.0, float(self.cfg.alpha_angular)) if use_angular else 0.0,
            ],
            dtype=torch.float32,
            device=self.cpu,
        )
        s = float(alphas.sum().item())
        if s <= 0:
            alphas = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.cpu)
        else:
            alphas = alphas / s
        self.alphas = alphas

    def _make_bin_prob(self) -> None:
        """Assign per-bin sampling probability, with optional low-k boost and EMA adaption."""
        B    = self.B
        prob = torch.zeros(B, dtype=torch.float32, device=self.cpu)
        if self.nonempty_bins.numel() > 0:
            prob[self.nonempty_bins] = 1.0 / float(self.nonempty_bins.numel())

        # Boost innermost K bins to over-sample low-spatial-frequency signal.
        K = max(0, min(int(self.cfg.low_k_bins), B))
        if K > 0 and self.nonempty_bins.numel() > 0:
            boost    = max(1.0, float(self.cfg.low_k_boost))
            low_bins = torch.arange(0, K, dtype=torch.long)
            low_bins = low_bins[torch.isin(low_bins, self.nonempty_bins)]
            prob[low_bins] = prob[low_bins] * boost

        # Optionally replace uniform-per-bin with EMA loss (adaptive).
        if int(self.cfg.adapt_every) > 0:
            ema = torch.clamp(self.bin_loss_ema, min=0.0)
            if torch.any(ema > 0):
                ema = ema ** float(self.cfg.adapt_power)
                ema_masked = torch.zeros_like(prob)
                ema_masked[self.nonempty_bins] = ema[self.nonempty_bins]
                if float(ema_masked.sum().item()) > 0:
                    prob = ema_masked

        s = float(prob[self.nonempty_bins].sum().item())
        if s <= 0:
            prob[self.nonempty_bins] = 1.0 / float(self.nonempty_bins.numel())
        else:
            prob[self.nonempty_bins] = prob[self.nonempty_bins] / s
        self.bin_prob = prob

    def update_with_batch(self, idx: torch.Tensor, nll_vec: torch.Tensor) -> None:
        """Update per-bin EMA loss from the current mini-batch (for adaptive reweighting)."""
        idx_cpu  = idx.detach().to(self.cpu)
        nll_cpu  = nll_vec.detach().to(self.cpu)
        bin_ids  = self.bin_idx[idx_cpu]
        ema_new  = torch.zeros(self.B, dtype=torch.float32, device=self.cpu)
        cnt      = torch.zeros(self.B, dtype=torch.float32, device=self.cpu)
        ema_new.scatter_add_(0, bin_ids, nll_cpu)
        cnt.scatter_add_(0, bin_ids, torch.ones_like(nll_cpu))
        mask = cnt > 0
        ema_new[mask] = ema_new[mask] / cnt[mask]
        alpha = float(self.cfg.adapt_ema)
        self.bin_loss_ema = (
            alpha * self.bin_loss_ema + (1.0 - alpha) * ema_new
        )
        self._make_bin_prob()

    @staticmethod
    def _multinomial_large(
        p: torch.Tensor,
        num_samples: int,
        replacement: bool = True,
        max_cats_safe: int = (1 << 24) - 4096,
        block_size: int = 8_000_000,
    ) -> torch.Tensor:
        """
        torch.multinomial wrapper that works even when len(p) exceeds PyTorch's
        internal 2^24 category limit by using a two-level block-hierarchical scheme.
        """
        N = int(p.numel())
        if N <= max_cats_safe:
            if float(p.sum().item()) <= 0:
                return torch.randint(0, N, (num_samples,), device=p.device)
            return torch.multinomial(p, num_samples, replacement=replacement)

        # Hierarchical: sample a block, then sample within the block.
        n_blocks     = max(1, (N + block_size - 1) // block_size)
        block_weights = torch.tensor(
            [p[b * block_size : min((b + 1) * block_size, N)].sum().item()
             for b in range(n_blocks)],
            dtype=torch.float32,
            device=p.device,
        )
        total_bw = float(block_weights.sum().item())
        if total_bw <= 0:
            return torch.randint(0, N, (num_samples,), device=p.device)

        block_ids = torch.multinomial(block_weights / total_bw, num_samples, replacement=True)
        counts    = torch.bincount(block_ids, minlength=n_blocks)
        out_parts: List[torch.Tensor] = []
        for b in range(n_blocks):
            k = int(counts[b].item())
            if k == 0:
                continue
            s  = b * block_size
            e  = min(s + block_size, N)
            pb = p[s:e]
            ps = float(pb.sum().item())
            if ps <= 0:
                idx_local = torch.randint(0, e - s, (k,), device=p.device)
            else:
                idx_local = torch.multinomial(pb / ps, k, replacement=True)
            out_parts.append(s + idx_local)

        out  = torch.cat(out_parts, dim=0)
        perm = torch.randperm(out.numel(), device=out.device)
        return out[perm]

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw *batch_size* visibility indices and return their mixture densities.

        Returns
        -------
        idx   : shape [batch_size]  — indices into the full visibility array.
        p_mix : shape [batch_size]  — mixture density at each drawn index.
        """
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be > 0")

        # Assign each sample to one mixture component.
        comp_idx = torch.multinomial(self.alphas, bs, replacement=True)
        counts   = torch.bincount(comp_idx, minlength=5)
        nU, nR, nI, nW, nA = [int(counts[i].item()) for i in range(5)]
        idx_list: List[torch.Tensor] = []

        if nU > 0:
            idx_list.append(torch.randint(0, self.N, (nU,), device=self.cpu))

        if nR > 0:
            prob_ne = self.bin_prob[self.nonempty_bins]
            prob_ne = prob_ne / float(prob_ne.sum().item())
            bsel    = torch.multinomial(prob_ne, nR, replacement=True)
            bins    = self.nonempty_bins[bsel]
            uniq_bins, per_counts = bins.unique(return_counts=True)
            ir_parts: List[torch.Tensor] = []
            for b, k in zip(uniq_bins.tolist(), per_counts.tolist()):
                nb_count = int(self.bin_counts[int(b)].item())
                if nb_count <= 0:
                    ir_parts.append(torch.randint(0, self.N, (k,), device=self.cpu))
                else:
                    pos = torch.randint(0, nb_count, (k,), device=self.cpu)
                    ir_parts.append(self.bin_indices[int(b)][pos])
            idx_list.append(torch.cat(ir_parts).view(-1))

        if nI > 0:
            idx_list.append(self._multinomial_large(self.p_inv, nI, replacement=True))

        if nW > 0:
            if float(self.p_w.sum().item()) <= 0:
                idx_list.append(torch.randint(0, self.N, (nW,), device=self.cpu))
            else:
                idx_list.append(self._multinomial_large(self.p_w, nW, replacement=True))

        if nA > 0 and self.ang_indices is not None:
            nonempty_ang = torch.tensor(
                [a for a in range(self.A) if self.ang_counts[a] > 0], dtype=torch.long
            )
            if nonempty_ang.numel() == 0:
                idx_list.append(torch.randint(0, self.N, (nA,), device=self.cpu))
            else:
                pang = torch.full(
                    (nonempty_ang.numel(),), 1.0 / float(nonempty_ang.numel()),
                    dtype=torch.float32,
                )
                asel    = torch.multinomial(pang, nA, replacement=True)
                bins_a  = nonempty_ang[asel]
                ia_parts: List[torch.Tensor] = []
                for a in bins_a.tolist():
                    arr = self.ang_indices[int(a)]
                    pos = torch.randint(0, int(arr.numel()), (1,), device=self.cpu)
                    ia_parts.append(arr[pos])
                idx_list.append(torch.cat(ia_parts).view(-1))

        idx_cpu = torch.cat(idx_list).to(torch.long)

        # Compute per-sample mixture density for SNIS correction.
        b_i   = self.bin_idx[idx_cpu]
        n_b   = self.bin_counts[b_i].to(torch.float32)
        p_rad = torch.where(
            n_b > 0,
            self.bin_prob[b_i] * (1.0 / n_b),
            torch.zeros_like(n_b),
        )
        p_inv = self.p_inv[idx_cpu]
        p_w   = self.p_w[idx_cpu]

        αU, αR, αI, αW, αA = [self.alphas[i].item() for i in range(5)]
        p_mix = (
            αU * float(self.uni_prob.item())
            + αR * p_rad
            + αI * p_inv
            + αW * p_w
        ).clamp_min(1e-18)

        perm    = torch.randperm(idx_cpu.numel())
        idx_cpu = idx_cpu[perm]
        p_mix   = p_mix[perm]
        return idx_cpu.to(self.dev), p_mix.to(self.dev)


def reconstruct_dip_memory(
    uv: "np.ndarray | torch.Tensor",
    vis: Optional["np.ndarray | torch.Tensor"] = None,
    weight: Optional["np.ndarray | torch.Tensor"] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DIP reconstruction using the memory-efficient (mini-batch + SNIS) path.

    Parameters
    ----------
    uv, vis, weight, img_size, cfg, device
        Same as ``reconstruct_dip_small``.

    Returns
    -------
    image, dirty, psf — same as ``reconstruct_dip_small``.
    """
    if cfg is None:
        cfg = DIPConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(device)

    def to_t(x: Optional["np.ndarray | torch.Tensor"]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2:
        raise ValueError("uv must be 2-D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone()
        im_t = uv_t[:, 3].clone()
        w_t  = uv_t[:, 4].clone()
        uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None:
            raise ValueError("Provide vis & weight, or pass uv with 5 columns.")
        v_t  = to_t(vis)
        re_t = v_t[:, 0].clone()
        im_t = v_t[:, 1].clone()
        w_t  = to_t(weight).clone()

    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648_000.0)
    l, m = _grid_coords(H, W, cell_size_rad, dev, torch.float32)

    stream_chunk = cfg.stream_chunk_vis if (cfg.stream_chunk_vis and cfg.stream_chunk_vis > 0) else cfg.batch_vis
    with torch.no_grad():
        dirty_t, psf_t = make_dirty_and_psf_stream(
            uv_t, re_t, im_t, w_t, H, W, cell_size_rad, dev,
            torch.float32, True, stream_chunk,
        )

    net = DIPUNet(
        input_depth=cfg.input_depth,
        out_ch=1,
        base_ch=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(
        nu=cfg.nu,
        learn_sigma=cfg.learn_sigma,
        init_sigma=cfg.init_sigma,
        reduction="none",
    ).to(dev)
    params = list(net.parameters())
    if cfg.learn_sigma:
        params.append(student.log_sigma)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    sampler  = VisibilitySampler(uv_t, w_t, cfg, dev)
    N        = uv_t.shape[0]
    bs       = max(1, min(cfg.batch_vis, N))
    u_unif   = 1.0 / float(N)   # density of the uniform distribution over N points

    best_img: Optional[torch.Tensor] = None
    best_data = float("inf")

    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))

        idx, p_mix = sampler.sample(bs)
        p_mix = p_mix.clamp(min=1e-12)
        u_b   = uv_t[idx, 0];  v_b   = uv_t[idx, 1]
        re_b  = re_t[idx];     im_b  = im_t[idx];   w_b = w_t[idx]

        pred_b  = predict_vis_chunk(img, u_b, v_b, l, m)
        targ_b  = torch.stack([re_b, im_b], dim=1)
        nll_vec = student(pred_b, targ_b, weight=w_b)

        imp = u_unif / p_mix
        if cfg.importance_snis:
            data_term = torch.sum(imp * nll_vec) / (torch.sum(imp) + 1e-12)
        else:
            data_term = torch.mean(nll_vec * imp)

        reg  = cfg.tv_weight * tv_loss(img)
        (data_term + reg).backward()
        optimizer.step()

        if cfg.adapt_every > 0 and (it + 1) % cfg.adapt_every == 0:
            sampler.update_with_batch(idx, nll_vec)

        if (it + 1) % cfg.out_every == 0 or it == 0:
            msg = (
                f"[{it+1:5d}/{cfg.num_iters}]"
                f"  data={float(data_term.detach().cpu()):.6f}"
                f"  tv={float(reg.detach().cpu()):.6f}"
            )
            if cfg.learn_sigma:
                msg += f"  σ={float(torch.exp(student.log_sigma).detach().cpu()):.4g}"
            print(msg)

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item()
                best_img  = img.detach().clone()

    if best_img is None:
        best_img = img.detach()
    return (
        best_img[0, 0].cpu().numpy(),
        dirty_t.cpu().numpy(),
        psf_t.cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Public entry point: auto-selects small vs memory path
# ---------------------------------------------------------------------------

def reconstruct_dip(
    uv: "np.ndarray | torch.Tensor",
    vis: Optional["np.ndarray | torch.Tensor"] = None,
    weight: Optional["np.ndarray | torch.Tensor"] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DIP reconstruction, automatically choosing the small or memory path.

    The path is selected by comparing N (number of visibilities) against
    ``cfg.auto_threshold_n`` when ``cfg.force_mode == "auto"``.

    Parameters
    ----------
    uv     : Shape [N, 2] (u, v) or [N, 5] (u, v, Re, Im, weight).
    vis    : Shape [N, 2].  Required when *uv* has 2 columns.
    weight : Shape [N].     Required when *uv* has 2 columns.
    img_size : (H, W) in pixels.
    cfg    : ``DIPConfig`` instance.  Falls back to defaults when None.
    device : ``"mps"``, ``"cuda"``, ``"cpu"``, or None (auto).

    Returns
    -------
    image : float32 ndarray [H, W] — DIP reconstructed intensity.
    dirty : float32 ndarray [H, W] — dirty (back-projected) image.
    psf   : float32 ndarray [H, W] — point spread function.
    """
    if cfg is None:
        cfg = DIPConfig()
    uv_arr = uv.numpy() if torch.is_tensor(uv) else np.asarray(uv)
    if uv_arr.ndim != 2:
        raise ValueError("uv must have shape [N, 2] or [N, 5]")

    N    = uv_arr.shape[0]
    mode = cfg.force_mode.lower()
    if mode not in ("auto", "small", "memory"):
        raise ValueError("cfg.force_mode must be 'auto' | 'small' | 'memory'")

    use_small = mode == "small" or (mode == "auto" and N <= int(cfg.auto_threshold_n))
    if use_small:
        print(f"[selector] SMALL path  (N={N:,} ≤ threshold {cfg.auto_threshold_n:,})")
        return reconstruct_dip_small(uv, vis=vis, weight=weight, img_size=img_size, cfg=cfg, device=device)
    else:
        print(f"[selector] MEMORY path (N={N:,} > threshold {cfg.auto_threshold_n:,})")
        return reconstruct_dip_memory(uv, vis=vis, weight=weight, img_size=img_size, cfg=cfg, device=device)


# ---------------------------------------------------------------------------
# Bootstrap uncertainty estimation
# ---------------------------------------------------------------------------

def _hashed_uniform_01(idx_cpu: np.ndarray, replicate: int, seed: int = 0) -> np.ndarray:
    """
    Stateless pseudo-random uniform(0, 1) mapping via SplitMix64 mixing.

    Avoids storing an O(N × B) weight matrix by computing bootstrap weights
    deterministically from the visibility index and replicate number alone.

    Parameters
    ----------
    idx_cpu   : uint64 array of visibility indices.
    replicate : bootstrap replicate index (0-based).
    seed      : global RNG seed for reproducibility.

    Returns
    -------
    float64 array in [0, 1).
    """
    x = (
        idx_cpu.astype(np.uint64)
        ^ np.uint64(seed)
        ^ (np.uint64(replicate) * np.uint64(0x9E3779B97F4A7C15))
    )
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(30)
    x  = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(27)
    x  = (x * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(31)
    return ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))


def _bootstrap_weights_for_indices(
    idx: torch.Tensor,
    method: str,
    replicate: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return bootstrap multipliers s_i for each visibility index in *idx*.

    Parameters
    ----------
    idx       : Visibility indices (any device).
    method    : ``"poisson"`` (integer counts ∼ Poisson(1)) or
                ``"bayesian"`` (continuous weights ∼ Exp(1)).
    replicate : Bootstrap replicate (determines the hash offset).
    seed      : Global seed.
    device    : Target device for the returned tensor.

    Returns
    -------
    float32 tensor of shape [len(idx)].
    """
    idx_np = idx.detach().cpu().numpy().astype(np.uint64)
    u      = _hashed_uniform_01(idx_np, replicate, seed)

    if method == "poisson":
        # Invert the Poisson(1) CDF using tabulated thresholds (k = 0..8).
        cdf = np.array(
            [0.36787944117144233, 0.73575888234288470, 0.91969860292860590,
             0.98101184312384630, 0.99634015317265630, 0.99940581518241830,
             0.99991675885071100, 0.99998975080332560, 0.99999887426905190],
            dtype=np.float64,
        )
        s = np.zeros_like(u, dtype=np.float32)
        for threshold in cdf:
            s += (u >= threshold).astype(np.float32)
    elif method == "bayesian":
        s = -np.log(np.clip(u, 1e-16, 1.0)).astype(np.float32)
    else:
        raise ValueError("bootstrap method must be 'poisson' or 'bayesian'")

    return torch.as_tensor(s, device=device, dtype=torch.float32)


def _train_small_bootstrap_once(
    uv_t: torch.Tensor,
    re_t: torch.Tensor,
    im_t: torch.Tensor,
    w_t: torch.Tensor,
    img_size: Tuple[int, int],
    cfg: DIPConfig,
    device: torch.device,
    replicate: int,
    boot_method: str,
    boot_seed: int,
) -> np.ndarray:
    """
    One bootstrap replicate using the small path.

    Weights the per-visibility NLL by deterministic Poisson/Bayesian
    bootstrap multipliers before summing.
    """
    torch.manual_seed(cfg.seed + replicate)
    np.random.seed(cfg.seed + replicate)
    dev = pick_device(str(device))
    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648_000.0)

    Cul, Sul, Cvm, Svm = precompute_uv_phases_full(
        uv_t, H, W, cell_size_rad, dev, torch.float32
    )
    net = DIPUNet(
        input_depth=cfg.input_depth, out_ch=1,
        base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout,
    ).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(
        nu=cfg.nu, learn_sigma=cfg.learn_sigma,
        init_sigma=cfg.init_sigma, reduction="none",
    ).to(dev)
    params = list(net.parameters())
    if cfg.learn_sigma:
        params.append(student.log_sigma)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    N       = uv_t.shape[0]
    all_idx = torch.arange(N, device=dev, dtype=torch.long)

    best_img: Optional[torch.Tensor] = None
    best_data = float("inf")

    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img      = pos(net(z))
        pred_vis = predict_vis_from_precomp(img, Cul, Sul, Cvm, Svm)
        targ_vis = torch.stack([re_t, im_t], dim=1)
        nll_vec  = student(pred_vis, targ_vis, weight=w_t)

        # Deterministic bootstrap weights (no extra RNG state needed).
        s = _bootstrap_weights_for_indices(
            all_idx, boot_method, replicate, cfg.seed + boot_seed, dev
        )
        data_term = torch.sum(s * nll_vec) / (torch.sum(s) + 1e-12)
        reg       = cfg.tv_weight * tv_loss(img)
        (data_term + reg).backward()
        optimizer.step()

        if (it + 1) % cfg.out_every == 0 or it == 0:
            print(
                f"[boot {replicate:03d}] [{it+1:5d}/{cfg.num_iters}]"
                f"  data={float(data_term.detach().cpu()):.6f}"
                f"  tv={float(reg.detach().cpu()):.6f}"
            )

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item()
                best_img  = img.detach().clone()

    if best_img is None:
        best_img = img.detach()
    return best_img[0, 0].cpu().numpy()


def _train_memory_bootstrap_once(
    uv_t: torch.Tensor,
    re_t: torch.Tensor,
    im_t: torch.Tensor,
    w_t: torch.Tensor,
    img_size: Tuple[int, int],
    cfg: DIPConfig,
    device: torch.device,
    replicate: int,
    boot_method: str,
    boot_seed: int,
) -> np.ndarray:
    """
    One bootstrap replicate using the memory path.

    Combines SNIS importance weights with bootstrap multipliers s_i so
    the effective gradient estimate is::

        ∇ loss ≈ ∑_i (u_unif / p_mix_i) · s_i · ∇ nll_i
                 ─────────────────────────────────────────
                 ∑_i (u_unif / p_mix_i) · s_i
    """
    torch.manual_seed(cfg.seed + replicate)
    np.random.seed(cfg.seed + replicate)
    dev = pick_device(str(device))
    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648_000.0)
    l, m = _grid_coords(H, W, cell_size_rad, dev, torch.float32)

    net = DIPUNet(
        input_depth=cfg.input_depth, out_ch=1,
        base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout,
    ).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(
        nu=cfg.nu, learn_sigma=cfg.learn_sigma,
        init_sigma=cfg.init_sigma, reduction="none",
    ).to(dev)
    params = list(net.parameters())
    if cfg.learn_sigma:
        params.append(student.log_sigma)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    sampler = VisibilitySampler(uv_t, w_t, cfg, dev)
    N       = uv_t.shape[0]
    bs      = max(1, min(cfg.batch_vis, N))
    u_unif  = 1.0 / float(N)

    best_img: Optional[torch.Tensor] = None
    best_data = float("inf")

    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))

        idx, p_mix = sampler.sample(bs)
        p_mix = p_mix.clamp(min=1e-12)
        u_b  = uv_t[idx, 0];  v_b  = uv_t[idx, 1]
        re_b = re_t[idx];     im_b = im_t[idx];   w_b = w_t[idx]

        pred_b  = predict_vis_chunk(img, u_b, v_b, l, m)
        targ_b  = torch.stack([re_b, im_b], dim=1)
        nll_vec = student(pred_b, targ_b, weight=w_b)

        s_i = _bootstrap_weights_for_indices(
            idx, boot_method, replicate, cfg.seed + boot_seed, dev
        )
        imp = u_unif / p_mix
        if cfg.importance_snis:
            data_term = (
                torch.sum(imp * s_i * nll_vec)
                / (torch.sum(imp * s_i) + 1e-12)
            )
        else:
            data_term = torch.mean(nll_vec * imp * s_i)

        reg = cfg.tv_weight * tv_loss(img)
        (data_term + reg).backward()
        optimizer.step()

        if cfg.adapt_every > 0 and (it + 1) % cfg.adapt_every == 0:
            sampler.update_with_batch(idx, nll_vec)

        if (it + 1) % cfg.out_every == 0 or it == 0:
            print(
                f"[boot {replicate:03d}] [{it+1:5d}/{cfg.num_iters}]"
                f"  data={float(data_term.detach().cpu()):.6f}"
                f"  tv={float(reg.detach().cpu()):.6f}"
            )

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item()
                best_img  = img.detach().clone()

    if best_img is None:
        best_img = img.detach()
    return best_img[0, 0].cpu().numpy()


def bootstrap_reconstruct(
    uv: "np.ndarray | torch.Tensor",
    vis: Optional["np.ndarray | torch.Tensor"],
    weight: Optional["np.ndarray | torch.Tensor"],
    img_size: Tuple[int, int],
    cfg: DIPConfig,
    device: Optional[str] = None,
    *,
    B: int = 20,
    method: str = "poisson",
    percentiles: Tuple[float, float] = (16.0, 84.0),
    return_all: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run B bootstrap replications and return per-pixel uncertainty maps.

    Each replicate re-runs DIP with reweighted visibility data:

    - ``method="poisson"``  — integer counts ∼ Poisson(1) per visibility.
    - ``method="bayesian"`` — continuous weights ∼ Exp(1) per visibility.

    Weights are computed deterministically from a hash of (index, replicate),
    so no O(N × B) storage is needed.

    Parameters
    ----------
    uv, vis, weight, img_size, cfg, device
        Same as ``reconstruct_dip``.
    B           : Number of bootstrap replicates.
    method      : ``"poisson"`` or ``"bayesian"``.
    percentiles : (lo, hi) percentile pair, e.g. (16, 84) for ±1σ equivalent.
    return_all  : If True, include the full [B, H, W] sample stack in the output.

    Returns
    -------
    dict with keys:
        ``mean``   — pixel-wise bootstrap mean           [H, W]
        ``std``    — pixel-wise bootstrap std            [H, W]
        ``p_lo``   — pixel-wise lower percentile map     [H, W]
        ``p_hi``   — pixel-wise upper percentile map     [H, W]
        ``samples``— full replicate stack (if return_all)[B, H, W]
    """
    if cfg is None:
        cfg = DIPConfig()
    dev = pick_device(device)

    def to_t(x: Optional["np.ndarray | torch.Tensor"]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2:
        raise ValueError("uv must be 2-D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone();  im_t = uv_t[:, 3].clone()
        w_t  = uv_t[:, 4].clone();  uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None:
            raise ValueError("Provide vis & weight, or pass uv with 5 columns.")
        v_t  = to_t(vis)
        re_t = v_t[:, 0].clone();   im_t = v_t[:, 1].clone()
        w_t  = to_t(weight).clone()

    N = int(uv_t.shape[0])
    mode      = cfg.force_mode.lower()
    use_small = mode == "small" or (mode == "auto" and N <= int(cfg.auto_threshold_n))

    H, W  = int(img_size[0]), int(img_size[1])
    imgs  = np.empty((B, H, W), dtype=np.float32)
    boot_seed = 12345   # separate stream from the training seed

    for b in range(B):
        print(f"\n=== Bootstrap replicate {b + 1}/{B} ===")
        if use_small:
            imgs[b] = _train_small_bootstrap_once(
                uv_t, re_t, im_t, w_t, img_size, cfg, dev, b, method, boot_seed
            )
        else:
            imgs[b] = _train_memory_bootstrap_once(
                uv_t, re_t, im_t, w_t, img_size, cfg, dev, b, method, boot_seed
            )

    mean = imgs.mean(axis=0)
    std  = imgs.std(axis=0, ddof=1) if B > 1 else np.zeros_like(mean)
    p_lo = np.percentile(imgs, percentiles[0], axis=0)
    p_hi = np.percentile(imgs, percentiles[1], axis=0)

    out: Dict[str, np.ndarray] = {
        "mean": mean,
        "std":  std,
        "p_lo": p_lo,
        "p_hi": p_hi,
    }
    if return_all:
        out["samples"] = imgs
    return out
