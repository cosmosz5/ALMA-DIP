"""
run_alma_dip_bootstrapping.py
==============================
Entry point for DIP reconstruction and bootstrap uncertainty estimation
of the DG Tau ALMA visibility dataset.

Usage
-----
    python run_alma_dip_bootstrapping.py

Output FITS files (written to the current directory)
-----------------------------------------------------
    DG_Tau_dip_image.fits          — DIP reconstructed intensity map
    DG_Tau_dip_dirty.fits          — Dirty (back-projected) image
    DG_Tau_dip_beam.fits           — Point spread function (PSF / beam)
    DG_Tau_bootstrap_mean.fits     — Bootstrap pixel-mean
    DG_Tau_bootstrap_std.fits      — Bootstrap pixel-std  (1-sigma uncertainty)
    DG_Tau_bootstrap_p16.fits      — 16th-percentile map  (lower ±1σ equivalent)
    DG_Tau_bootstrap_p84.fits      — 84th-percentile map  (upper ±1σ equivalent)

Tuning guide
------------
* Fewer bootstrap replicates (B) → faster run, noisier uncertainty maps.
* Reduce num_iters (and proportionally out_every) for quick test runs.
* Set force_mode="small"  to pin the small path (OK when RAM/VRAM allows).
* Set force_mode="memory" to pin the memory path (safer for large datasets).
* Increase batch_vis for more accurate SNIS gradients at the cost of
  higher per-iteration memory on the memory path.
"""

import numpy as np
import astropy.io.fits as fits

from alma_dip_bootstrapping import DIPConfig, bootstrap_reconstruct, reconstruct_dip, pick_device

# ---------------------------------------------------------------------------
# 1.  Load visibility data
# ---------------------------------------------------------------------------
# Expected columns: u  v  Re  Im  weight
# Row 0 is a header line (skiprows=1).
data = np.loadtxt("DG_Tau.txt", skiprows=1).astype(np.float32)
uv     = data[:, :2]   # baseline coordinates  [N, 2]
vis    = data[:, 2:4]  # complex visibilities   [N, 2]  (Re, Im)
weight = data[:, 4]    # natural weights        [N]

print(f"Loaded {len(uv):,} visibilities from DG_Tau.txt")

# ---------------------------------------------------------------------------
# 2.  Configuration
# ---------------------------------------------------------------------------
cfg = DIPConfig(
    # ---- Optimisation ----
    num_iters=1800,       # gradient steps per reconstruction
    lr=1e-3,

    # ---- Regularisation ----
    tv_weight=1e-2,       # total-variation strength (higher → smoother image)

    # ---- Image geometry ----
    cell_size_arcsec=0.00075 * 4,   # pixel scale in arcseconds

    # ---- Logging ----
    out_every=200,        # print a progress line every N iterations

    # ---- Likelihood ----
    nu=3.0,               # Student-t degrees of freedom (robustness to outliers)
    learn_sigma=False,    # fix the scale parameter σ

    # ---- Path selection ----
    force_mode="auto",          # "auto" | "small" | "memory"
    auto_threshold_n=5_000_000, # use memory path if N > this value

    # ---- Memory-path sampler ----
    batch_vis=16384,
    stream_chunk_vis=32768,
    radial_bins=48,
    radial_quantiles=True,
    low_k_bins=6,
    low_k_boost=6.0,
    alpha_uniform=0.05,
    alpha_radial=0.55,
    alpha_inv_radius=0.40,
    quantile_mode="auto",
)

# ---------------------------------------------------------------------------
# 3.  Device selection
# ---------------------------------------------------------------------------
device = str(pick_device("mps"))   # change to "cuda" or "cpu" if needed
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 4.  Reference DIP reconstruction (single run, best-loss image)
# ---------------------------------------------------------------------------
print("\n=== Reference reconstruction ===")
image, dirty, beam = reconstruct_dip(
    uv=uv,
    vis=vis,
    weight=weight,
    img_size=(540, 540),
    cfg=cfg,
    device=device,
)

fits.writeto("DG_Tau_dip_image.fits", image, overwrite=True)
fits.writeto("DG_Tau_dip_dirty.fits", dirty, overwrite=True)
fits.writeto("DG_Tau_dip_beam.fits",  beam,  overwrite=True)
print("Saved: DG_Tau_dip_image.fits  DG_Tau_dip_dirty.fits  DG_Tau_dip_beam.fits")

# ---------------------------------------------------------------------------
# 5.  Bootstrap uncertainty estimation
# ---------------------------------------------------------------------------
print("\n=== Bootstrap uncertainty estimation ===")
boot = bootstrap_reconstruct(
    uv=uv,
    vis=vis,
    weight=weight,
    img_size=(540, 540),
    cfg=cfg,
    device=device,
    B=20,                      # number of bootstrap replicates
    method="poisson",          # "poisson" (counts) or "bayesian" (Exp(1) weights)
    percentiles=(16, 84),      # maps to ±1σ equivalent for a Gaussian distribution
    return_all=False,          # set True to also save the full [B,H,W] stack
)

fits.writeto("DG_Tau_bootstrap_mean.fits", boot["mean"], overwrite=True)
fits.writeto("DG_Tau_bootstrap_std.fits",  boot["std"],  overwrite=True)
fits.writeto("DG_Tau_bootstrap_p16.fits",  boot["p_lo"], overwrite=True)
fits.writeto("DG_Tau_bootstrap_p84.fits",  boot["p_hi"], overwrite=True)
print(
    "Saved: DG_Tau_bootstrap_mean.fits  DG_Tau_bootstrap_std.fits"
    "  DG_Tau_bootstrap_p16.fits  DG_Tau_bootstrap_p84.fits"
)
