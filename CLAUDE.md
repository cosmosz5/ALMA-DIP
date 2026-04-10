# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the reconstruction

```bash
python run_alma_dip_bootstrapping.py
```

This loads `DG_Tau.txt` (ALMA visibility data: u, v, Re, Im, weight columns), runs DIP reconstruction, and saves FITS output files.

**Dependencies:** `numpy`, `torch`, `astropy`

**Device selection:** The script calls `pick_device("mps")` — change to `"cuda"` or `"cpu"` as needed. `pick_device()` with no argument auto-detects MPS → CUDA → CPU.

## Architecture

The codebase implements **Deep Image Prior (DIP) for ALMA radio interferometric imaging** with bootstrap uncertainty estimation. There are two files:

- `alma_dip_bootstrapping.py` — core library (all classes/functions)
- `run_alma_dip_bootstrapping.py` — entry point that configures and runs reconstruction

### Two computation paths (auto-selected by `DIPConfig.force_mode`)

**Small path** (`reconstruct_dip_small`): Precomputes full trig tables (`Cul`, `Sul`, `Cvm`, `Svm`) for all N visibilities. Memory cost is O(N × image_size). Used when N ≤ `auto_threshold_n` (default 5M).

**Memory path** (`reconstruct_dip_memory`): Uses `VisibilitySampler` to draw mini-batches from a mixture distribution (uniform + radial stratified + inverse-radius + weight-proportional + angular). Applies Self-Normalized Importance Sampling (SNIS) correction so the stochastic gradient is unbiased. Used when N > threshold.

**Auto selector:** `reconstruct_dip()` dispatches to the appropriate path based on N vs. `cfg.auto_threshold_n`.

### Key components

- **`DIPUNet`**: U-Net generator with reflection padding, GroupNorm, bilinear upsampling, and skip connections. Input is a fixed random tensor `z`; output is the reconstructed image.
- **`StudentTLoss`**: Student's-t negative log-likelihood used as the data fidelity term (more robust to outliers than Gaussian NLL). Controlled by `cfg.nu`.
- **`VisibilitySampler`**: Mixture sampler over (u,v) space with multiple components. Handles large-N quantile computation via histogram approximation to avoid memory blowout.
- **`bootstrap_reconstruct`**: Runs B independent DIP reconstructions with perturbed visibility weights (Poisson or Bayesian bootstrap). Bootstrap weights are computed deterministically via a hash function (`_hashed_uniform_01`) rather than storing O(N×B) random arrays.

### `DIPConfig` key parameters

| Parameter | Effect |
|-----------|--------|
| `force_mode` | `"auto"` / `"small"` / `"memory"` — selects computation path |
| `auto_threshold_n` | N threshold for auto path selection |
| `num_iters` / `lr` | Optimizer iterations and learning rate |
| `tv_weight` | Total variation regularization strength |
| `cell_size_arcsec` | Image pixel scale (arcsec/pixel) |
| `nu` | Student-t degrees of freedom (robustness) |
| `batch_vis` | Mini-batch size for memory path |
| `radial_bins`, `alpha_*` | Stratified sampler mixture weights |

### Data format

`DG_Tau.txt`: whitespace-delimited, one header row, columns: `u v Re Im weight` (all float32 after loading).

### Output FITS files

- `DG_Tau_alma_dip_ref_bootstrapping.fits` — DIP reconstructed image
- `DG_Tau_alma_dip_ref_bootstrapping_dirty.fits` — dirty map
- `DG_Tau_alma_dip_ref_bootstrapping_beam.fits` — PSF/beam
- `DG_Tau_bootstrap_{mean,std,p16,p84}.fits` — bootstrap uncertainty maps

> **Note:** `run_alma_dip_bootstrapping.py` currently has a `pdb.set_trace()` breakpoint between the reference reconstruction and bootstrap loop. Remove it before running non-interactively.
