# Project log

## 2026-11-09 - Block 1 (Farida): Colab setup + baseline - milestone M1 PASSED

**Setup (free Colab, Tesla T4 15 GB)**
- InstantSplat commit `b951567`, MASt3R `ViTLarge_BaseDecoder_512_catmlpdpt_metric`; Python 3.13, PyTorch 2.14 + CUDA 13.0.
- Fixes needed on current Colab: `torch.load(..., weights_only=False)` for the MASt3R file; `#include <cstdint>` in the rasterizer; `open3d` must be installed; `np.linalg.linalg.*` -> `np.linalg.*` (NumPy 2).
- Notebook: `notebooks/01_block1_setup_baseline.ipynb` (Run all; finished runs are skipped).

**Protocol (frozen)**
- Data: Mip-NeRF 360, 8 scenes (flowers excluded), image size `images_8`, 1000 training iterations.
- Split (Free360-style): test = every 8th photo (idx % 8 == 0); train = 3/6/9 photos spaced evenly among the rest. Exact lists: `views_manifest.json` on Drive.
- Test-pose start: InstantSplat's default interpolated start was 78-90 deg (max ~177 deg) and 134-150% of the scene radius away from the reference on garden, while training poses were within ~1 deg. We patched `init_geo.py`: each test pose starts from its reference pose mapped into the estimated frame with one Sim(3) (Umeyama) fitted on the training camera centres. Then 500 test-pose optimization steps (InstantSplat default). **Same rule for every configuration.**
- Image metrics: InstantSplat's `metrics.py` (PSNR / SSIM / LPIPS) on the test photos.

**Baseline C3 (full InstantSplat), garden + bonsai**

| Scene | Views | PSNR | SSIM | LPIPS | Time (min) | Peak GPU (MB) |
|---|---|---|---|---|---|---|
| garden | 3 | 14.65 | 0.273 | 0.555 | 8.9 | 3241 |
| garden | 6 | 17.58 | 0.345 | 0.396 | 12.4 | 3653 |
| garden | 9 | 18.04 | 0.344 | 0.377 | 16.8 | 4827 |
| bonsai | 3 | 14.75 | 0.358 | 0.581 | 9.9 | 3261 |
| bonsai | 6 | 15.20 | 0.390 | 0.503 | 15.4 | 3695 |
| bonsai | 9 | 16.34 | 0.434 | 0.449 | 21.1 | 4913 |

| Views | Our PSNR (2 scenes) | Free360 PSNR (8 scenes) | Diff | Our SSIM / LPIPS | Free360 SSIM / LPIPS |
|---|---|---|---|---|---|
| 3 | 14.70 | 14.10 | +0.60 | 0.315 / 0.568 | 0.296 / 0.529 |
| 6 | 16.39 | 15.83 | +0.56 | 0.367 / 0.450 | 0.351 / 0.480 |
| 9 | 17.19 | 17.16 | +0.03 | 0.389 / 0.413 | 0.402 / 0.443 |

- **M1 decision: PASSED** - every view count within 0.6 dB of Free360 (threshold 1.5 dB), and quality rises with the number of views. Caveat: Free360 numbers are 8-scene averages; ours are 2 scenes.
- Time: mean 14.1 min per run on the T4; ~65% of it is test-pose optimization (render step). 96 runs = ~22.5 GPU-hours (~11 h per account). This is above the tracker's 10-min fallback threshold -> decision on compute (paid Colab units / workstation / 4 scenes) pending with Farida.
- Earlier runs with InstantSplat's default test-pose start (garden 14.23 / 14.52 / 14.76) are kept on Drive as `results/C3_baseline_OLD_interp_init` (not used).

**Pose error - InstantSplat's numbers are NOT used**
- InstantSplat's `metrics.py` builds the reference pose from a camera-to-world rotation plus a world-to-camera translation, and compares it with its own world-to-camera poses. Its RPE_r came out at 78-145 deg, which is not plausible.
- New `scripts/pose_error.py` (tracker task F2.3): camera centres + orientations on both sides, one Sim(3) alignment on the training centres, then ATE (divided by scene size), mean rotation error (deg), and RPE over all camera pairs. Self-test passes (identity, Sim(3) invariance, w2c round trip, known 10-deg error) and a synthetic COLMAP `images.bin` test passes; a wrong camera order gives large errors, as it should.
- Notebook v2 runs it for every run, before and after pose refinement, and saves the poses to Drive. The 6 baseline runs will run once more with v2 to get valid pose errors (Colab GPU limit reached on 9 Nov, so this is pending).

**Still open (Block 1)**
- Share the Drive folder `3dgs-project` with Mona (Editor); Mona runs the notebook once on her account (M1.1).
- Overleaf + Zotero; DPR-Splat PDF.

## 2026-11-12 - Block 1 main work finished (Farida): baseline re-run with valid pose error; batch tools ready (F4.1)

**Baseline C3 re-run with notebook v2** (same settings; Colab T4; 86.9 min for 6 runs)

| Scene | Photos | PSNR | SSIM | LPIPS | ATE before -> after refinement | Rotation error before -> after (deg) | Time (min) |
|---|---|---|---|---|---|---|---|
| garden | 3 | 14.67 | 0.274 | 0.555 | 0.019 -> 0.018 | 1.09 -> 1.11 | 9.3 |
| garden | 6 | 17.85 | 0.364 | 0.392 | 0.026 -> 0.026 | 1.33 -> 1.32 | 12.4 |
| garden | 9 | 18.13 | 0.351 | 0.379 | 0.020 -> 0.019 | 0.75 -> 0.75 | 16.3 |
| bonsai | 3 | 14.77 | 0.362 | 0.577 | 0.107 -> 0.106 | 6.84 -> 6.83 | 9.8 |
| bonsai | 6 | 15.37 | 0.397 | 0.504 | 0.078 -> 0.078 | 4.51 -> 4.55 | 15.3 |
| bonsai | 9 | 16.55 | 0.444 | 0.446 | 0.138 -> 0.138 | 2.23 -> 2.24 | 20.7 |

- Mean PSNR 14.72 / 16.61 / 17.34 at 3 / 6 / 9 photos vs Free360 14.10 / 15.83 / 17.16 -> **M1 still passed** (within 0.8 dB).
- Run-to-run variation vs the 9 Nov runs: +0.02 to +0.27 dB PSNR (training is random) -> seed repeats are worth keeping.
- ATE = camera-position error divided by the scene size (0.02 = 2 %). `pose_error.py` validated on real data: garden training cameras within ~1 deg, as found on 9 Nov.
- **First observation for RQ3:** pose refinement barely changes the training poses (all changes < 0.05 deg). bonsai (indoor) starts with larger errors (up to 6.8 deg at 3 photos) and refinement does not reduce them.
- Mean time per run 14.0 min -> 96 runs ~22.4 GPU-hours on the T4.

**Saved on Drive so nothing is repeated**
- `data/mipnerf360_8scenes_images_8.tar` (779 MB): dataset copy -> no 12 GB download in later sessions (F2.2 done).
- `init_cache/<scene>/<n>_views/`: MASt3R output (init_geo) for garden + bonsai x 3/6/9 (29-90 MB each). The MASt3R step is identical for C1-C4, so it runs once per scene x photo count and is reused (saves ~1-2 min per run, ~1.5 GPU-h over the batch, and every config starts from exactly the same AI output).
- Results, logs, test renders and camera poses of every run.

**Batch tools (F4.1)**
- `scripts/run_all.py`: config x scene x photos loop, result after every run, skips finished runs, reuses `C3_baseline` as C3, MASt3R cache. C2 = training without `--optim_pose` (InstantSplat then keeps the poses fixed; checked in train.py). C1 / C4 hooks are left for Block 2.
- `scripts/aggregate.py`: all_runs.csv, means_by_config.csv, effects.csv (RQ1 = C1-C2, RQ2 = C3-C4, RQ3 = C3-C2, matched scenes only); self-test passes.
- `scripts/colab_prepare.py` + `notebooks/02_experiments.ipynb`: experiments notebook that restores everything from Drive. Notebook 01 stays as the one-time setup + baseline.
