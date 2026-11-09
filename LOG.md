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
