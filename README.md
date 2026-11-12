# What Makes AI-Based Sparse-View 3D Gaussian Splatting Work?

**Separating the roles of predicted camera poses, predicted 3D points and pose refinement**

| Team | Supervisor | Deadline |
|---|---|---|
| Farida Gaber, Mona Yasser | Dr. Wedad Sallam | 25 November 2026 |

**Quick links:** [Live tracker](https://docs.google.com/spreadsheets/d/1JH4ZOhLBre2fi6JLzDMeKQmHHIarsd1svu3o1-JARoo/edit) · [Shared Drive folder](https://drive.google.com/drive/folders/1VXgL1JKVs8lrfZSx1xRJKzLIpw3f48NM?usp=drive_link) · [Progress report](4-reports/Report.pdf) · [Project log](LOG.md)

## Overview
3D Gaussian Splatting (3DGS) builds a 3D scene from photos, but it needs the camera pose of every photo. With only a few photos, the classic tool (COLMAP) often fails. AI-based methods such as **InstantSplat** solve this by combining three parts: AI-predicted camera poses, AI-predicted 3D points (MASt3R), and pose refinement during training.

**Research question:** how much of the final quality comes from each of these three parts, and how does this change from 3 to 9 photos? In particular, do the dense-view findings of VGGT-X still hold with only 3–9 photos?

This is an analysis (ablation) study of an existing method, not a new algorithm.

## Method
We change one part at a time and keep everything else fixed.

| Config | Camera poses | 3D points | Refinement | Purpose |
|---|---|---|---|---|
| C1 | Reference (dataset) | MASt3R | Off | Best case for poses |
| C2 | MASt3R | MASt3R | Off | Pure AI prediction |
| C3 | MASt3R | MASt3R | On | Full InstantSplat (baseline) |
| C4 | MASt3R | Random | On | Value of the AI points |

- **Dataset:** Mip-NeRF 360, 8 scenes (flowers excluded), 3 / 6 / 9 training photos, every 8th photo used for testing (Free360 protocol).
- **Metrics:** PSNR, SSIM, LPIPS (image quality), camera-pose error, run time and GPU memory.
- **Scale:** 4 configurations × 3 photo counts × 8 scenes = 96 runs on Google Colab (T4 GPU).

## Results so far
Our baseline (C3, full InstantSplat; re-run 12 Nov with the final protocol) is consistent with the InstantSplat numbers published by Free360 (milestone M1):

| Photos | Our PSNR (garden + bonsai) | Free360 PSNR (8 scenes) | Difference |
|---|---|---|---|
| 3 | 14.72 | 14.10 | +0.62 |
| 6 | 16.61 | 15.83 | +0.78 |
| 9 | 17.34 | 17.16 | +0.18 |

Caveats: 2 vs 8 scenes, Free360 used a higher resolution for indoor scenes, and run-to-run noise is about 0.3 dB. An independent audit (12 Nov) verified the code and fixed the protocol issues it found. Details in the [project log](LOG.md).

## Repository structure
| Folder | Contents |
|---|---|
| [`1-proposal`](1-proposal) | Project idea and detailed research proposal |
| [`2-related-work`](2-related-work) | Literature review, novelty check, Elicit and Consensus search results |
| [`3-tracker`](3-tracker) | Link to the live task tracker |
| [`4-reports`](4-reports) | Progress report, presentation, and the Block 2 handover for Mona (step by step) |
| [`notebooks`](notebooks) | Colab notebook for setup and experiments, plus saved runs |
| [`scripts`](scripts) | `pose_error.py` – camera-pose error with self-test |
| [`assets`](assets) | Figures used in documents |

## Getting started
1. Open [`notebooks/01_block1_setup_baseline.ipynb`](notebooks/01_block1_setup_baseline.ipynb) in Google Colab and select a **T4 GPU** runtime.
2. Click **Runtime → Run all**. Results are saved to the shared Drive folder after each run, and finished runs are skipped on restart.

## Timeline
| Block | Dates | Lead | Status |
|---|---|---|---|
| Related work and research gap | 3–6 Nov | Farida | Done |
| Setup and baseline | 9–11 Nov | Farida | Main result done |
| Experiment switches and pilot | 12–15 Nov | Mona | Next |
| Full experiments (96 runs) | 16–18 Nov | Both | Planned |
| Analysis and paper draft | 19–23 Nov | Mona | Planned |
| Supervisor feedback and delivery | 24–25 Nov | Both | Planned |

## Built on
[InstantSplat](https://github.com/NVlabs/InstantSplat) (NVIDIA) · [MASt3R](https://github.com/naver/mast3r) (Naver) · [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)
