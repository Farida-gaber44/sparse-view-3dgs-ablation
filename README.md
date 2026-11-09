# Sparse-View 3DGS Ablation Study

**What Makes AI-Based Sparse-View 3D Gaussian Splatting Work?**
Separating the roles of predicted camera poses, predicted 3D points and pose refinement.

**Team:** Farida Gaber, Mona Yasser · **Supervisor:** Dr. Wedad Sallam · **Deadline:** 25 Nov 2026

## Links
| What | Link |
|---|---|
| Live tracker (Google Sheet) - update task status here | [Project Tracker](https://docs.google.com/spreadsheets/d/1JH4ZOhLBre2fi6JLzDMeKQmHHIarsd1svu3o1-JARoo/edit) |
| Shared Google Drive folder - setup files, dataset copy, all results | [3dgs-project](https://drive.google.com/drive/folders/1VXgL1JKVs8lrfZSx1xRJKzLIpw3f48NM?usp=drive_link) |
| Latest progress report (start here if you are new) | [4-reports/Report.pdf](4-reports/Report.pdf) |
| Project log - every result and decision | [LOG.md](LOG.md) |

## The idea in 3 lines
- 3D Gaussian Splatting needs camera poses; with only 3-9 photos, the classic tool (COLMAP) often fails.
- InstantSplat uses an AI model (MASt3R) for the poses and 3D points, then refines the poses during training.
- We change one part at a time (reference poses, random points, refinement off) on Mip-NeRF 360 to see which part matters most.

## Folders
| Folder | Contents |
|---|---|
| [`1-proposal/`](1-proposal) | The idea in 8 lines, detailed research proposal (PDF) |
| [`2-related-work/`](2-related-work) | Novelty check and literature review (PDF), Elicit and Consensus search results |
| [`3-tracker/`](3-tracker) | Link to the live Google Sheet tracker + offline backup copy |
| [`4-reports/`](4-reports) | Progress report 1 (3-9 Nov: related work, setup, baseline results) and the presentation for the supervisor |
| [`notebooks/`](notebooks) | Colab notebook (`01_block1_setup_baseline.ipynb`) and saved runs with outputs (`run_outputs/`) |
| [`scripts/`](scripts) | `pose_error.py` - our camera-pose error (ATE, rotation, RPE) |
| [`assets/`](assets) | Images used in documents |

## How to run (Google Colab)
1. Open `notebooks/01_block1_setup_baseline.ipynb` in Colab and set `Runtime -> Change runtime type -> T4 GPU`.
2. `Runtime -> Run all`. Results are saved to the shared Drive folder after every run; finished runs are skipped on restart.
3. Check the pose-error script anywhere: `python scripts/pose_error.py --selftest`

## Status (9 Nov)
| Block | Dates | Lead | Status |
|---|---|---|---|
| B0 Related work + research gap | 3-6 Nov | Farida | Done |
| B1 Setup + baseline | 9-11 Nov | Farida | Main result done (baseline within 0.6 dB of Free360, milestone M1 passed); 4 small items open |
| B2 Switches C1, C2, C4 + pilot | 12-15 Nov | Mona | Next |
| B3 All 96 runs, tables, figures | 16-18 Nov | Both | Planned |
| B4 Analysis + paper draft | 19-23 Nov | Mona | Planned |
| Final: supervisor feedback, delivery | 24-25 Nov | Both | Planned |
