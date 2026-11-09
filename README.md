# Sparse-View 3DGS Ablation Study

**What Makes AI-Based Sparse-View 3D Gaussian Splatting Work?**
Separating the roles of predicted camera poses, predicted 3D points and pose refinement.

**Team:** Farida Gaber, Mona Yasser | **Supervisor:** Dr. Wedad Sallam | **Deadline:** 25 Nov 2026

## Idea in 3 lines
- 3D Gaussian Splatting needs camera poses; with 3-9 photos, COLMAP often fails.
- InstantSplat uses an AI model (MASt3R) for poses and 3D points, then refines the poses.
- We change one part at a time (reference poses, random points, refinement off) on Mip-NeRF 360 to see which part matters most.

## Files
- `Related work/` - novelty check PDF + Elicit and Consensus results (what is already published)
- `2 - Research Proposal (Detailed).pdf` - full proposal, tools, CPU/GPU
- `3 - Presentation (Progress update).pptx` - 20 slides for the supervisor: proposal + first results (9 Nov)
- `4 - Project Tracker (Day by Day).xlsx` - tasks per person per day (workstation plan)
- **Live tracker (Google Sheet, update statuses here):** https://docs.google.com/spreadsheets/d/1JH4ZOhLBre2fi6JLzDMeKQmHHIarsd1svu3o1-JARoo/edit
- `5 - Project Tracker (Colab version).xlsx` - backup copy of the live tracker (9-25 Nov plan on Colab); may lag behind the Google Sheet
- `6 - Progress Report (9 Nov).docx` - 5-page progress report: what was done, first results, next steps (for anyone new to the project)
- `Proposed Idea.txt` - the idea in 8 lines
- `LOG.md` - what was done, results and decisions (start here)
- `notebooks/01_block1_setup_baseline.ipynb` - Colab notebook: setup + baseline (Run all)
- `notebooks/run_outputs/` - saved copies of finished notebook runs (with outputs)
- `scripts/pose_error.py` - our camera-pose error (ATE, rotation, RPE); `python scripts/pose_error.py --selftest`

## Status
- Block 1 (9-11 Nov), in progress: main result done - baseline reproduces Free360 within 0.6 dB PSNR (milestone M1 passed). Still open: notebook v2 re-run, Mona runs the notebook, Overleaf + Zotero, hand-off call. See `LOG.md`.


