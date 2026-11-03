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
- `3 - Presentation (15 slides).pptx` - slides for the supervisor
- `4 - Project Tracker (Day by Day).xlsx` - tasks per person per day
- `Proposed Idea.txt` - the idea in 8 lines


