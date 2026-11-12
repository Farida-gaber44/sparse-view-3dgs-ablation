"""Recompute the pose error of finished runs with the current pose_error.py (CPU only, no retraining).

Every run saves its training poses (results/<config>/<scene>/<n>_views/poses/pose_org.npy and
pose_optimized.npy), so a better pose-error measure can be applied later without using the GPU.
Old values are kept in result.json under "pose_v<old version>".

    import recompute_pose_error as rp
    rp.recompute_all(DRIVE + "/results", DATA)
"""
import json
from pathlib import Path

import pose_error

KEYS = ["ate", "rot_deg", "rpe_t", "rpe_deg"]


def recompute_run(run_dir, data):
    run_dir = Path(run_dir)
    r = json.load(open(run_dir / "result.json"))
    old_version = r.get("pose_error_version", 1)
    if r.get("pose_metric") != "pose_error.py" or old_version >= pose_error.VERSION:
        return None
    if not (run_dir / "poses" / "pose_optimized.npy").exists():
        return None
    gt = f"{data}/{r['scene']}/sparse/0/images.bin"
    names = r["train_images"]
    after = pose_error.evaluate_run(gt, run_dir / "poses" / "pose_optimized.npy", names)
    before = pose_error.evaluate_run(gt, run_dir / "poses" / "pose_org.npy", names)
    r[f"pose_v{old_version}"] = {k: r.get(k) for k in KEYS + [k + "_init" for k in KEYS]}
    r.update({k: after[k] for k in KEYS})
    r.update({k + "_init": before[k] for k in KEYS})
    r["flatness"] = after["flatness"]
    r["pose_error_version"] = pose_error.VERSION
    json.dump(r, open(run_dir / "result.json", "w"), indent=1)
    return r


def recompute_all(results_root, data):
    changed = []
    for f in sorted(Path(results_root).glob("*/*/*_views/result.json")):
        r = recompute_run(f.parent, data)
        if r:
            old = r[[k for k in r if k.startswith("pose_v")][0]]
            print(f"{f.parent.relative_to(results_root)}: rot {old['rot_deg']:.2f} -> {r['rot_deg']:.2f} deg, "
                  f"ATE {old['ate']:.3f} -> {r['ate']:.3f}, flatness {r['flatness']:.2f}")
            changed.append(r)
    print(f"{len(changed)} runs updated to pose error version {pose_error.VERSION}")
    return changed
