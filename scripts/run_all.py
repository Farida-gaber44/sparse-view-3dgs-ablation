"""Batch runner for the ablation study (tracker task F4.1).

Runs every (config x scene x photo count) once, saves a small result.json + logs + test renders +
camera poses to Drive after EVERY run, and skips runs that are already finished, so a Colab
session can stop at any time and the next "Run all" continues where it stopped.

Configurations (see the proposal):
  C1  reference poses   + MASt3R points + refinement OFF   (needs hook: use_reference_poses - Block 2, Mona)
  C2  MASt3R poses      + MASt3R points + refinement OFF   (works now: train without --optim_pose)
  C3  MASt3R poses      + MASt3R points + refinement ON    (works now: full InstantSplat)
  C4  MASt3R poses      + random points + refinement ON    (needs hook: use_random_points - Block 2, Mona)

Usage in Colab (after the setup cells of the notebook):
  import run_all
  ctx = run_all.Context(drive=DRIVE, repo=REPO, data=DATA, local_out=LOCAL_OUT, manifest=manifest,
                        commit=COMMIT, gpu=GPU_NAME)
  run_all.run_batch(ctx, configs=["C2", "C3"], scenes=["garden"], views=[3])
  run_all.run_batch(ctx, ..., dry_run=True)     # only print what would run
"""
import datetime
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pose_error

POSE_KEYS = ["ate", "rot_deg", "rpe_t", "rpe_deg"]
EXPECTED_COMMIT = "b951567"     # InstantSplat version used for ALL runs (pinned in setup)
LLFFHOLD = 8                    # init_geo.py and metrics.py use InstantSplat's default llffhold = 8


# ----------------------------------------------------------------- settings shared by all runs
@dataclass
class Context:
    drive: str                       # /content/drive/MyDrive/3dgs-project
    repo: str                        # /content/InstantSplat
    data: str                        # /content/data
    local_out: str                   # /content/out (heavy files, not kept)
    manifest: dict                   # views_manifest.json: exact train/test photos per scene
    commit: str = ""
    gpu: str = ""
    res_dir: str = "images_8"
    iters: int = 1000
    optim_test_pose_iter: int = 500
    test_stride: int = 8
    seed: int = 0                    # used by the C4 switch (random points); InstantSplat itself seeds training with 0
    env_file: str = ""               # package list of this Colab session (written once per session)
    reuse: dict = field(default_factory=lambda: {"C3": "C3_baseline"})   # finished baseline runs count as C3
    use_init_cache: bool = True      # reuse the saved MASt3R output (init_geo) for every config of a scene/photo count


# ----------------------------------------------------------------- MASt3R output cache (saves GPU time)
# init_geo.py writes everything into <scene>/sparse_<n> (poses, points, confidences, test-pose start).
# Its output is the same for C1-C4, so we run it ONCE per scene x photo count and keep a copy on Drive.
# The switches (C1, C4) change only the local copy, after it has been restored.
def init_cache_dir(ctx, scene, n_views):
    return f"{ctx.drive}/init_cache/{scene}/{n_views}_views"


def save_init_cache(ctx, scene, n_views, init_seconds):
    src = Path(ctx.data) / scene / f"sparse_{n_views}"
    assert (src / "0" / "points3D.ply").exists(), f"no init_geo output in {src}"
    d = init_cache_dir(ctx, scene, n_views)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), f"init_{scene}_{n_views}.tar")
    subprocess.run(["tar", "-cf", tmp, "-C", str(src.parent), src.name], check=True)
    shutil.copy(tmp, f"{d}/sparse.tar.part")
    os.replace(f"{d}/sparse.tar.part", f"{d}/sparse.tar")
    os.remove(tmp)
    json.dump({"init_geo_seconds": init_seconds, "instantsplat_commit": ctx.commit, "gpu": ctx.gpu,
               "date": datetime.datetime.now().isoformat(timespec="seconds")}, open(f"{d}/info.json", "w"), indent=1)


def restore_init_cache(ctx, scene, n_views):
    """Return the original init_geo seconds if the cache was restored, else None."""
    d = init_cache_dir(ctx, scene, n_views)
    if not (ctx.use_init_cache and os.path.exists(f"{d}/sparse.tar") and os.path.exists(f"{d}/info.json")):
        return None
    info = json.load(open(f"{d}/info.json"))
    if ctx.commit and info.get("instantsplat_commit") and info["instantsplat_commit"] != ctx.commit:
        raise RuntimeError(f"MASt3R cache {d} was made with InstantSplat {info['instantsplat_commit']}, "
                           f"this session uses {ctx.commit}. Do not mix versions.")
    shutil.rmtree(f"{ctx.data}/{scene}/sparse_{n_views}", ignore_errors=True)
    subprocess.run(["tar", "-xf", f"{d}/sparse.tar", "-C", f"{ctx.data}/{scene}"], check=True)
    return info["init_geo_seconds"]


def init_train_names(ctx, scene, n_views):
    """Training photo names written by init_geo (sparse_<n>/0/images.txt), in camera-id order."""
    rows = {}
    with open(f"{ctx.data}/{scene}/sparse_{n_views}/0/images.txt") as f:
        for l in f:
            p = l.split()
            # image lines have exactly 10 fields (ID QW QX QY QZ TX TY TZ CAMERA_ID NAME);
            # 2D-point lines have a multiple of 3 fields or are empty
            if len(p) == 10 and not l.startswith("#"):
                rows[int(p[0])] = p[9]
    return [rows[k] for k in sorted(rows)]


def check_init_names(ctx, scene, n_views):
    """The MASt3R output must belong to exactly the training photos of the manifest (same order)."""
    got = init_train_names(ctx, scene, n_views)
    want = ctx.manifest[scene]["train_images"][str(n_views)]
    if got != want:
        raise RuntimeError(f"{scene} {n_views} views: MASt3R output is for {got}, manifest says {want}")


def env_snapshot(ctx):
    """Save the package list of this session once (Colab updates its software without warning)."""
    if ctx.env_file:
        return ctx.env_file
    os.makedirs(f"{ctx.drive}/env", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    ctx.env_file = f"{ctx.drive}/env/pip_freeze_{stamp}.txt"
    with open(ctx.env_file, "w") as f:
        f.write(subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout)
    return ctx.env_file


def check_context(ctx):
    assert ctx.test_stride == LLFFHOLD, "init_geo.py / metrics.py use llffhold = 8; keep test_stride = 8"
    if ctx.commit and not ctx.commit.startswith(EXPECTED_COMMIT[:7]):
        raise RuntimeError(f"InstantSplat commit is {ctx.commit}, expected {EXPECTED_COMMIT}.")


# ----------------------------------------------------------------- the switches
def hook_none(ctx, scene, n_views, model):
    return {}


def use_reference_poses(ctx, scene, n_views, model):
    """C1 (Block 2, Mona): replace the MASt3R training poses written by init_geo.py
    (<scene>/sparse_<n>/0/images.bin|txt) with the dataset's reference poses, and move the MASt3R
    points into the reference frame with one Sim(3) (Umeyama) fitted on the camera centres
    (reuse pose_error.umeyama). Return what was done, e.g. {"sim3_scale": s}."""
    raise NotImplementedError("C1 switch is a Block 2 task (M3.1)")


def use_random_points(ctx, scene, n_views, model):
    """C4 (Block 2, Mona): replace the MASt3R points written by init_geo.py with the SAME number of
    points drawn uniformly in the bounding box of (points + camera centres), fixed seed ctx.seed,
    random colours. Return {"n_points": N, "bbox_min": [...], "bbox_max": [...]} for the log."""
    raise NotImplementedError("C4 switch is a Block 2 task (M2.1)")


use_reference_poses.not_ready = True     # remove when the switch is written (notebook 02 replaces these)
use_random_points.not_ready = True

CONFIGS = {
    "C1": dict(after_init=use_reference_poses, optim_pose=False, label="reference poses, refinement off"),
    "C2": dict(after_init=hook_none, optim_pose=False, label="MASt3R poses, refinement off"),
    "C3": dict(after_init=hook_none, optim_pose=True, label="full InstantSplat"),
    "C4": dict(after_init=use_random_points, optim_pose=True, label="random points, refinement on"),
}


# ----------------------------------------------------------------- helpers
class VramSampler(threading.Thread):
    """Highest GPU memory used while a run is going (nvidia-smi every 2 s)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop_flag = 0, False

    def run(self):
        while not self.stop_flag:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                self.peak = max(self.peak, int(r.stdout.split()[0]))
            except Exception:
                pass
            time.sleep(2)


def result_dir(ctx, config, scene, n_views):
    return f"{ctx.drive}/results/{config}/{scene}/{n_views}_views"


def is_done(path):
    f = Path(path) / "result.json"
    if not f.exists():
        return False
    return json.load(open(f)).get("pose_metric") == "pose_error.py"


def commands(ctx, cfg, src, model, n_views):
    pose_flag = " --optim_pose" if cfg["optim_pose"] else ""
    return [
        ("init_geo", f"python -W ignore init_geo.py -s {src} -m {model} --n_views {n_views} --focal_avg --co_vis_dsp --conf_aware_ranking"),
        ("train", f"python train.py -s {src} -m {model} -r 1 --n_views {n_views} --iterations {ctx.iters} --pp_optimizer{pose_flag}"),
        ("render_test", f"python render.py -s {src} -m {model} -r 1 --n_views {n_views} --iterations {ctx.iters} --eval --optim_test_pose_iter {ctx.optim_test_pose_iter}"),
        ("metrics", f"python metrics.py -s {src} -m {model} --n_views {n_views}"),
    ]


# ----------------------------------------------------------------- one run
def run_one(ctx, config, scene, n_views, force=False, dry_run=False):
    cfg = CONFIGS[config]
    drive_dir = result_dir(ctx, config, scene, n_views)
    if not force and is_done(drive_dir):
        print(f"skip (done): {config}/{scene}/{n_views}_views")
        return json.load(open(f"{drive_dir}/result.json"))
    old = ctx.reuse.get(config)
    if not force and old and is_done(result_dir(ctx, old, scene, n_views)):
        os.makedirs(os.path.dirname(drive_dir), exist_ok=True)
        if not dry_run:
            shutil.copytree(result_dir(ctx, old, scene, n_views), drive_dir, dirs_exist_ok=True)
        print(f"reuse {old} -> {config}/{scene}/{n_views}_views (same settings)")
        return json.load(open(f"{result_dir(ctx, old, scene, n_views)}/result.json"))

    src = f"{ctx.data}/{scene}"
    model = f"{ctx.local_out}/{config}/{scene}/{n_views}_views"
    steps = commands(ctx, cfg, src, model, n_views)
    print(f"=== {config}/{scene}/{n_views}_views ({cfg['label']})")
    if dry_run:
        for name, cmd in steps:
            print(f"   [{name}] {cmd}")
        return None

    shutil.rmtree(f"{src}/sparse_{n_views}", ignore_errors=True)   # init_geo writes here; start clean (or restore cache)
    shutil.rmtree(model, ignore_errors=True)
    os.makedirs(model, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)

    sampler = VramSampler()
    sampler.start()
    stage_s, switch_info, t_all = {}, {}, time.time()
    env_snapshot(ctx)
    init_cached = restore_init_cache(ctx, scene, n_views)
    try:
        for name, cmd in steps:
            if name == "init_geo" and init_cached is not None:
                stage_s[name] = 0.0
                print(f"   init_geo     from Drive cache (originally {init_cached:.0f} s)")
                check_init_names(ctx, scene, n_views)
                switch_info = cfg["after_init"](ctx, scene, n_views, model) or {}
                continue
            t0 = time.time()
            log = f"{model}/{name}.log"
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, shell=True, cwd=ctx.repo, stdout=lf, stderr=subprocess.STDOUT).returncode
            stage_s[name] = round(time.time() - t0, 1)
            print(f"   {name:12s} {stage_s[name]:7.1f} s")
            if rc != 0:
                tail = "".join(open(log).readlines()[-25:])
                raise RuntimeError(f"Stage '{name}' failed for {config}/{scene}/{n_views}. Last lines:\n{tail}")
            if name == "init_geo":
                check_init_names(ctx, scene, n_views)
                if ctx.use_init_cache:
                    save_init_cache(ctx, scene, n_views, stage_s[name])      # BEFORE any switch changes it
                switch_info = cfg["after_init"](ctx, scene, n_views, model) or {}
    finally:
        sampler.stop_flag = True

    if not os.path.exists(f"{model}/results.json"):           # metrics.py hides its own errors
        tail = "".join(open(f"{model}/metrics.log").readlines()[-25:])
        raise RuntimeError(f"metrics.py wrote no results.json for {config}/{scene}/{n_views}. Log:" + chr(10) + f"{tail}")
    res = json.load(open(f"{model}/results.json"))
    m = res[sorted(res)[0]]
    renders = Path(f"{model}/test/ours_{ctx.iters}/renders")
    pose_dir = Path(f"{model}/pose/ours_{ctx.iters}")
    names = ctx.manifest[scene]["train_images"][str(n_views)]
    gt_bin = f"{src}/sparse/0/images.bin"
    pe_after = pose_error.evaluate_run(gt_bin, pose_dir / "pose_optimized.npy", names)
    pe_before = pose_error.evaluate_run(gt_bin, pose_dir / "pose_org.npy", names)

    os.makedirs(f"{drive_dir}/poses", exist_ok=True)
    for f in ["pose_org.npy", "pose_optimized.npy"]:
        shutil.copy(pose_dir / f, f"{drive_dir}/poses/{f}")
    for f in ["results.json", "per_view.json", "train_time.txt"] + [p.name for p in Path(model).glob("*.log")]:
        if os.path.exists(f"{model}/{f}"):
            shutil.copy(f"{model}/{f}", drive_dir)
    shutil.copytree(renders, f"{drive_dir}/test_renders", dirs_exist_ok=True)

    result = dict(config=config, config_label=cfg["label"], scene=scene, n_views=n_views,
                  psnr=m["PSNR"], ssim=m["SSIM"], lpips=m["LPIPS"],
                  **{k: pe_after[k] for k in POSE_KEYS},
                  **{k + "_init": pe_before[k] for k in POSE_KEYS},
                  pose_metric="pose_error.py", pose_error_version=pe_after["pose_error_version"],
                  flatness=pe_after["flatness"], train_images=names, switch_info=switch_info,
                  init_from_cache=init_cached is not None,
                  init_geo_seconds=init_cached if init_cached is not None else stage_s.get("init_geo"),
                  n_test_images=len(list(renders.glob("*"))),
                  stage_seconds=stage_s, total_seconds=round(time.time() - t_all, 1),
                  total_seconds_incl_init=round(time.time() - t_all + (init_cached or 0), 1),
                  peak_vram_mb=sampler.peak, gpu=ctx.gpu, resolution=ctx.res_dir, iterations=ctx.iters,
                  optim_test_pose_iter=ctx.optim_test_pose_iter, test_stride=ctx.test_stride, seed=ctx.seed,
                  instantsplat_commit=ctx.commit, env_file=ctx.env_file, date=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(result, open(f"{drive_dir}/result.json", "w"), indent=1)
    print(f"   done: PSNR {result['psnr']:.2f} | pose before->after: ATE {result['ate_init']:.3f}->{result['ate']:.3f}, "
          f"rot {result['rot_deg_init']:.2f}->{result['rot_deg']:.2f} deg | {result['total_seconds'] / 60:.1f} min")
    return result


# ----------------------------------------------------------------- many runs
def run_batch(ctx, configs, scenes, views, force=False, dry_run=False, stop_on_error=False):
    """Loop config x scene x views. A failed run is reported and skipped (unless stop_on_error)."""
    check_context(ctx)
    t0, done, failed, unavailable = time.time(), [], [], set()
    for config in configs:
        if getattr(CONFIGS[config]["after_init"], "not_ready", False):   # switch not written yet: no GPU spent
            print(f"   {config}: switch not written yet - skipped")
            failed.append((config, "-", "-", "switch not written yet"))
            continue
        for scene in scenes:
            for v in views:
                if config in unavailable:
                    continue
                try:
                    r = run_one(ctx, config, scene, v, force=force, dry_run=dry_run)
                    if r:
                        done.append(r)
                except NotImplementedError as e:
                    print(f"   not available yet: {e}")
                    failed.append((config, scene, v, str(e)))
                    unavailable.add(config)                # skip the rest of this config
                except Exception as e:
                    print(f"   FAILED: {e}")
                    failed.append((config, scene, v, str(e)[:300]))
                    if stop_on_error:
                        raise
    print(f"\nBatch finished in {(time.time() - t0) / 60:.1f} min: {len(done)} runs ready, {len(failed)} not done.")
    for f in failed:
        print("   not done:", f[:3], "-", f[3][:120])
    return done, failed
