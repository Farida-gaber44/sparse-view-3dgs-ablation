"""Shared Colab preparation for every experiment notebook (02, 03, ...).

Notebook 01 must have been run once on the account: it builds the GPU programs and saves them,
the MASt3R model, setup_colab.sh and the dataset copy to Drive. This module only restores them.

    import colab_prepare as cp
    gpu = cp.check_gpu()
    commit = cp.install(DRIVE, REPO)
    cp.restore_data(DRIVE, DATA, SCENES, RES_DIR)
    cp.patch_instantsplat(REPO)
    manifest = cp.build_manifest(DATA, SCENES, VIEWS, RES_DIR, TEST_STRIDE, DRIVE)

The two InstantSplat patches below are the same as in notebook 01 (same markers, so a patch is
never applied twice):
  1. Free360-style split: test = every 8th photo; train = n photos spaced evenly among the rest.
  2. Aligned test-pose start: each test camera starts from its reference pose mapped into the
     estimated frame with one Sim(3) fitted on the training cameras (InstantSplat's default start
     was 80-90 degrees off in 360-degree scenes).
"""
import json
import os
import numpy as np
import re
import struct
import subprocess
from pathlib import Path


def sh(cmd, cwd=None):
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{(p.stdout + p.stderr)[-3000:]}")
    return p.stdout


# ----------------------------------------------------------------- GPU, install, data
def check_gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit("NO GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again.")
    name = r.stdout.strip()
    if "T4" not in name:
        print("WARNING: not a T4. All FINAL results must come from the same GPU type.")
    return name


INSTANTSPLAT_COMMIT = "b951567"   # every run of the study uses this exact version


def pin_commit(repo, commit=INSTANTSPLAT_COMMIT):
    """Make sure the cloned InstantSplat is exactly the pinned version (GitHub may move on)."""
    head = sh(f"cd {repo} && git rev-parse --short HEAD").strip()
    if not head.startswith(commit[:7]):
        sh(f"cd {repo} && git fetch -q origin && git checkout -q {commit} && git submodule update -q --init --recursive")
        head = sh(f"cd {repo} && git rev-parse --short HEAD").strip()
    assert head.startswith(commit[:7]), f"InstantSplat is at {head}, expected {commit}"
    return head


def install(drive, repo):
    """Run the setup script saved by notebook 01 (cached GPU programs -> about 1-3 minutes)."""
    quick = subprocess.run(["python", "-c", "import diff_gaussian_rasterization, simple_knn, fused_ssim, open3d"],
                           capture_output=True, text=True)
    if not (os.path.isdir(repo) and quick.returncode == 0):
        setup = f"{drive}/setup_colab.sh"
        if not os.path.exists(setup):
            raise SystemExit("setup_colab.sh is not on Drive. Run notebook 01 once on this account first.")
        print(sh(f"bash {setup}")[-300:])
    return pin_commit(repo)


def restore_data(drive, data, scenes, res_dir="images_8"):
    """Unpack the dataset copy that notebook 01 saved to Drive and link images -> images_8."""
    def ready(s):
        return Path(f"{data}/{s}/{res_dir}").is_dir() and Path(f"{data}/{s}/sparse/0/images.bin").exists()

    tar = f"{drive}/data/mipnerf360_8scenes_{res_dir}.tar"
    if not all(ready(s) for s in scenes):
        if not os.path.exists(tar):
            raise SystemExit(f"Dataset copy not found on Drive ({tar}). Run notebook 01 (version 2) once first.")
        os.makedirs(data, exist_ok=True)
        sh(f"tar -xf '{tar}' -C {data}")
    counts = {}
    for s in scenes:
        assert ready(s), f"scene {s} missing after unpacking"
        link = Path(f"{data}/{s}/images")
        if not link.exists():
            os.symlink(res_dir, link)
        counts[s] = len(os.listdir(f"{data}/{s}/{res_dir}"))
    return counts


# ----------------------------------------------------------------- InstantSplat patches (same as notebook 01)
SPLIT_SRC = '''def split_train_test(image_files, llffhold=8, n_views=None, verbose=True):
    # FREE360-STYLE SPLIT (patched by our notebook):
    # test = every llffhold-th image (idx % llffhold == 0); train = n_views images spaced evenly among the rest
    n = len(image_files)
    test_idx = [i for i in range(n) if i % llffhold == 0]
    train_pool = [i for i in range(n) if i % llffhold != 0]
    sparse_idx = np.linspace(0, len(train_pool) - 1, num=n_views, dtype=int)
    train_idx = [train_pool[i] for i in sparse_idx]
    if verbose:
        print(">> Spliting Train-Test Set (Free360-style): ")
        print(" - train_set_indices:  ", train_idx)
        print(" - test_set_indices:   ", test_idx)
    train_img_files = [image_files[i] for i in train_idx]
    test_img_files = [image_files[i] for i in test_idx]
    return train_img_files, test_img_files
'''

INIT_FN = '''
def aligned_test_init(source_path, train_files, test_files, train_w2c):
    # ALIGNED TEST-POSE INIT: Sim(3) (Umeyama) from reference to estimated training camera centres,
    # then the reference test poses are mapped into the estimated frame. Same rule for every configuration.
    import numpy as np, os
    from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
    gt = {os.path.basename(v.name): v for v in read_extrinsics_binary(os.path.join(source_path, "sparse/0/images.bin")).values()}
    def gt_w2c(f):
        e = gt[os.path.basename(f)]
        M = np.eye(4); M[:3, :3] = qvec2rotmat(e.qvec); M[:3, 3] = e.tvec
        return M
    est_c2w = np.linalg.inv(np.asarray(train_w2c))
    gt_c2w = np.stack([np.linalg.inv(gt_w2c(f)) for f in train_files])
    X, Y = gt_c2w[:, :3, 3], est_c2w[:, :3, 3]
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / Xc.var(0).sum(); t = my - s * R @ mx
    out = []
    for f in test_files:
        g = np.linalg.inv(gt_w2c(f)); c2w = np.eye(4)
        c2w[:3, :3] = R @ g[:3, :3]; c2w[:3, 3] = s * R @ g[:3, 3] + t
        out.append(np.linalg.inv(c2w))
    return np.stack(out)
'''


def patch_instantsplat(repo):
    sfm = Path(f"{repo}/utils/sfm_utils.py")
    text = sfm.read_text()
    if "FREE360-STYLE SPLIT" not in text:
        new, n = re.subn(r"def split_train_test\(.*?\n(?=def get_sorted_image_files)", lambda m: SPLIT_SRC + "\n\n",
                         text, flags=re.S)
        assert n == 1, "split_train_test not found (InstantSplat changed?)"
        compile(new, "sfm_utils.py", "exec")
        sfm.write_text(new)
        print("patched: Free360-style split")
    ig = Path(f"{repo}/init_geo.py")
    text = ig.read_text()
    if "ALIGNED TEST-POSE INIT" not in text:
        anchor = "        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)"
        assert text.count(anchor) == 1 and text.count("\ndef main(") == 1, "init_geo.py changed - check the patch"
        text = text.replace(anchor, "        pose_test_init = aligned_test_init(source_path, train_img_files, test_img_files, extrinsics_w2c)\n" + anchor)
        text = text.replace("\ndef main(", INIT_FN + "\n\ndef main(", 1)
        compile(text, "init_geo.py", "exec")
        ig.write_text(text)
        print("patched: aligned test-pose start")
    return True


# ----------------------------------------------------------------- exact photo lists
def colmap_image_names(path):
    names = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(64)
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            names.append(name.decode())
    return names


def instantsplat_order(folder):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}]

    def num(p):
        m = re.search(r"\d+", p.stem)
        return int(m.group()) if m else 10 ** 18
    return [p.name for p in sorted(files, key=num)]


def build_manifest(data, scenes, views, res_dir="images_8", test_stride=8, drive=None):
    import numpy as np                                    # used by SPLIT_SRC
    ns = {"np": np}
    exec(SPLIT_SRC, ns)
    split = ns["split_train_test"]
    manifest = {}
    for s in scenes:
        ours = instantsplat_order(f"{data}/{s}/{res_dir}")
        colmap = sorted(colmap_image_names(f"{data}/{s}/sparse/0/images.bin"))
        assert ours == colmap, f"{s}: photo order differs between InstantSplat and the reference poses"
        manifest[s] = {"n_images": len(ours), "test_images": split(ours, test_stride, 3, verbose=False)[1],
                       "train_images": {str(v): split(ours, test_stride, v, verbose=False)[0] for v in views}}
    if drive:
        saved = f"{drive}/views_manifest.json"
        if os.path.exists(saved):                         # the photo lists must never change between notebooks
            old = json.load(open(saved))
            for s in scenes:
                if s in old:
                    assert old[s]["test_images"] == manifest[s]["test_images"], f"{s}: test photos changed!"
                    for v in views:
                        if str(v) in old[s]["train_images"]:
                            assert old[s]["train_images"][str(v)] == manifest[s]["train_images"][str(v)], \
                                f"{s} {v} views: training photos changed!"
    return manifest


# ----------------------------------------------------------------- where the training cameras are
def camera_coverage(data, manifest, scenes, views, drive=None):
    """For each scene x photo count: how the training cameras are spread around the scene.
    span_deg  = largest arc covered around the scene (360 = all sides); a small value means one-sided views
    flatness  = 2nd / 1st singular value of the camera positions (0 = on a line; low values make 3-camera
                alignment less reliable)
    The photo choice is InstantSplat's rule (evenly spaced by photo number). Several outdoor captures go
    around the object many times, so this can give one-sided sets (e.g. bicycle 3 photos)."""
    from pose_error import read_images_bin, w2c_to_cam
    out = {}
    for s in scenes:
        ref = read_images_bin(f"{data}/{s}/sparse/0/images.bin")
        allc = np.array([w2c_to_cam(*ref[n])[1] for n in sorted(ref)])
        mu = allc.mean(0)
        _, _, vt = np.linalg.svd(allc - mu)
        az = lambda c: float(np.degrees(np.arctan2((c - mu) @ vt[1], (c - mu) @ vt[0])) % 360)
        out[s] = {}
        for v in views:
            names = manifest[s]["train_images"][str(v)]
            c = np.array([w2c_to_cam(*ref[n])[1] for n in names])
            a = sorted(az(x) for x in c)
            gaps = [(a[(i + 1) % len(a)] - a[i]) % 360 for i in range(len(a))]
            sv = np.linalg.svd(c - c.mean(0), compute_uv=False)
            out[s][str(v)] = {"azimuths_deg": [round(x) for x in a], "span_deg": round(360 - max(gaps)),
                              "flatness": round(float(sv[1] / sv[0]), 3)}
    if drive:
        json.dump(out, open(f"{drive}/views_coverage.json", "w"), indent=1)
    return out
