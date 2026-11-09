"""Camera-pose error for one sparse-view run (tracker task F2.3).

Why our own script: InstantSplat's metrics.py builds the reference pose from a camera-to-world
rotation and a world-to-camera translation (mixed), and compares it with its own world-to-camera
poses. Its ATE / RPE numbers are therefore not reliable (RPE_r came out at 78-145 degrees).

Here both sides are turned into camera centres + camera orientations, aligned with ONE Sim(3)
(Umeyama: scale, rotation, shift) fitted on the training camera centres, then:
  ate      RMSE of the camera-centre error, divided by the scene size
           (mean distance of the reference centres from their centroid) -> unit-free
  rot_deg  mean angle between aligned and reference camera orientations (degrees)
  rpe_t    relative-pose translation error over ALL camera pairs, divided by the scene size
  rpe_deg  relative-pose rotation error over ALL camera pairs (degrees; needs no alignment)
Sparse views are not a video, so all pairs are used instead of "consecutive" frames.

Usage:
  python pose_error.py --images-bin <scene>/sparse/0/images.bin \
                       --est <model>/pose/ours_1000/pose_optimized.npy \
                       --names IMG_1.JPG,IMG_9.JPG,...      (training photos, same order as --est)
  python pose_error.py --selftest
"""
import argparse
import json
import struct

import numpy as np


# ----------------------------------------------------------------- reading
def read_images_bin(path):
    """COLMAP images.bin -> {name: (R_w2c 3x3, t_w2c 3)}."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)                                         # image_id
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", f.read(56))
            f.read(4)                                         # camera_id
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)                                # x, y, point3D_id per point
            out[name.decode()] = (qvec_to_R(np.array([qw, qx, qy, qz])), np.array([tx, ty, tz]))
    return out


def qvec_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def w2c_to_cam(R_w2c, t_w2c):
    """world-to-camera -> (camera-to-world rotation, camera centre)."""
    return R_w2c.T, -R_w2c.T @ t_w2c


# ----------------------------------------------------------------- maths
def umeyama(X, Y):
    """Sim(3) with Y ~ s * R @ X + t (X, Y: N x 3)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


def angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(ref_rot, ref_ctr, est_rot, est_ctr):
    """All inputs: camera-to-world rotations (N,3,3) and camera centres (N,3), same camera order."""
    ref_rot, ref_ctr, est_rot, est_ctr = map(np.asarray, (ref_rot, ref_ctr, est_rot, est_ctr))
    n = len(ref_ctr)
    assert n == len(est_ctr) >= 3, "need the same number of cameras on both sides (at least 3)"
    s, R, t = umeyama(est_ctr, ref_ctr)
    al_ctr = (s * (R @ est_ctr.T)).T + t
    al_rot = R @ est_rot
    size = np.linalg.norm(ref_ctr - ref_ctr.mean(0), axis=1).mean()

    ate = np.sqrt(((al_ctr - ref_ctr) ** 2).sum(1).mean()) / size
    rot = np.mean([angle_deg(ref_rot[i], al_rot[i]) for i in range(n)])
    rpe_t, rpe_r = [], []
    for i in range(n):
        for j in range(i + 1, n):
            # relative motion of camera j seen from camera i
            g_t = ref_rot[i].T @ (ref_ctr[j] - ref_ctr[i])
            e_t = al_rot[i].T @ (al_ctr[j] - al_ctr[i])
            rpe_t.append(np.linalg.norm(g_t - e_t) / size)
            rpe_r.append(angle_deg(ref_rot[i].T @ ref_rot[j], al_rot[i].T @ al_rot[j]))
    return {"ate": float(ate), "rot_deg": float(rot), "rpe_t": float(np.mean(rpe_t)),
            "rpe_deg": float(np.mean(rpe_r)), "n_cams": n, "sim3_scale": float(s)}


def evaluate_run(images_bin, est_npy, names):
    """est_npy: InstantSplat pose_optimized.npy / pose_org.npy = world-to-camera 4x4, training order."""
    ref = read_images_bin(images_bin)
    est = np.load(est_npy).reshape(-1, 4, 4)
    assert len(est) == len(names), f"{len(est)} estimated poses but {len(names)} names"
    ref_cam = [w2c_to_cam(*ref[nm]) for nm in names]
    est_cam = [w2c_to_cam(P[:3, :3], P[:3, 3]) for P in est]
    return pose_error([r for r, _ in ref_cam], [c for _, c in ref_cam],
                      [r for r, _ in est_cam], [c for _, c in est_cam])


# ----------------------------------------------------------------- self-test
def random_rot(rng):
    q = rng.normal(size=4)
    return qvec_to_R(q)


def selftest():
    rng = np.random.default_rng(0)
    n = 9
    rot = np.stack([random_rot(rng) for _ in range(n)])
    ctr = rng.normal(size=(n, 3)) * 3

    # 1) identity: estimate == reference
    e = pose_error(rot, ctr, rot, ctr)
    assert all(e[k] < 1e-6 for k in ("ate", "rot_deg", "rpe_t", "rpe_deg")), e

    # 2) same cameras in another frame (scale 0.3, rotated, shifted) -> still ~0
    s, Rg, tg = 0.3, random_rot(rng), rng.normal(size=3)
    e = pose_error(rot, ctr, Rg @ rot, (s * (Rg @ ctr.T)).T + tg)
    assert all(e[k] < 1e-6 for k in ("ate", "rot_deg", "rpe_t", "rpe_deg")), e

    # 3) world-to-camera round trip (the format InstantSplat saves)
    w2c = [(r.T, -r.T @ c) for r, c in zip(rot, ctr)]
    back = [w2c_to_cam(*p) for p in w2c]
    assert np.allclose([b[1] for b in back], ctr) and np.allclose([b[0] for b in back], rot)

    # 4) a known 10-degree error on one camera is detected
    bad = rot.copy()
    ax = np.array([0.0, 0.0, np.radians(10)])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    bad[0] = rot[0] @ (np.eye(3) + np.sin(np.radians(10)) / np.radians(10) * K
                       + (1 - np.cos(np.radians(10))) / np.radians(10) ** 2 * K @ K)
    e = pose_error(rot, ctr, bad, ctr)
    assert abs(e["rot_deg"] - 10 / n) < 1e-6 and e["ate"] < 1e-9, e
    print("selftest OK - identity, Sim(3)-invariance, w2c round trip and a known 10-degree error all pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-bin")
    ap.add_argument("--est")
    ap.add_argument("--names", help="comma-separated training photo names, in the order of --est")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print(json.dumps(evaluate_run(a.images_bin, a.est, a.names.split(",")), indent=1))
