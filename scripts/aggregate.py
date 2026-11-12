"""Collect all result.json files into tables (tracker task F4.1).

Output (written next to the results folder):
  all_runs.csv          one row per run
  means_by_config.csv   mean over scenes for each config x photo count
  effects.csv           the three research questions as differences of means:
                          RQ1 poses      = C1 - C2
                          RQ2 3D points  = C3 - C4
                          RQ3 refinement = C3 - C2
Usage:
  python aggregate.py /content/drive/MyDrive/3dgs-project/results
  python aggregate.py --selftest
"""
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
ALIASES = {"C3_baseline": "C3"}      # baseline runs of notebook 01 = C3 with the same settings
METRICS = ["psnr", "ssim", "lpips", "ate", "rot_deg", "rpe_deg", "ate_init", "rot_deg_init",
           "total_seconds_incl_init", "total_seconds", "peak_vram_mb"]
EFFECTS = [("RQ1 poses (C1 - C2)", "C1", "C2"),
           ("RQ2 3D points (C3 - C4)", "C3", "C4"),
           ("RQ3 refinement (C3 - C2)", "C3", "C2")]


def load(results_root):
    """One row per (config, scene, photos). Old-format runs are skipped; a C3_baseline run counts as C3
    unless a separate C3 run exists for the same scene and photo count."""
    rows, aliased = {}, {}
    for f in sorted(Path(results_root).glob("*/*/*_views/result.json")):
        r = json.load(open(f))
        if r.get("pose_metric") != "pose_error.py":
            continue                                   # old runs without a valid pose error
        c = r.get("config")
        if c in ALIASES:
            r = dict(r, config=ALIASES[c], source_config=c)
            target = aliased
        elif c in CONFIGS:
            target = rows
        else:
            continue
        # time of the full pipeline: runs that reused the saved MASt3R output add its original time back
        if r.get("total_seconds") is not None:
            extra = r.get("init_geo_seconds") if r.get("init_from_cache") else 0
            r["total_seconds_incl_init"] = r["total_seconds"] + (extra or 0)
        target[(r["config"], r["scene"], r["n_views"])] = r
    for key, r in aliased.items():
        rows.setdefault(key, r)
    return [rows[k] for k in sorted(rows)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def tables(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["n_views"])].append(r)
    means = {}
    for (c, v), rs in sorted(groups.items()):
        means[(c, v)] = {"config": c, "n_views": v, "n_scenes": len(rs),
                         "scenes": " ".join(sorted(r["scene"] for r in rs)),
                         **{m: mean([r.get(m) for r in rs]) for m in METRICS}}
    effects = []
    views = sorted({v for _, v in means})
    for name, a, b in EFFECTS:
        for v in views:
            if (a, v) in means and (b, v) in means:
                # compare only scenes present in BOTH configs, so a missing run cannot bias the difference
                sa = {r["scene"]: r for r in groups[(a, v)]}
                sb = {r["scene"]: r for r in groups[(b, v)]}
                common = sorted(set(sa) & set(sb))
                if not common:
                    continue
                row = {"effect": name, "n_views": v, "n_scenes": len(common)}
                for m in ["psnr", "ssim", "lpips", "ate", "rot_deg"]:
                    d = [sa[s][m] - sb[s][m] for s in common if sa[s].get(m) is not None and sb[s].get(m) is not None]
                    row[f"d_{m}"] = mean(d)
                effects.append(row)
    return list(means.values()), effects


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main(results_root):
    rows = load(results_root)
    means, effects = tables(rows)
    out = Path(results_root)
    write_csv(out / "all_runs.csv", rows, ["config", "scene", "n_views"] + METRICS + ["date"])
    write_csv(out / "means_by_config.csv", means)
    write_csv(out / "effects.csv", effects)
    print(f"{len(rows)} runs -> all_runs.csv, means_by_config.csv, effects.csv in {out}")
    for m in means:
        print(f"  {m['config']} {m['n_views']} photos ({m['n_scenes']} scenes): PSNR {m['psnr']:.2f}  "
              f"ATE {m['ate']:.3f}  rot {m['rot_deg']:.2f} deg")
    for e in effects:
        print(f"  {e['effect']} at {e['n_views']} photos ({e['n_scenes']} scenes): dPSNR {e['d_psnr']:+.2f}")
    return rows, means, effects


def selftest():
    with tempfile.TemporaryDirectory() as d:
        def put(c, s, v, psnr, ate=0.1, rot=1.0, pm="pose_error.py"):
            p = Path(d) / c / s / f"{v}_views"
            p.mkdir(parents=True)
            json.dump(dict(config=c, scene=s, n_views=v, psnr=psnr, ssim=0.3, lpips=0.5, ate=ate, rot_deg=rot,
                           rpe_deg=1, ate_init=ate * 2, rot_deg_init=rot * 2, total_seconds=600, peak_vram_mb=4000,
                           pose_metric=pm, date="x"), open(p / "result.json", "w"))
        for s, base in [("garden", 15.0), ("bonsai", 14.0)]:
            put("C1", s, 3, base + 2.0, ate=0.0, rot=0.0)
            put("C2", s, 3, base)
            put("C3", s, 3, base + 0.5)
            put("C4", s, 3, base - 1.0)
        put("C3_baseline", "garden", 3, 99.0)             # a real C3 run exists -> the baseline copy is ignored
        put("C3_baseline", "kitchen", 3, 13.0)            # no C3 run -> counts as C3
        put("C2", "kitchen", 3, 12.0)
        put("C3", "room", 3, 50.0, pm=None)               # old format, must be ignored
        put("C4", "stump", 3, 10.0)                       # only in C4: must not enter the C3-C4 effect
        rows, means, effects = main(d)
        assert len(rows) == 11, len(rows)
        assert not any(r["psnr"] == 99.0 for r in rows), "baseline copy must not replace a real C3 run"
        e = {x["effect"]: x for x in effects}
        r3 = e["RQ3 refinement (C3 - C2)"]
        assert r3["n_scenes"] == 3 and abs(r3["d_psnr"] - (0.5 + 0.5 + 1.0) / 3) < 1e-9, r3
        e = {x["effect"]: x for x in effects}
        assert abs(e["RQ1 poses (C1 - C2)"]["d_psnr"] - 2.0) < 1e-9
        assert abs(e["RQ2 3D points (C3 - C4)"]["d_psnr"] - 1.5) < 1e-9 and e["RQ2 3D points (C3 - C4)"]["n_scenes"] == 2
        assert (Path(d) / "effects.csv").exists()
    print("selftest OK - aggregate ignores old/baseline folders and compares only matching scenes")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        main(sys.argv[1])
