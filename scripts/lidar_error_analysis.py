"""Explain why official Waymo 3-D AP is far below the BEV proxy for the LiDAR models.

Works on the dumps written by `evaluate.py --dump-detections` and needs only numpy, so it runs on
any machine (the official evaluator itself needs Linux and TensorFlow). It matches predictions to
ground truth per frame, measures box errors, recomputes rotated 3-D IoU with its own
implementation, and attributes the loss to height, heading, size or localisation with
counterfactuals (replace one box component by the ground truth and re-measure).

  python scripts/lidar_error_analysis.py runs/centerpoint-v1.2-s42 runs/pointpillars-v1.2-s42

Writes `lidar_error_analysis.json` into each run directory. The AP values here are a replica of
the official L1 matching rules (greedy, score ordered, difficulty from point counts), not the
official evaluator, so they are checked against `official_metrics.json` and may differ slightly.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CLASSES = ("vehicle", "pedestrian", "cyclist")
OFFICIAL_IOU = (0.7, 0.5, 0.5)         # Waymo 3-D IoU thresholds per class
GATE_M = (2.0, 1.0, 1.0)               # centre-distance gate for the error-distribution matching
MODEL_RANGE_M = 50.0                   # voxel range of both models (configs/*.yaml)
MIN_L1_POINTS = 5


# ---------------------------------------------------------------- geometry ---------------------------------------------

def corners(cx, cy, dx, dy, heading):
    """Counter-clockwise corners of a box whose length dx lies along `heading`."""
    c, s = math.cos(heading), math.sin(heading)
    return [(cx + c * x - s * y, cy + s * x + c * y)
            for x, y in ((dx / 2, dy / 2), (-dx / 2, dy / 2), (-dx / 2, -dy / 2), (dx / 2, -dy / 2))]


def _clip(subject, clip):
    """Sutherland-Hodgman: clip polygon `subject` by the convex counter-clockwise polygon `clip`."""
    out = subject
    for i in range(len(clip)):
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        points, out = out, []
        if not points:
            break
        sx, sy = points[-1]
        s_in = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax) >= 0
        for ex, ey in points:
            e_in = (bx - ax) * (ey - ay) - (by - ay) * (ex - ax) >= 0
            if e_in != s_in:
                d1x, d1y, d2x, d2y = ex - sx, ey - sy, bx - ax, by - ay
                denom = d1x * d2y - d1y * d2x
                t = ((ax - sx) * d2y - (ay - sy) * d2x) / denom if denom else 0.0
                out.append((sx + t * d1x, sy + t * d1y))
            if e_in:
                out.append((ex, ey))
            sx, sy, s_in = ex, ey, e_in
    return out


def _area(poly):
    return 0.5 * abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))) if len(poly) > 2 else 0.0


def bev_intersection(a, b):
    return _area(_clip(corners(a[0], a[1], a[3], a[4], a[6]), corners(b[0], b[1], b[3], b[4], b[6])))


def iou_bev_and_3d(a, b):
    """Rotated BEV IoU and rotated 3-D IoU of two boxes [cx cy cz dx dy dz heading]."""
    inter = bev_intersection(a, b)
    area_a, area_b = a[3] * a[4], b[3] * b[4]
    bev = inter / max(area_a + area_b - inter, 1e-9)
    z = max(min(a[2] + a[5] / 2, b[2] + b[5] / 2) - max(a[2] - a[5] / 2, b[2] - b[5] / 2), 0.0)
    vol = inter * z
    return bev, vol / max(area_a * a[5] + area_b * b[5] - vol, 1e-9)


def iou_aabb(a, b):
    """Axis-aligned BEV IoU on (cx, cy, dx, dy), heading ignored, as in the BEV proxy."""
    w = min(a[0] + a[3] / 2, b[0] + b[3] / 2) - max(a[0] - a[3] / 2, b[0] - b[3] / 2)
    h = min(a[1] + a[4] / 2, b[1] + b[4] / 2) - max(a[1] - a[4] / 2, b[1] - b[4] / 2)
    inter = max(w, 0.0) * max(h, 0.0)
    return inter / max(a[3] * a[4] + b[3] * b[4] - inter, 1e-9)


def self_test() -> None:
    """Hand-checked cases for the IoU implementation."""
    unit = [0, 0, 0, 1, 1, 1, 0.0]
    close = lambda x, y: abs(x - y) < 1e-6  # noqa: E731
    assert close(iou_bev_and_3d(unit, unit)[1], 1.0)
    assert close(iou_bev_and_3d(unit, [0.5, 0, 0, 1, 1, 1, 0.0])[0], 1 / 3)            # half shifted in x
    assert close(iou_bev_and_3d(unit, [0, 0, 0.5, 1, 1, 1, 0.0])[1], 1 / 3)            # half shifted in z
    assert close(iou_bev_and_3d(unit, [0, 0, 0, 1, 1, 1, math.pi / 4])[0], 1 / (2 / (2 * (math.sqrt(2) - 1)) - 1))
    assert close(iou_bev_and_3d([0, 0, 0, 2, 1, 1, 0.0], [0, 0, 0, 2, 1, 1, math.pi / 2])[0], 1 / 3)  # 90 degrees
    assert close(iou_bev_and_3d(unit, [5, 0, 0, 1, 1, 1, 0.0])[1], 0.0)                # disjoint
    assert close(iou_bev_and_3d(unit, [0, 0, 0, 0.5, 0.5, 0.5, 0.3])[1], 0.125)        # contained: volume ratio
    assert close(iou_bev_and_3d(unit, [0, 0, 0, 1, 1, 1, math.pi])[1], 1.0)            # heading flip, square footprint
    print("selftest ok", flush=True)


# ---------------------------------------------------------------- loading ----------------------------------------------

def load(run_dir: Path):
    d = np.load(run_dir / "detections.npz")
    gt, pred = d["gt"], d["pred"]
    n_points, n_top = gt[:, 9], gt[:, 10]
    difficulty = gt[:, 8].astype(int)
    level = np.where(difficulty == 0, np.where(n_points >= MIN_L1_POINTS, 1, 2), difficulty)   # as waymo_official_eval.py
    return gt, d["gt_frame"], pred, d["pred_frame"], level, n_top


def group(frames, n_frames):
    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], np.arange(n_frames + 1))
    return order, bounds


# ---------------------------------------------------------------- pair table -------------------------------------------

def build_pairs(gt, gt_frame, pred, pred_frame, n_frames):
    """All same-class (pred, gt) pairs whose circumscribed circles overlap, with their IoUs."""
    gorder, gb = group(gt_frame, n_frames)
    porder, pb = group(pred_frame, n_frames)
    rows = []
    for f in range(n_frames):
        gi, pi = gorder[gb[f]:gb[f + 1]], porder[pb[f]:pb[f + 1]]
        if len(gi) == 0 or len(pi) == 0:
            continue
        g, p = gt[gi], pred[pi]
        dist = np.hypot(p[:, None, 0] - g[None, :, 0], p[:, None, 1] - g[None, :, 1])
        reach = (np.hypot(p[:, 3], p[:, 4]) / 2)[:, None] + (np.hypot(g[:, 3], g[:, 4]) / 2)[None, :]
        same = p[:, 7][:, None] == g[:, 7][None, :]
        for a, b in zip(*np.nonzero((dist < reach) & same)):
            bev, i3 = iou_bev_and_3d(p[a, :7], g[b, :7])
            rows.append((pi[a], gi[b], dist[a, b], iou_aabb(p[a, :7], g[b, :7]), bev, i3))
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def greedy_match(pairs, score, criterion, higher_is_better, threshold, valid_gt=None):
    """Score-ordered greedy matching. Returns {pred: gt}. `criterion` is a pairs column index."""
    by_pred: dict[int, list] = {}
    for row in pairs:
        by_pred.setdefault(int(row[0]), []).append(row)
    used, matched = set(), {}
    for p in sorted(by_pred, key=lambda i: -score[i]):
        best = None
        for row in by_pred[p]:
            g, value = int(row[1]), row[criterion]
            ok = value >= threshold if higher_is_better else value <= threshold
            if not ok or g in used:
                continue
            if best is None or (value > best[1] if higher_is_better else value < best[1]):
                best = (g, value)
        if best is not None:
            used.add(best[0])
            matched[p] = best[0]
    return matched


def average_precision(tp, valid_gt, interpolate):
    """AP from score-ordered TP flags (1/0, entries to ignore already removed)."""
    if valid_gt == 0 or len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = cum_tp / valid_gt, cum_tp / (cum_tp + cum_fp)
    if not interpolate:
        return float(np.trapezoid(precision, recall))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([envelope[np.searchsorted(recall, r)] if r <= recall[-1] else 0.0 for r in np.linspace(0, 1, 101)]))


# ---------------------------------------------------------------- analysis ---------------------------------------------

def pct(x):
    return [round(float(v), 3) for v in np.percentile(x, [10, 50, 90])] if len(x) else None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def analyse(run_dir: Path) -> dict:
    gt, gt_frame, pred, pred_frame, level, n_top = load(run_dir)
    n_frames = int(max(gt_frame.max(), pred_frame.max())) + 1
    score = pred[:, 8]
    pairs = build_pairs(gt, gt_frame, pred, pred_frame, n_frames)
    l1 = level == 1
    result = {"run": run_dir.name, "frames": n_frames, "pairs": int(len(pairs))}

    # ---- GT population that the models can and cannot reach -------------------------------------------------------
    in_range = (np.abs(gt[:, 0]) <= MODEL_RANGE_M) & (np.abs(gt[:, 1]) <= MODEL_RANGE_M)
    top_ok = n_top >= MIN_L1_POINTS
    population = {}
    for c, name in enumerate(CLASSES):
        m = gt[:, 7] == c
        population[name] = {
            "gt_total": int(m.sum()), "official_L1": int((m & l1).sum()),
            "L1_beyond_50m": int((m & l1 & ~in_range).sum()),
            "L1_with_under_5_top_lidar_points": int((m & l1 & ~top_ok).sum()),
            "L1_reachable_in_range_and_top_points": int((m & l1 & in_range & top_ok).sum()),
            "proxy_gt_count_top_points_ge_5": int((m & top_ok).sum()),
        }
    result["gt_population"] = population

    # ---- replica of official L1 AP and of the proxy (validates the matching code) --------------------------------
    replica = {}
    for c, name in enumerate(CLASSES):
        pm = pred[:, 7] == c
        thr = OFFICIAL_IOU[c]
        gm = gt[:, 7] == c
        cls_pairs = pairs[np.isin(pairs[:, 0], np.nonzero(pm)[0]) & (pairs[:, 2] >= 0)]
        # official-like: match to every GT, ignore predictions matched to non-L1 GT, count only L1 GT
        match3d = greedy_match(cls_pairs, score, 5, True, thr)
        order = [i for i in np.nonzero(pm)[0][np.argsort(-score[pm])] if not (i in match3d and not l1[match3d[i]])]
        tp = [1 if i in match3d else 0 for i in order]
        official_like = average_precision(tp, int((gm & l1).sum()), True)
        # proxy-like: aabb IoU, GT with >=5 top points, no ignoring
        matchp = greedy_match(cls_pairs[np.isin(cls_pairs[:, 1], np.nonzero(gm & top_ok)[0])], score, 3, True, thr)
        orderp = np.nonzero(pm)[0][np.argsort(-score[pm])]
        proxy_like = average_precision([1 if i in matchp else 0 for i in orderp], int((gm & top_ok).sum()), False)
        replica[name] = {"official_like_ap_L1": round(official_like, 4), "proxy_like_ap": round(proxy_like, 4)}
    result["ap_replica"] = replica
    try:
        official = json.loads((run_dir / "official_metrics.json").read_text())["rows"]
        proxy = {c["class_name"]: c["ap_proxy"] for c in json.loads((run_dir / "evaluation.json").read_text())["per_class"]}
        for name in CLASSES:
            replica[name]["official_reported_L1"] = round(next(r["ap"] for r in official if r["shard"] == name.upper() and r["difficulty"] == "LEVEL_1"), 4)
            replica[name]["proxy_reported"] = round(proxy[name], 4)
    except (OSError, KeyError, StopIteration):
        pass

    # ---- per-class box errors on centre-distance matches against L1 GT -------------------------------------------
    result["classes"] = {}
    for c, name in enumerate(CLASSES):
        pm, gm = pred[:, 7] == c, gt[:, 7] == c
        sel = np.isin(pairs[:, 0], np.nonzero(pm)[0]) & np.isin(pairs[:, 1], np.nonzero(gm & l1)[0])
        cp = pairs[sel]
        matched = greedy_match(cp, score, 2, False, GATE_M[c])
        thr = OFFICIAL_IOU[c]
        n_l1 = int((gm & l1).sum())
        info = {"l1_gt": n_l1, "matched_within_gate_m": GATE_M[c], "matched": len(matched)}

        # recall of L1 GT at several criteria (best score-ordered greedy matching, any score >= 0.2)
        for label, col, t in (("bev_aabb_0.5", 3, 0.5), ("bev_aabb_0.7", 3, 0.7), ("bev_rotated_0.5", 4, 0.5),
                              ("bev_rotated_0.7", 4, 0.7), ("iou3d_0.5", 5, 0.5), ("iou3d_0.7", 5, 0.7)):
            info[f"recall_{label}"] = round(len(greedy_match(cp, score, col, True, t)) / max(n_l1, 1), 4)
        info["recall_official_threshold"] = info[f"recall_iou3d_{thr}"]

        if not matched:
            result["classes"][name] = info
            continue
        P = np.array([pred[p, :7] for p in matched])
        G = np.array([gt[g, :7] for g in matched.values()])
        # errors in the ground-truth box frame: along the box's heading and lateral to it
        d = P[:, :2] - G[:, :2]
        cos, sin = np.cos(G[:, 6]), np.sin(G[:, 6])
        along, lateral = d[:, 0] * cos + d[:, 1] * sin, -d[:, 0] * sin + d[:, 1] * cos
        herr = wrap(P[:, 6] - G[:, 6])
        folded = np.abs((herr + math.pi / 2) % math.pi - math.pi / 2)        # fold 180-degree flips
        info["centre_error_m_p10_p50_p90"] = {"x": pct(d[:, 0]), "y": pct(d[:, 1]), "z": pct(P[:, 2] - G[:, 2]),
                                              "along_heading": pct(along), "lateral": pct(lateral)}
        info["bottom_z_error_m"] = pct((P[:, 2] - P[:, 5] / 2) - (G[:, 2] - G[:, 5] / 2))
        info["size_ratio_pred_over_gt_p10_p50_p90"] = {k: pct(P[:, i] / G[:, i]) for k, i in (("dx", 3), ("dy", 4), ("dz", 5))}
        info["abs_heading_error_deg_p10_p50_p90"] = pct(np.degrees(np.abs(herr)))
        info["abs_heading_error_folded_deg_p10_p50_p90"] = pct(np.degrees(folded))
        info["heading_flip_share_over_90deg"] = round(float(np.mean(np.abs(herr) > math.pi / 2)), 4)
        info["heading_error_over_45deg_after_folding_share"] = round(float(np.mean(folded > math.pi / 4)), 4)

        # counterfactuals: replace one component of the prediction with ground truth and re-measure 3-D IoU
        def frac(variant):
            vals = np.array([iou_bev_and_3d(v, g)[1] for v, g in zip(variant, G)])
            return {"median_iou3d": round(float(np.median(vals)), 3), "share_ge_threshold": round(float(np.mean(vals >= thr)), 4)}

        def with_cols(cols, source=G):
            v = P.copy()
            v[:, cols] = source[:, cols]
            return v

        variants = {
            "as_predicted": P,
            "gt_z_and_dz": with_cols([2, 5]),
            "gt_heading": with_cols([6]),
            "gt_z_dz_and_heading": with_cols([2, 5, 6]),
            "gt_centre_xy": with_cols([0, 1]),
            "gt_size_dx_dy_dz": with_cols([3, 4, 5]),
            "gt_everything_but_xy": with_cols([2, 3, 4, 5, 6]),
        }
        info["counterfactual_iou3d"] = {k: frac(v) for k, v in variants.items()}

        # convention checks: would a systematic change in the prediction convention help?
        conv = {}
        for label, make in (("heading_plus_90deg", lambda v: np.c_[v[:, :6], v[:, 6] + math.pi / 2]),
                            ("heading_negated", lambda v: np.c_[v[:, :6], -v[:, 6]]),
                            ("heading_plus_180deg", lambda v: np.c_[v[:, :6], v[:, 6] + math.pi]),
                            ("swap_dx_dy", lambda v: np.c_[v[:, :3], v[:, 4], v[:, 3], v[:, 5:]]),
                            ("z_shifted_by_median_bottom_error", lambda v: np.c_[v[:, :2], v[:, 2] - np.median((P[:, 2] - P[:, 5] / 2) - (G[:, 2] - G[:, 5] / 2)), v[:, 3:]])):
            conv[label] = frac(make(P))["share_ge_threshold"]
        info["convention_check_share_ge_threshold"] = conv
        result["classes"][name] = info

    # ---- precision side at the official 3-D threshold ------------------------------------------------------------
    result["precision"] = {}
    for c, name in enumerate(CLASSES):
        pm, gm = pred[:, 7] == c, gt[:, 7] == c
        cp = pairs[np.isin(pairs[:, 0], np.nonzero(pm)[0])]
        match3d = greedy_match(cp, score, 5, True, OFFICIAL_IOU[c])
        loose = {int(r[0]) for r in cp if r[3] >= 0.3}                      # overlaps some GT in BEV
        idx = np.nonzero(pm & ~np.isin(np.arange(len(pred)), [i for i, g in match3d.items() if not l1[g]]))[0]
        bins = [(0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        rows = []
        for lo, hi in bins:
            b = idx[(score[idx] >= lo) & (score[idx] < hi)]
            tp = sum(1 for i in b if i in match3d)
            rows.append({"score": f"{lo}-{min(hi, 1.0)}", "predictions": int(len(b)), "tp_at_official_iou": tp,
                         "precision": round(tp / max(len(b), 1), 4)})
        fp = [i for i in idx if i not in match3d]
        result["precision"][name] = {
            "predictions": int(pm.sum()), "per_frame": round(float(pm.sum()) / n_frames, 2),
            "l1_gt_per_frame": round(float((gm & l1).sum()) / n_frames, 2),
            "false_positives": len(fp),
            "fp_overlapping_a_gt_in_bev_aabb_iou_0.3_share": round(float(np.mean([i in loose for i in fp])) if fp else 0.0, 4),
            "by_score": rows,
            "min_score_in_dump": round(float(score[pm].min()), 4),
            "share_of_predictions_below_0.3": round(float(np.mean(score[pm] < 0.3)), 4),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    self_test()
    for run_dir in args.run_dirs:
        result = analyse(run_dir)
        out = run_dir / "lidar_error_analysis.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}", flush=True)
        print(json.dumps({k: result[k] for k in ("gt_population", "ap_replica")}, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
