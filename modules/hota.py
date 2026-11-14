"""HOTA (Higher Order Tracking Accuracy, Luiten et al. 2020), after the official TrackEval implementation.

HOTA = mean over IoU thresholds alpha in {0.05, ..., 0.95} of sqrt(DetA(alpha) * AssA(alpha)), where
DetA is detection accuracy and AssA is association accuracy. Similarity is box IoU, with no pre-threshold.
The matching per frame is a Hungarian assignment maximising (global id alignment score) x (box similarity),
so a pairing of identities that agree over the whole sequence is preferred to one that is merely closest
in a single frame. Identities never cross sequences: score each sequence on its own, then combine.

Usage:
    results = [hota_sequence(frames) for frames in sequences]    # one per segment, for one class
    summary = summarize(combine(results))                        # {"HOTA": ..., "DetA": ..., "AssA": ..., ...}

`frames` is a list, one entry per timestep, of (gt_ids, gt_boxes, tracker_ids, tracker_boxes) with
boxes as (n, 4) xyxy and ids as any hashable values.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.tracking import iou_matrix

ALPHAS = np.arange(0.05, 0.99, 0.05)            # 0.05 ... 0.95, as in TrackEval
EPS = np.finfo(float).eps
_INTEGER_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
_ARRAY_FIELDS = _INTEGER_FIELDS + ("AssA", "AssRe", "AssPr", "LocA")


def _empty() -> dict[str, np.ndarray]:
    return {field: np.zeros(len(ALPHAS)) for field in _ARRAY_FIELDS}


def _final_fields(res: dict) -> dict:
    tp, fn, fp = res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"]
    res["DetRe"] = tp / np.maximum(1, tp + fn)
    res["DetPr"] = tp / np.maximum(1, tp + fp)
    res["DetA"] = tp / np.maximum(1, tp + fn + fp)
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    return res


def hota_sequence(frames: list) -> dict[str, np.ndarray]:
    """Per-alpha HOTA statistics of one sequence (one class)."""
    gt_map: dict = {}
    tracker_map: dict = {}
    prepared = []
    for gt_ids, gt_boxes, tracker_ids, tracker_boxes in frames:
        g = np.array([gt_map.setdefault(i, len(gt_map)) for i in gt_ids], dtype=int)
        t = np.array([tracker_map.setdefault(i, len(tracker_map)) for i in tracker_ids], dtype=int)
        gb = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        tb = np.asarray(tracker_boxes, dtype=float).reshape(-1, 4)
        prepared.append((g, t, iou_matrix(gb, tb) if len(g) and len(t) else np.zeros((len(g), len(t)))))
    num_gt_dets = sum(len(g) for g, _, _ in prepared)
    num_tracker_dets = sum(len(t) for _, t, _ in prepared)

    res = _empty()
    if num_tracker_dets == 0:                    # TrackEval's shortcut for an empty tracker output
        res["HOTA_FN"] = np.full(len(ALPHAS), float(num_gt_dets))
        res["LocA"] = np.ones(len(ALPHAS))
        return res
    if num_gt_dets == 0:
        res["HOTA_FP"] = np.full(len(ALPHAS), float(num_tracker_dets))
        res["LocA"] = np.ones(len(ALPHAS))
        return res

    num_gt_ids, num_tracker_ids = len(gt_map), len(tracker_map)
    potential = np.zeros((num_gt_ids, num_tracker_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tracker_ids))

    # Pass 1: accumulate the global, similarity-weighted overlap between every pair of identities.
    for g, t, similarity in prepared:
        if len(g) and len(t):
            denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            mask = denom > EPS
            sim_iou[mask] = similarity[mask] / denom[mask]
            potential[g[:, np.newaxis], t[np.newaxis, :]] += sim_iou
        gt_id_count[g] += 1
        tracker_id_count[0, t] += 1
    global_alignment = potential / (gt_id_count + tracker_id_count - potential)
    matches_counts = [np.zeros_like(potential) for _ in ALPHAS]

    # Pass 2: per timestep, match identities, then count TP / FN / FP at every alpha.
    for g, t, similarity in prepared:
        if len(g) == 0:
            res["HOTA_FP"] += len(t)
            continue
        if len(t) == 0:
            res["HOTA_FN"] += len(g)
            continue
        score = global_alignment[g[:, np.newaxis], t[np.newaxis, :]] * similarity
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            keep = similarity[rows, cols] >= alpha - EPS
            kept_rows, kept_cols = rows[keep], cols[keep]
            matches = len(kept_rows)
            res["HOTA_TP"][a] += matches
            res["HOTA_FN"][a] += len(g) - matches
            res["HOTA_FP"][a] += len(t) - matches
            if matches:
                res["LocA"][a] += similarity[kept_rows, kept_cols].sum()
                matches_counts[a][g[kept_rows], t[kept_cols]] += 1

    for a in range(len(ALPHAS)):
        mc = matches_counts[a]
        tp = np.maximum(1, res["HOTA_TP"][a])
        res["AssA"][a] = np.sum(mc * (mc / np.maximum(1, gt_id_count + tracker_id_count - mc))) / tp
        res["AssRe"][a] = np.sum(mc * (mc / np.maximum(1, gt_id_count))) / tp
        res["AssPr"][a] = np.sum(mc * (mc / np.maximum(1, tracker_id_count))) / tp
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _final_fields(res)


def combine(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Combine per-sequence results the way TrackEval does: sum counts, weight association terms by TP."""
    res = {field: sum(r[field] for r in results) for field in _INTEGER_FIELDS}
    for field in ("AssA", "AssRe", "AssPr"):
        res[field] = sum(r[field] * r["HOTA_TP"] for r in results) / np.maximum(1.0, res["HOTA_TP"])
    loca = sum(r["LocA"] * r["HOTA_TP"] for r in results)
    res["LocA"] = np.maximum(1e-10, loca) / np.maximum(1e-10, res["HOTA_TP"])
    return _final_fields(res)


def summarize(res: dict[str, np.ndarray]) -> dict[str, float]:
    """Headline numbers: each per-alpha curve averaged over the alpha thresholds."""
    return {name: float(np.mean(res[name])) for name in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")}
