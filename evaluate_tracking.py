from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from DetectPlayer import detect_players
from deep_detector import TorchvisionDetector, build_fasterrcnn, load_checkpoint
from soccernet import image_files, read_gt
from tracking import KalmanMultiObjectTracker


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou(a, b):
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _detector(args):
    if args.detector == "classical":
        return detect_players
    model = build_fasterrcnn(args.num_classes, pretrained_backbone=False)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    return TorchvisionDetector(model, score_threshold=args.score_threshold)


def _fallback_metrics(frames_data, iou_threshold):
    """Dependency-free tracking summary: MOTA-like counts + ID switches.

    motmetrics is preferred for standard MOTA/MOTP/IDF1. This fallback still makes
    the pipeline runnable without extra packages.
    """
    misses = fp = matches = ids = 0
    iou_values = []
    last_assignment = {}
    for gt_objs, tracks in frames_data:
        pairs = []
        for gi, g in enumerate(gt_objs):
            for ti, t in enumerate(tracks):
                ov = _iou(g.bbox, t.bbox)
                if ov >= iou_threshold:
                    pairs.append((-ov, gi, ti))
        pairs.sort()
        ug, ut = set(), set()
        for negov, gi, ti in pairs:
            if gi in ug or ti in ut:
                continue
            ug.add(gi); ut.add(ti)
            matches += 1
            iou_values.append(-negov)
            gid, tid = gt_objs[gi].track_id, tracks[ti].id
            if gid in last_assignment and last_assignment[gid] != tid:
                ids += 1
            last_assignment[gid] = tid
        misses += len(gt_objs) - len(ug)
        fp += len(tracks) - len(ut)
    gt_total = matches + misses
    mota = 1.0 - (misses + fp + ids) / gt_total if gt_total else 0.0
    return {
        "matches": matches, "misses": misses, "false_positives": fp,
        "id_switches": ids, "mota": mota,
        "motp_iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "note": "Install motmetrics for standard IDF1/IDP/IDR and MOTChallenge-compatible metrics."
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate tracking against SoccerNet/MOT ground truth")
    p.add_argument("sequence")
    p.add_argument("--detector", choices=["classical", "deep"], default="classical")
    p.add_argument("--checkpoint")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gt-class", type=int, action="append", dest="gt_classes")
    p.add_argument("--out", default="tracking_metrics.json")
    args = p.parse_args()
    if args.detector == "deep" and not args.checkpoint:
        p.error("--checkpoint is required with --detector deep")

    detector = _detector(args)
    tracker = KalmanMultiObjectTracker(max_distance=80, max_missed=8, min_hits=2)
    gt = read_gt(args.sequence)
    frames = image_files(args.sequence)
    if args.limit:
        frames = frames[:args.limit]

    frames_data = []
    for frame_num, path in enumerate(frames, start=1):
        image = cv2.imread(str(path))
        detections = [d for d in detector(image) if d.get("class_name", "player") == "player"]
        tracks = tracker.update(detections)
        gt_objs = gt.get(frame_num, [])
        if args.gt_classes:
            wanted = set(args.gt_classes)
            gt_objs = [x for x in gt_objs if x.class_id in wanted]
        frames_data.append((gt_objs, tracks))

    try:
        import motmetrics as mm
        acc = mm.MOTAccumulator(auto_id=True)
        for gt_objs, tracks in frames_data:
            gt_ids = [g.track_id for g in gt_objs]
            track_ids = [t.id for t in tracks]
            distances = np.full((len(gt_objs), len(tracks)), np.nan, dtype=float)
            for gi, g in enumerate(gt_objs):
                for ti, t in enumerate(tracks):
                    ov = _iou(g.bbox, t.bbox)
                    if ov >= args.iou:
                        distances[gi, ti] = 1.0 - ov
            acc.update(gt_ids, track_ids, distances)
        mh = mm.metrics.create()
        metrics = ["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches", "num_misses", "num_false_positives"]
        summary = mh.compute(acc, metrics=metrics, name="sequence")
        result = {k: float(v) if hasattr(v, "item") else v for k, v in summary.loc["sequence"].to_dict().items()}
        result["metric_backend"] = "motmetrics"
    except ImportError:
        result = _fallback_metrics(frames_data, args.iou)
        result["metric_backend"] = "built-in fallback"

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
