from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from DetectPlayer import detect_players
from deep_detector import TorchvisionDetector, build_fasterrcnn, load_checkpoint
from evaluation import detection_metrics
from soccernet import image_files, read_gt, gt_as_detections
from tracking import KalmanMultiObjectTracker


def _filter_gt(objects, class_ids=None):
    if not class_ids:
        return list(objects)
    wanted = set(class_ids)
    return [x for x in objects if x.class_id in wanted]


def _load_detector(args):
    if args.detector == "classical":
        return detect_players
    model = build_fasterrcnn(num_classes=args.num_classes, pretrained_backbone=False)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    return TorchvisionDetector(model, score_threshold=args.score_threshold)


def main():
    p = argparse.ArgumentParser(description="Evaluate classical or PyTorch detector on a SoccerNet/MOT sequence")
    p.add_argument("sequence", help="Sequence directory containing img1/ and gt/")
    p.add_argument("--detector", choices=["classical", "deep"], default="classical")
    p.add_argument("--checkpoint", help=".pth checkpoint for --detector deep")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N frames (0=all)")
    p.add_argument("--gt-class", type=int, action="append", dest="gt_classes", help="Only evaluate these GT class IDs; repeat flag")
    p.add_argument("--out", default="evaluation_result.json")
    args = p.parse_args()

    if args.detector == "deep" and not args.checkpoint:
        p.error("--checkpoint is required with --detector deep")

    frames = image_files(args.sequence)
    if args.limit:
        frames = frames[:args.limit]
    gt = read_gt(args.sequence)
    detector = _load_detector(args)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    ious = []
    times = []
    per_frame = []

    for idx, path in enumerate(frames, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        t0 = time.perf_counter()
        pred = [d for d in detector(image) if d.get("class_name", "player") == "player"]
        times.append(time.perf_counter() - t0)
        gt_objs = _filter_gt(gt.get(idx, []), args.gt_classes)
        gt_det = gt_as_detections(gt_objs)
        m = detection_metrics(pred, gt_det, iou_threshold=args.iou)
        for key in totals:
            totals[key] += m[key]
        if m["mean_iou"] > 0:
            ious.append(m["mean_iou"])
        per_frame.append({"frame": idx, **m, "predictions": len(pred), "ground_truth": len(gt_det)})

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_t = float(np.mean(times)) if times else 0.0
    summary = {
        "detector": args.detector,
        "frames": len(per_frame),
        "iou_threshold": args.iou,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "latency_ms": mean_t * 1000.0,
        "fps": (1.0 / mean_t) if mean_t > 0 else 0.0,
    }
    payload = {"summary": summary, "per_frame": per_frame}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
