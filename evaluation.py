import time
import numpy as np


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def detection_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Compute TP/FP/FN, precision, recall, F1 using greedy IoU matching for one frame."""
    matches = []
    used_gt = set()
    for pi, pred in sorted(enumerate(predictions), key=lambda x: x[1].get("score", 1.0), reverse=True):
        best_j, best_iou = None, 0.0
        for gj, gt in enumerate(ground_truth):
            if gj in used_gt:
                continue
            ov = iou(pred["bbox"], gt["bbox"])
            if ov > best_iou:
                best_iou, best_j = ov, gj
        if best_j is not None and best_iou >= iou_threshold:
            used_gt.add(best_j)
            matches.append((pi, best_j, best_iou))

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "mean_iou": mean_iou}


def benchmark_detector(detector, images, warmup=2):
    for image in images[:warmup]:
        detector(image)
    times = []
    counts = []
    for image in images:
        t0 = time.perf_counter()
        detections = detector(image)
        times.append(time.perf_counter() - t0)
        counts.append(len(detections))
    latency_ms = 1000.0 * float(np.mean(times)) if times else 0.0
    fps = 1.0 / float(np.mean(times)) if times and np.mean(times) > 0 else 0.0
    return {"latency_ms": latency_ms, "fps": fps,
            "avg_detections": float(np.mean(counts)) if counts else 0.0}
