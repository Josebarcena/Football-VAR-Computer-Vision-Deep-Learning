from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(script, args):
    cmd = [sys.executable, str(HERE / script)] + args
    subprocess.run(cmd, check=True, cwd=HERE)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Compare OpenCV vs deep learning on detection and tracking")
    p.add_argument("sequence")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--gt-class", type=int, action="append", dest="gt_classes")
    p.add_argument("--out", default="classical_vs_deep.json")
    args = p.parse_args()

    common = [args.sequence, "--iou", str(args.iou)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    for c in args.gt_classes or []:
        common += ["--gt-class", str(c)]

    paths = {
        "classical_detection": HERE / "_classical_detection.json",
        "deep_detection": HERE / "_deep_detection.json",
        "classical_tracking": HERE / "_classical_tracking.json",
        "deep_tracking": HERE / "_deep_tracking.json",
    }

    _run("evaluate_soccernet.py", common + ["--detector", "classical", "--out", str(paths["classical_detection"])])
    _run("evaluate_soccernet.py", common + ["--detector", "deep", "--checkpoint", args.checkpoint,
                                          "--num-classes", "2", "--score-threshold", str(args.score_threshold),
                                          "--out", str(paths["deep_detection"])])
    _run("evaluate_tracking.py", common + ["--detector", "classical", "--out", str(paths["classical_tracking"])])
    _run("evaluate_tracking.py", common + ["--detector", "deep", "--checkpoint", args.checkpoint,
                                         "--num-classes", "2", "--score-threshold", str(args.score_threshold),
                                         "--out", str(paths["deep_tracking"])])

    result = {
        "detection": {
            "classical": _json(paths["classical_detection"])["summary"],
            "deep": _json(paths["deep_detection"])["summary"],
        },
        "tracking": {
            "classical": _json(paths["classical_tracking"]),
            "deep": _json(paths["deep_tracking"]),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nDETECTION")
    print(f"{'metric':<18}{'classical':>14}{'deep':>14}")
    for m in ("precision", "recall", "f1", "mean_iou", "fps"):
        print(f"{m:<18}{result['detection']['classical'][m]:>14.4f}{result['detection']['deep'][m]:>14.4f}")

    print("\nTRACKING")
    c, d = result["tracking"]["classical"], result["tracking"]["deep"]
    metrics = [m for m in ("mota", "motp", "motp_iou", "idf1", "idp", "idr", "num_switches", "id_switches") if m in c or m in d]
    print(f"{'metric':<18}{'classical':>14}{'deep':>14}")
    for m in metrics:
        cv, dv = c.get(m, float('nan')), d.get(m, float('nan'))
        print(f"{m:<18}{cv:>14.4f}{dv:>14.4f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
