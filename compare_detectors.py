from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_eval(sequence, detector, outfile, extra):
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "evaluate_soccernet.py"), sequence, "--detector", detector, "--out", str(outfile)] + extra
    subprocess.run(cmd, check=True)
    return json.loads(Path(outfile).read_text(encoding="utf-8"))["summary"]


def main():
    p = argparse.ArgumentParser(description="Run the same SoccerNet detection evaluation for OpenCV and deep learning")
    p.add_argument("sequence")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--gt-class", type=int, action="append", dest="gt_classes")
    p.add_argument("--out", default="comparison.json")
    args = p.parse_args()

    common = ["--iou", str(args.iou)]
    if args.limit:
        common += ["--limit", str(args.limit)]
    for c in args.gt_classes or []:
        common += ["--gt-class", str(c)]

    classical = run_eval(args.sequence, "classical", "_eval_classical.json", common)
    deep_extra = common + ["--checkpoint", args.checkpoint, "--num-classes", "2", "--score-threshold", str(args.score_threshold)]
    deep = run_eval(args.sequence, "deep", "_eval_deep.json", deep_extra)
    result = {"classical": classical, "deep": deep}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    cols = ["precision", "recall", "f1", "mean_iou", "latency_ms", "fps"]
    print("\nmetric              classical          deep")
    print("-" * 48)
    for c in cols:
        print(f"{c:<18} {classical[c]:>12.4f} {deep[c]:>13.4f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
