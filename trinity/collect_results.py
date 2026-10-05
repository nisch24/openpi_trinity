"""Gather every LIBERO evaluation log into one results table.

Reads logs/eval-*.out (written by trinity/eval.sbatch) and prints, per job: config, checkpoint,
suite, trials per task, episodes and success rate. Also writes logs/results.csv.
Run from the repo root on PACE:  python3 trinity/collect_results.py
"""
import csv
import pathlib
import re

HEADER = re.compile(r"Evaluating (\S+) @ (\S+) on (\S+) \((\d+) trials/task\)")
RATE = re.compile(r"Total success rate: ([0-9.]+)")
EPISODES = re.compile(r"Total episodes: (\d+)")

rows = []
for log in sorted(pathlib.Path("logs").glob("eval-*.out")):
    text = log.read_text(errors="replace")
    head = HEADER.search(text)
    if not head:
        continue  # not an evaluation log written by eval.sbatch
    config, ckpt, suite, trials = head.groups()
    rate = RATE.findall(text)  # the last match is the final rate for the whole suite
    episodes = EPISODES.findall(text)
    rows.append({
        "job": log.stem.removeprefix("eval-"),
        "config": config,
        # "published" for openpi's own checkpoint, otherwise run/step, e.g. "run1/29999"
        "checkpoint": "published" if ckpt.startswith("gs://") else "/".join(ckpt.rstrip("/").split("/")[-2:]),
        "suite": suite,
        "trials_per_task": trials,
        "episodes": episodes[-1] if episodes else "",
        "success_rate": f"{float(rate[-1]) * 100:.1f}" if rate else "unfinished",
    })

if not rows:
    print("No evaluation logs found in logs/.")
else:
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    with open("logs/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nWritten to logs/results.csv")
