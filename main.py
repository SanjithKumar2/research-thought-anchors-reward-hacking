# main.py — top-level entry point
# Usage:
#   python main.py --sanity          # 10 rollouts on task_01, read output by hand
#   python main.py --phase 1         # full rollout collection (all tasks)
#   python main.py --phase 2         # hack detection summary
#   python main.py --phase 3         # thought anchor resampling
#   python main.py --phase 4         # analysis + figures

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="MATS Thought Anchors Experiment")
    parser.add_argument("--sanity",  action="store_true",
                        help="Sanity check: 10 rollouts on task_01 only")
    parser.add_argument("--phase",   type=int, choices=[1,2,3,4],
                        help="Run a specific phase")
    parser.add_argument("--task",    default=None,
                        help="Restrict to a specific task id")
    parser.add_argument("--n",       type=int, default=None,
                        help="Override number of rollouts")
    args = parser.parse_args()

    if args.sanity:
        print("Running sanity check (10 rollouts, task_01)...")
        cmd = [sys.executable, "rollout.py", "--sanity"]
        subprocess.run(cmd)

    elif args.phase == 1:
        print("Phase 1: Rollout collection")
        cmd = [sys.executable, "rollout.py"]
        if args.task: cmd += ["--task", args.task]
        if args.n:    cmd += ["--n", str(args.n)]
        subprocess.run(cmd)

    elif args.phase == 2:
        print("Phase 2: Hack detection summary")
        from pathlib import Path
        import json
        from config import RESULTS_DIR
        results_dir = Path(RESULTS_DIR)
        total = hacks = 0
        for f in results_dir.glob("*_rollouts.jsonl"):
            for line in f.read_text().strip().split('\n'):
                if line:
                    r = json.loads(line)
                    total += 1
                    if r.get("is_hack"):
                        hacks += 1
        print(f"\nTotal rollouts: {total}")
        print(f"Hacking:        {hacks} ({hacks/total:.0%})" if total else "No rollouts found.")

    elif args.phase == 3:
        print("Phase 3: Thought anchor resampling")
        cmd = [sys.executable, "resample.py"]
        if args.task: cmd += ["--task", args.task]
        subprocess.run(cmd)

    elif args.phase == 4:
        print("Phase 4: Analysis")
        cmd = [sys.executable, "analyse.py"]
        subprocess.run(cmd)

    else:
        parser.print_help()
        print("\nStart with: python main.py --sanity")


if __name__ == "__main__":
    main()
