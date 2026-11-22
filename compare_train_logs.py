"""Compare per-step loss and grad norm of align.py training logs (e.g. two envs or optimizers).

python compare_train_logs.py NAME=LOG ...

The first log is the reference; every other log is compared with it step by step.
"""
import ast
import re
import sys

LINE = re.compile(r"\{'loss':.*?\}")


def read(path):
    steps = []
    with open(path, errors="replace") as handle:
        for line in handle:
            for match in LINE.findall(line):
                row = ast.literal_eval(match)
                steps.append({k: float(v) for k, v in row.items()})
    return steps


def main():
    runs = [arg.split("=", 1) for arg in sys.argv[1:]]
    logs = {name: read(path) for name, path in runs}
    ref_name = runs[0][0]
    ref = logs[ref_name]
    for name, steps in logs.items():
        last = steps[-1] if steps else {}
        print(f"{name}: {len(steps)} logged steps, last loss {last.get('loss')}, "
              f"last grad_norm {last.get('grad_norm')}, mean loss "
              f"{sum(s['loss'] for s in steps) / max(len(steps), 1):.5f}")
    for name, steps in logs.items():
        if name == ref_name:
            continue
        n = min(len(ref), len(steps))
        for key in ("loss", "grad_norm"):
            rel = [abs(a[key] - b[key]) / max(abs(a[key]), 1e-12) for a, b in zip(ref[:n], steps[:n])]
            absd = [abs(a[key] - b[key]) for a, b in zip(ref[:n], steps[:n])]
            print(f"{name} vs {ref_name} {key:9}: max rel diff {max(rel):.4%}, mean rel diff "
                  f"{sum(rel) / n:.4%}, max abs diff {max(absd):.5f} over {n} steps")


if __name__ == "__main__":
    main()
