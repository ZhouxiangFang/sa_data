"""plot_bars_overall.py reads eval.py's files for the runs and the instruct models."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval as evaluation
import helper
import plot_bars_overall as pbo


def write_results(root, name, harmful):
    safety_dir, if_dir, prefix = (Path(path) if i < 2 else path
                                  for i, path in enumerate(evaluation.result_dirs(root, name)))
    safety_dir.mkdir(parents=True, exist_ok=True)
    if_dir.mkdir(parents=True, exist_ok=True)
    tests = helper.safety_test_datasets + ["Avg"]
    pd.DataFrame({"dataset": tests, "harmful_score_pct": [harmful] * len(tests)}).to_csv(
        safety_dir / f"{prefix}_overall.csv", index=False)
    pd.DataFrame({"dataset": ["xstest", "or-bench-hard", "Avg"],
                  "overrefusal_score_pct": [10.0, 20.0, 15.0]}).to_csv(
        safety_dir / f"{prefix}_overrefusal.csv", index=False)
    pd.DataFrame({"benchmark": ["IFBench", "IFEval", "Average"],
                  **{score: [0.3, 0.6, 0.45] for score in ("prompt_strict", "instruction_strict",
                                                           "prompt_loose", "instruction_loose")}}
                 ).to_csv(if_dir / f"{prefix}_if_summary.csv", index=False)


class PlotTests(unittest.TestCase):
    def test_figures_for_trained_runs_and_the_instruct_comparison(self):
        models = ["m1_git20k", "m2_git20k"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in models:
                write_results(root, model, 40.0)
                write_results(root, model.replace("_git20k", "-ins"), 30.0)
                for abbr, harmful in (("random", 9.0), ("evenly", 8.0), ("toxic", 7.0)):
                    write_results(root, helper.checkpoint_name(model, 1800, abbr, 43), harmful)
            argv = ["plot_bars_overall.py", "--results_dir", directory, "--baseline_dir",
                    directory, "--models", *models]
            with mock.patch.object(sys, "argv", argv):
                pbo.main()
            figures = root / "figs" / "seed43"
            for name in ("average.png", "m1.png", "sorry-bench/average.png", "scores.csv"):
                self.assertTrue((figures / name).exists(), name)
            scores = pd.read_csv(figures / "scores.csv")
            row = scores.query("model == 'Average' and version == 'toxic'").iloc[0]
            self.assertEqual((row["harmful"], row["if"]), (7.0, 45.0))
            instruct = pd.read_csv(root / "figs" / "instruct_vs_git20k.csv")
            self.assertEqual(instruct["IFEval_prompt_loose"].tolist(), [60.0] * 6)
            self.assertTrue((root / "figs" / "instruct_vs_git20k.png").exists())


if __name__ == "__main__":
    unittest.main()
