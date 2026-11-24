"""plot_bars_overall.py reads eval.py's scores CSVs of the runs and the instruct models."""

import contextlib
import io
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
    rows = [{"metric": "harmful", "test_set": test, "score": harmful}
            for test in helper.safety_test_datasets + ["Avg"]]
    rows += [{"metric": "overrefusal", "test_set": test, "score": score}
             for test, score in (("xstest", 10.0), ("or-bench-hard", 20.0), ("Avg", 15.0))]
    rows += [{"metric": f"if_{score}", "test_set": test, "score": value}
             for score in ("prompt_strict", "instruction_strict", "prompt_loose",
                           "instruction_loose")
             for test, value in (("IFBench", 30.0), ("IFEval", 60.0), ("Avg", 45.0))]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation.save_scores(evaluation.result_paths(root, name)[0], rows)


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
