import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_safe
import helper


def guard_output(text):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def responses(labels, column="is_refusal"):
    return pd.DataFrame({
        "prompt": [f"prompt {i}" for i in range(len(labels))],
        "subcategory": ["category"] * len(labels),
        "generated_response": ["answer"] * len(labels),
        column: labels,
    })


class OverrefusalTests(unittest.TestCase):
    def test_xstest_excludes_unsafe_unknown_and_missing_prompts(self):
        source = pd.DataFrame({
            "prompt": ["safe prompt", "unsafe prompt", "unknown", None],
            "type": ["homonyms", "contrast_homonyms", "other", "homonyms"],
            "label": ["safe", "unsafe", None, "safe"],
        })
        with patch.object(helper, "load_dataset") as load:
            load.return_value.to_pandas.return_value = source
            result = helper.load_safety_dataset("XSTest", "test")
        load.assert_called_once_with("Paul/XSTest", split="train")
        self.assertEqual(result["prompt"].tolist(), ["safe prompt"])
        self.assertEqual(result["subcategory"].tolist(), ["homonyms"])
        self.assertEqual(result["prompt_harm_label"].tolist(), ["safe"])

    def test_or_bench_uses_only_hard_config(self):
        with patch.object(helper, "load_dataset") as load:
            load.return_value.to_pandas.return_value = pd.DataFrame({
                "prompt": ["safe prompt"], "category": ["privacy"],
            })
            result = helper.load_safety_dataset("or-bench-hard", "test")
        load.assert_called_once_with(
            "bench-llm/or-bench", "or-bench-hard-1k", split="train"
        )
        self.assertEqual(result["subcategory"].tolist(), ["privacy"])
        self.assertEqual(result["prompt_harm_label"].tolist(), ["safe"])
        with self.assertRaises(ValueError):
            helper.load_safety_dataset("or-bench-hard", "train")

    def test_refusal_and_harmfulness_are_independent(self):
        guard = Mock()
        guard.generate.return_value = [guard_output(text) for text in (
            "Harmful request: no\nResponse refusal: yes\nHarmful response: no",
            "Harmful request: no\nResponse refusal: NO\nHarmful response: yes",
            "Response refusal: unclear\nHarmful response: no",
            "Harmful response: no",
        )]
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=Mock())}):
            refusals = eval_safe.evaluate_safety(
                "wildguard", guard, ["prompt"] * 4, ["answer"] * 4, overrefusal=True
            )
            guard.generate.assert_called_once()
            harmful = eval_safe.evaluate_safety(
                "wildguard", guard, ["prompt"] * 4, ["answer"] * 4
            )
        self.assertEqual(refusals, ([1, 0, None, None], 2))
        self.assertEqual(harmful, ([0, 1, 0, 0], 0))

    def test_rates_exclude_missing_labels_and_macro_average(self):
        datasets = {
            "xstest": responses([1, 0, None]),
            "or-bench-hard": responses([1, 1, 1, 0]),
        }
        result = eval_safe.overall_results(datasets, overrefusal=True).set_index("dataset")
        self.assertEqual(result.loc["xstest", "total"], 2)
        self.assertEqual(result.loc["xstest", "na_responses"], 1)
        self.assertEqual(result["overrefusal_score_pct"].tolist(), ["50.0000", "75.0000", "62.5000"])
        breakdown = eval_safe.subcategory_results("xstest", datasets["xstest"], overrefusal=True)
        self.assertEqual(breakdown["count"].tolist(), [2, 2])
        self.assertEqual(breakdown["refusals"].tolist(), [1, 1])
        self.assertEqual(breakdown["overrefusal_score_pct"].tolist(), ["50.0000", "50.0000"])

    def test_no_valid_labels_is_not_zero_refusal(self):
        for labels in ([None, None], []):
            with self.subTest(labels=labels):
                dataset = responses(labels)
                result = eval_safe.overall_results({"xstest": dataset}, overrefusal=True)
                self.assertTrue(pd.isna(float(result.iloc[0]["overrefusal_score_pct"])))
                self.assertEqual(result.iloc[0]["na_responses"], len(labels))
                breakdown = eval_safe.subcategory_results("xstest", dataset, overrefusal=True)
                self.assertTrue(pd.isna(float(breakdown.iloc[0]["overrefusal_score_pct"])))

    def test_saved_safety_and_overrefusal_reports_are_separate(self):
        datasets = {
            "ailuminate": responses([0, 1], "is_unsafe"),
            "xstest": responses([1, 0, None]),
            "or-bench-hard": responses([0]),
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            eval_safe.save_results(datasets, directory, "model")
            root = Path(directory)
            safety = pd.read_csv(root / "model_overall.csv")
            self.assertEqual(safety["dataset"].tolist(), ["ailuminate", "Avg"])
            self.assertEqual(safety["harmful_score_pct"].tolist(), [50, 50])
            overrefusal = pd.read_csv(root / "model_overrefusal.csv")
            self.assertEqual(overrefusal["dataset"].tolist(), ["xstest", "or-bench-hard", "Avg"])
            self.assertEqual(overrefusal["overrefusal_score_pct"].tolist(), [50, 0, 25])
            saved = pd.read_csv(root / "model_xstest_responses.csv")
            self.assertEqual(saved["is_refusal"].iloc[:2].tolist(), [1, 0])
            self.assertTrue(pd.isna(saved["is_refusal"].iloc[2]))
            self.assertNotIn("is_unsafe", saved)

    def test_cli_defaults_include_both_benchmarks(self):
        with patch.object(sys, "argv", ["eval_safe.py"]):
            args = eval_safe.parse_args()
        self.assertEqual(args.datasets, helper.safety_test_datasets + helper.overrefusal_test_datasets)

    def test_overrefusal_only_run_reuses_models_and_supports_permit(self):
        guard = Mock()
        guard.generate.return_value = [guard_output(
            "Harmful request: no\nResponse refusal: yes\nHarmful response: no"
        )]
        llm = Mock(side_effect=[Mock(), guard])
        modules = {
            "torch": Mock(),
            "transformers": Mock(),
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=Mock()),
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "argv", [
                    "eval_safe.py", "--model", "qwen2.5-ins", "--permit",
                    "--datasets", "xstest", "or-bench-hard", "xstest",
                    "--output_dir", directory,
                ]),
                patch.object(eval_safe, "load_safety_dataset", side_effect=lambda *args: responses([0])) as load,
                patch.object(eval_safe, "generate_responses", return_value=["answer"]) as generate,
            ):
                eval_safe.main()
            self.assertEqual(load.call_count, 2)
            self.assertEqual(llm.call_count, 2)
            self.assertEqual(guard.generate.call_count, 2)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args.args[2], [eval_safe.PERMIT_PROMPT + "prompt 0"])
            root = Path(directory) / "qwen2.5-ins_permit"
            saved = pd.read_csv(root / "qwen2.5-ins_permit_overrefusal.csv")
            self.assertEqual(saved["overrefusal_score_pct"].tolist(), [100, 100, 100])
            self.assertFalse(list(root.glob("*_overall.csv")))


if __name__ == "__main__":
    unittest.main()
