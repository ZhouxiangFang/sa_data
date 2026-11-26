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
import eval as evaluation
import eval_if
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
            result = helper.load_test_set("xstest")
        load.assert_called_once_with("Paul/XSTest", split="train")
        self.assertEqual(result["prompt"].tolist(), ["safe prompt"])
        self.assertEqual(result["subcategory"].tolist(), ["homonyms"])
        self.assertEqual(result["prompt_harm_label"].tolist(), ["safe"])

    def test_or_bench_uses_only_hard_config(self):
        with patch.object(helper, "load_dataset") as load:
            load.return_value.to_pandas.return_value = pd.DataFrame({
                "prompt": ["safe prompt"], "category": ["privacy"],
            })
            result = helper.load_test_set("or-bench-hard")
        load.assert_called_once_with(
            "bench-llm/or-bench", "or-bench-hard-1k", split="train"
        )
        self.assertEqual(result["subcategory"].tolist(), ["privacy"])
        self.assertEqual(result["prompt_harm_label"].tolist(), ["safe"])
        with self.assertRaises(ValueError):
            helper.load_test_set("or-bench")

    def test_refusal_and_harmfulness_are_independent(self):
        guard = Mock()
        guard.generate.return_value = [guard_output(text) for text in (
            "Harmful request: no\nResponse refusal: yes\nHarmful response: no",
            "Harmful request: no\nResponse refusal: NO\nHarmful response: yes",
            "Response refusal: unclear\nHarmful response: no",
            "Harmful response: no",
        )]
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=Mock())}):
            labels = eval_safe.evaluate_safety(guard, ["prompt"] * 4, ["answer"] * 4)
        guard.generate.assert_called_once()
        self.assertEqual(labels, {"is_harmful_request": [0, 0, None, None],
                                  "is_refusal": [1, 0, None, None],
                                  "is_unsafe": [0, 1, 0, 0]})

    def test_rates_exclude_missing_labels_and_macro_average(self):
        datasets = {
            "xstest": responses([1, 0, None]),
            "or-bench-hard": responses([1, 1, 1, 0]),
        }
        result = pd.DataFrame(eval_safe.score_rows(datasets)).set_index("test_set")
        self.assertEqual(result["metric"].unique().tolist(), ["overrefusal"])
        self.assertEqual(result.loc["xstest", "n"], 2)
        self.assertEqual(result.loc["xstest", "na_responses"], 1)
        self.assertEqual(result["score"].tolist(), [50, 75, 62.5])
        self.assertEqual(result.loc["Avg", ["n", "count", "na_responses"]].tolist(), [6, 4, 1])
        breakdown = eval_safe.subcategory_results("xstest", datasets["xstest"])
        self.assertEqual(breakdown["subcategory"].tolist(), ["overall", "category"])
        self.assertEqual(breakdown["n"].tolist(), [2, 2])
        self.assertEqual(breakdown["count"].tolist(), [1, 1])
        self.assertEqual(breakdown["score"].tolist(), [50, 50])

    def test_no_valid_labels_is_not_zero_refusal(self):
        for labels in ([None, None], []):
            with self.subTest(labels=labels):
                dataset = responses(labels)
                rows = eval_safe.score_rows({"xstest": dataset})
                self.assertTrue(pd.isna(rows[0]["score"]) and pd.isna(rows[-1]["score"]))
                self.assertEqual(rows[0]["na_responses"], len(labels))
                breakdown = eval_safe.subcategory_results("xstest", dataset)
                self.assertTrue(pd.isna(breakdown.iloc[0]["score"]))

    def test_saved_safety_and_overrefusal_reports_are_separate(self):
        datasets = {
            "ailuminate": responses([0, 1], "is_unsafe"),
            "xstest": responses([1, 0, None]),
            "or-bench-hard": responses([0]),
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rows = pd.DataFrame(eval_safe.save_results(datasets, directory))
            root = Path(directory)
            safety = rows[rows["metric"] == "harmful"]
            self.assertEqual(safety["test_set"].tolist(), ["ailuminate", "Avg"])
            self.assertEqual(safety["score"].tolist(), [50, 50])
            overrefusal = rows[rows["metric"] == "overrefusal"]
            self.assertEqual(overrefusal["test_set"].tolist(), ["xstest", "or-bench-hard", "Avg"])
            self.assertEqual(overrefusal["score"].tolist(), [50, 0, 25])
            self.assertEqual(overrefusal["na_responses"].tolist(), [1, 0, 1])
            saved = pd.read_csv(root / "xstest_responses.csv")
            self.assertEqual(saved["is_refusal"].iloc[:2].tolist(), [1, 0])
            self.assertTrue(pd.isna(saved["is_refusal"].iloc[2]))
            self.assertNotIn("is_unsafe", saved)

    def test_prompt_types_follow_each_test_set_labels(self):
        cases = {
            "wildguardmix": ({"prompt_harm_label": ["harmful", "unharmful", None]},
                             ["harmful", "benign", "unknown"]),
            "aegis": ({"prompt_label": ["unsafe", "Safe"]}, ["harmful", "benign"]),
            "wildjailbreak": ({"subcategory": ["adversarial_harmful", "adversarial_benign"]},
                              ["harmful", "benign"]),
            "sorry-bench": ({"prompt": ["a", "b"]}, ["harmful", "harmful"]),
            "ailuminate": ({"prompt": ["a"]}, ["harmful"]),
            "xstest": ({"prompt": ["a"]}, ["benign"]),
        }
        for name, (columns, expected) in cases.items():
            with self.subTest(name=name):
                types = helper.prompt_types(pd.DataFrame(columns), name)
                self.assertEqual(types.tolist(), expected)

    def test_harmful_score_counts_only_harmful_prompts(self):
        dataset = responses([1, 0, 0, 1], "is_unsafe").assign(
            prompt_type=["harmful", "harmful", "benign", "unknown"],
            subcategory=["a", "b", "benign", "b"],
        )
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rows = eval_safe.save_results({"wildguardmix": dataset}, directory)
            root = Path(directory)
            self.assertEqual((rows[0]["test_set"], rows[0]["n"], rows[0]["count"]),
                             ("wildguardmix", 2, 1))
            breakdown = pd.read_csv(root / "wildguardmix_subcategories.csv")
            self.assertEqual(breakdown["subcategory"].tolist(), ["overall", "a", "b"])
            # Benign and unlabeled prompts are still saved with their type.
            saved = pd.read_csv(root / "wildguardmix_responses.csv")
            self.assertEqual(saved["prompt_type"].tolist(),
                             ["harmful", "harmful", "benign", "unknown"])
            self.assertEqual(saved["is_unsafe"].tolist(), [1, 0, 0, 1])


class EvalMainTests(unittest.TestCase):
    """eval.py: one model load serves IF and safety, then the guard judges."""

    def run_main(self, directory, *argv, judge_text="Response refusal: yes\nHarmful response: no",
                 batch_invariant_vllm=True):
        model = Mock()
        model.generate.side_effect = lambda prompts, params, use_tqdm=False: [
            guard_output("if reply") for _ in prompts]
        guard = Mock()
        guard.generate.side_effect = lambda prompts, params, use_tqdm=False: [
            guard_output(judge_text) for _ in prompts]
        llm = Mock(side_effect=[model, guard])
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (80, 80)  # the model's GPU memory is free again
        # vLLM 0.11 has no VLLM_BATCH_INVARIANT; vllm.envs then lacks the attribute.
        envs = SimpleNamespace(**({"VLLM_BATCH_INVARIANT": True} if batch_invariant_vllm else {}))
        modules = {
            "torch": torch,
            "transformers": Mock(),
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=Mock(), envs=envs, __version__="test"),
            "vllm.envs": envs,
        }
        example = {"prompt": "write two lines", "instruction_id_list": [], "kwargs": []}
        with (
            patch.dict(sys.modules, modules),
            patch.object(eval_if, "load_benchmarks",
                         return_value={"IFBench": [example], "IFEval": [example, example]}),
            patch.object(eval_if, "check_response", return_value=[True]),
            patch.object(evaluation, "check_tokenizer"),  # the engines are mocks
            patch.object(eval_safe, "load_test_set",
                         side_effect=lambda name: responses([0, 0])),
            patch.object(eval_safe, "generate_responses",
                         side_effect=lambda llm, tok, queries, **kw: ["answer"] * len(queries)
                         ) as generate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            evaluation.main(["--output_dir", directory, *argv])
        return llm, model, guard, generate

    def test_cli_defaults_include_every_test_set(self):
        args = evaluation.parse_args(["--model", "qwen2.5-ins"])
        self.assertEqual(args.datasets, helper.safety_test_datasets + helper.overrefusal_test_datasets)
        self.assertFalse(args.skip_if or args.skip_safety)

    def test_one_model_load_serves_if_and_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ckpts" / "olmo2_git20k_1800_wildguardmix_toxic_seed43"
            checkpoint.mkdir(parents=True)
            llm, model, guard, generate = self.run_main(
                directory, "--model", str(checkpoint), "--datasets", "ailuminate", "xstest")
            self.assertEqual(llm.call_count, 2)  # the model, then the guard
            for call in llm.call_args_list:  # both batch-invariant on a tested backend
                self.assertEqual(call.kwargs["attention_config"],
                                 {"backend": evaluation.ATTENTION_BACKEND})
                self.assertNotIn("tokenizer", call.kwargs)  # each model's own tokenizer
                self.assertEqual(call.kwargs["tokenizer_mode"], "hf")  # the HF one, as in training
            self.assertEqual(model.generate.call_count, 2)  # IFBench, IFEval
            model.reset_prefix_cache.assert_called_once()
            self.assertEqual(generate.call_count, 2)  # one call per test set
            self.assertEqual(guard.generate.call_count, 2)
            scores_csv, responses = evaluation.result_paths(directory, checkpoint.name)
            self.assertEqual(scores_csv, Path(directory) / "seed43/olmo2_git20k/toxic.csv")
            self.assertTrue(evaluation.evaluated(scores_csv))
            scores = pd.read_csv(scores_csv)
            loose = scores[scores["metric"] == "if_prompt_loose"]
            self.assertEqual(loose["test_set"].tolist(), ["IFBench", "IFEval", "Avg"])
            self.assertEqual(loose["n"].tolist(), [1, 2, 3])
            self.assertEqual(scores["metric"].unique().tolist(), evaluation.METRIC_ORDER)
            for name in ("if_responses.csv", "ailuminate_responses.csv",
                         "xstest_subcategories.csv"):
                self.assertTrue((responses / name).exists(), name)
            saved = pd.read_csv(responses / "xstest_responses.csv")
            self.assertEqual(saved.columns.tolist(),
                             ["prompt", "subcategory", "prompt_type", "generated_response",
                              *eval_safe.GUARD_LABELS])
            self.assertEqual(saved["is_refusal"].tolist(), [1, 1])
            self.assertTrue(saved["is_harmful_request"].isna().all())  # not in judge_text
            self.assertEqual(pd.read_csv(responses / "if_responses.csv").columns.tolist(),
                             ["benchmark", "prompt", "generated_response"])

    def test_a_partial_evaluation_keeps_the_other_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.save_scores(path, [
                    {"metric": "if_prompt_loose", "test_set": "Avg", "n": 3, "count": 1,
                     "score": 100 / 3},
                    {"metric": "harmful", "test_set": "Avg", "n": 4, "count": 1,
                     "na_responses": 0, "score": 25.0}])
                self.assertFalse(evaluation.evaluated(path))
                evaluation.save_scores(path, [
                    {"metric": "harmful", "test_set": "Avg", "n": 4, "count": 2,
                     "na_responses": 0, "score": 50.0},
                    {"metric": "overrefusal", "test_set": "Avg", "n": 2, "count": 0,
                     "na_responses": 1, "score": 0.0}])
            self.assertTrue(evaluation.evaluated(path))
            scores = evaluation.read_scores(path)
            self.assertEqual(scores.index.get_level_values("metric").tolist(),
                             ["harmful", "overrefusal", "if_prompt_loose"])
            self.assertEqual(scores["harmful", "Avg"], 50.0)
            self.assertAlmostEqual(scores["if_prompt_loose", "Avg"], 33.3333)
            self.assertIn("if_prompt_loose,Avg,3,1,,33.3333,", path.read_text())
            # A checkpoint with another fingerprint replaces every row; digits stay text.
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.save_scores(path, [{"metric": "harmful", "test_set": "Avg",
                                               "score": 10.0}], "012345678901")
            self.assertEqual(evaluation.read_table(path)["fingerprint"].tolist(),
                             ["012345678901"])
            self.assertFalse(evaluation.evaluated(path))

    def test_a_checkpoint_needs_its_settings_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "olmo2_git20k_1800_wildguardmix_toxic_seed43"
            checkpoint.mkdir()
            with self.assertRaisesRegex(SystemExit, "needs --output_dir"):
                evaluation.main(["--model", str(checkpoint)])

    def test_refuses_a_vllm_that_ignores_batch_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                self.run_main(directory, "--model", "qwen2.5-ins", batch_invariant_vllm=False)
            self.assertIn("ignores VLLM_BATCH_INVARIANT", str(raised.exception))

    def test_tokenizer_check_compares_with_the_models_sentencepiece_file(self):
        sentencepiece = SimpleNamespace(SentencePieceProcessor=lambda model_file: SimpleNamespace(
            encode=lambda text: [5, 6, 7]))
        tokenizer = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"sentencepiece": sentencepiece}):
            tokenizer.encode.return_value = [5, 6, 7]
            evaluation.check_tokenizer(tokenizer, directory)  # no tokenizer.model: not checked
            tokenizer.encode.assert_not_called()
            Path(directory, "tokenizer.model").touch()
            evaluation.check_tokenizer(tokenizer, directory)  # splits as SentencePiece: fine
            tokenizer.encode.return_value = [5, 67]
            with self.assertRaises(SystemExit):
                evaluation.check_tokenizer(tokenizer, directory)

    def test_if_scores_do_not_depend_on_the_random_state(self):
        import random

        from langdetect import DetectorFactory

        example = {"prompt": "p", "instruction_id_list": ["a", "b"], "kwargs": [{}, {}]}
        # A checker that draws from `random`, as IFEval's do when a kwarg is unset.
        draw = lambda example, response, loose=False: [random.random() < 0.5 for _ in range(2)]
        scores = []
        with patch.object(eval_if, "check_response", side_effect=draw):
            for state in (1, 2, 3):
                random.seed(state)
                DetectorFactory.seed = None
                scores.append(eval_if.evaluate("IFEval", [example] * 50, ["r"] * 50))
                self.assertEqual(DetectorFactory.seed, eval_if.SCORING_SEED)
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[0], scores[2])

    def test_result_paths(self):
        self.assertEqual(
            evaluation.result_paths("R", "olmo2_git20k_1800_wildguardmix_material_harm_seed86"),
            (Path("R/seed86/olmo2_git20k/material_harm.csv"),
             Path("R/seed86/olmo2_git20k/material_harm")))
        # Instruct and git20k models are not trained runs.
        self.assertEqual(evaluation.result_paths("R", "olmo2-ins"),
                         (Path("R/olmo2-ins.csv"), Path("R/olmo2-ins")))
        # In results/, the responses go to the same path under --responses_root (without it,
        # next to the scores) and figures to figs/.
        setting = helper.RESULTS_DIR / "selection" / "wildjailbreak"
        name = "qwen3_git20k_1800_wildjailbreak_semantic_vendi_qwen3-embedding-0.6b_seed43"
        self.assertEqual(
            evaluation.result_paths(setting, name, Path("S")),
            (setting / "seed43/qwen3_git20k/semantic_vendi_qwen3-embedding-0.6b.csv",
             Path("S/selection/wildjailbreak/seed43/qwen3_git20k")
             / "semantic_vendi_qwen3-embedding-0.6b"))
        self.assertEqual(evaluation.result_paths(setting, name)[1],
                         setting / "seed43/qwen3_git20k/semantic_vendi_qwen3-embedding-0.6b")
        self.assertEqual(evaluation.result_paths(helper.BASELINES_DIR, "olmo2-ins", "S")[1],
                         Path("S/baselines/olmo2-ins"))
        self.assertEqual(helper.figs_dir(helper.STAGE1_DIR),
                         helper.FIGS_DIR / "correlations/default")
        self.assertEqual(evaluation.resolve_model("llama3.1_git20k"),
                         ("FocusV857/llama3.1_git20k", "llama3.1_git20k"))


if __name__ == "__main__":
    unittest.main()
