"""Training and diversity draw the same examples; correlations pair each run with its sample."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd
import correlate_diversity as corr
import helper

STATS = {name: {"abbr": name} for name in ("a", "b", "c", "benign")}


class FakeTokenizer:
    """Formats a prompt as itself and counts characters as tokens."""

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        return [conversation[0]["content"] for conversation in conversations]

    def __call__(self, texts, add_special_tokens, return_length, truncation):
        return {"length": [len(text) for text in texts]}


def wildguardmix_rows():
    """a: 10 prompts with a refusal and a harmless compliance each; b: 12 refusals; c: 3
    refusals; one a+b refusal; plus rows no stage may use (harmful response, benign prompt,
    empty response)."""
    rows = []
    for i in range(10):
        rows += [(f"a{i} please", f"a{i} refuses", "harmful", "unharmful", "refusal", ["a"]),
                 (f"a{i} please", f"a{i} complies", "harmful", "unharmful", "compliance", ["a"])]
    rows += [(f"b{i} please", f"b{i} refuses", "harmful", "unharmful", "refusal", ["b"])
             for i in range(12)]
    rows += [(f"c{i} please", f"c{i} refuses", "harmful", "unharmful", "refusal", ["c"])
             for i in range(3)]
    rows += [("ab please", "ab refuses", "harmful", "unharmful", "refusal", ["a", "b"]),
             ("a harms", "here is how", "harmful", "harmful", "compliance", ["a"]),
             ("hello", "hi", "unharmful", "unharmful", "compliance", ["benign"]),
             ("b empty", " ", "harmful", "unharmful", "refusal", ["b"])]
    return pd.DataFrame(rows, columns=["prompt", "response", "prompt_harm_label",
                                       "response_harm_label", "response_refusal_label",
                                       "sub_abbrs"])


def pairs(prefix, count):
    return pd.DataFrame({"prompt": [f"{prefix} {i}" for i in range(count)],
                         "response": ["ok"] * count})


class PoolTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(helper, "subcategory_stats", lambda dataset: STATS)
        patch.start()
        self.addCleanup(patch.stop)
        self.df = wildguardmix_rows()

    def test_safe_responses_keep_compliances_and_the_refusal_filter_drops_them(self):
        self.assertEqual(helper.group_sizes(self.df), {"a": 21, "b": 13, "c": 3})
        self.assertEqual(helper.group_sizes(self.df, "refusal"), {"a": 11, "b": 13, "c": 3})
        self.assertEqual(helper.training_groups(self.df, 12), ["a", "b"])
        self.assertEqual(helper.training_groups(self.df, 12, "refusal"), ["b"])
        refusals = helper.harmful_pool(self.df, "a", 6, "refusal")
        self.assertTrue(refusals["response"].str.endswith("refuses").all())
        # A baseline pools the training groups, a multi-label row once.
        self.assertEqual(len(helper.harmful_pool(self.df, "random", 6)), 21 + 13 - 1)
        with self.assertRaises(ValueError):
            helper.pool_tag("dedup")

    def test_evenly_gives_each_training_group_the_same_share(self):
        pool = helper.harmful_pool(self.df, "evenly", 10)
        drawn, _ = helper.sample_evenly(pool, 10, ["a", "b"], "test", 43, FakeTokenizer(), 4096)
        self.assertEqual(Counter(drawn["group"]), {"a": 5, "b": 5})
        self.assertEqual(len(set(drawn["prompt"] + drawn["response"])), 10)
        sample, _, _ = helper.harmful_sample(self.df, "evenly", 10, 43, FakeTokenizer(),
                                             samples_dir=None)
        self.assertTrue(sample.equals(drawn[["prompt", "response"]]))

    def test_training_and_diversity_use_the_same_sample_files(self):
        benign, general = pairs("benign", 20), pairs("general", 20)
        with tempfile.TemporaryDirectory() as directory:
            samples = Path(directory)
            for flt, folder in (("safe", "wildguardmix"), ("refusal", "wildguardmix_refusal")):
                for seed in (43, 86):
                    # Diversity draws and saves first; training then reads the same file.
                    drawn, path, _ = helper.harmful_sample(
                        self.df, "random", 4, seed, FakeTokenizer(), flt, samples_dir=samples)
                    self.assertEqual(path, samples / f"seed{seed}" / folder / "random_4.jsonl")
                    mixture, info = helper.training_examples(
                        "random", seed, FakeTokenizer(), 12, flt, samples_dir=samples,
                        train_df=self.df.iloc[::-1], benign_pool=benign, general_pool=general)
                    self.assertEqual(info["files"]["harmful"], path)
                    harmful = mixture[mixture["prompt"].str.endswith("please")]
                    self.assertEqual(sorted(harmful["prompt"]), sorted(drawn["prompt"]))
                    self.assertEqual(len(mixture), 12)
                    # Without files, the same seed draws the same examples.
                    fresh, _ = helper.training_examples(
                        "random", seed, FakeTokenizer(), 12, flt, samples_dir=None,
                        train_df=self.df, benign_pool=benign, general_pool=general)
                    self.assertTrue(fresh.equals(mixture))
            self.assertTrue((samples / "seed86" / "general" / "git_data_20k_4.jsonl").exists())
            self.assertTrue((samples / "seed86" / "wildjailbreak" / "vanilla_benign_4.jsonl").exists())
            first = helper.read_sample(samples / "seed43" / "wildguardmix" / "random_4.jsonl")
            second = helper.read_sample(samples / "seed86" / "wildguardmix" / "random_4.jsonl")
            self.assertNotEqual(first["prompt"].tolist(), second["prompt"].tolist())

    def test_a_sample_file_must_match_the_pool_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = Path(directory)
            helper.harmful_sample(self.df, "b", 4, 43, FakeTokenizer(), samples_dir=samples)
            with self.assertRaisesRegex(ValueError, "not in the current"):
                helper.harmful_sample(self.df.iloc[:20], "b", 4, 43, FakeTokenizer(),
                                      samples_dir=samples)

            class LongTokenizer(FakeTokenizer):
                def __call__(self, texts, **kwargs):
                    return {"length": [5000] * len(texts)}

            with self.assertRaisesRegex(ValueError, "tokens with this tokenizer"):
                helper.harmful_sample(self.df, "b", 4, 43, LongTokenizer(), samples_dir=samples)

    def test_names(self):
        self.assertEqual(helper.checkpoint_name("FocusV857/llama3.1_git20k", 1800, "toxic", 43),
                         "llama3.1_git20k_1800_wildguardmix_toxic_seed43")
        self.assertEqual(helper.mixture_counts(1800), (600, 600, 600))
        self.assertEqual(helper.mixture_counts(1500), (500, 500, 500))
        self.assertEqual(cd.diversity_csv("D", "bge-m3", 600, 86),
                         Path("D/seed86/bge-m3/csv/wildguardmix_diversity_600.csv"))


class CorrelationTests(unittest.TestCase):
    def diversity(self):
        rows = []
        for abbr, base in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
            for seed, offset in ((43, 0.0), (86, 0.5)):
                rows.append({"abbr": abbr, "seed": seed, **dict.fromkeys(cd.METRICS, 0.0),
                             "lexical_hdd↑": base + offset,
                             "structural_self_bleu_4↓": base + offset})
        return pd.DataFrame(rows)

    def test_runs_pair_with_their_own_seed_and_the_average_needs_every_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model, shift in (("m1_git20k", 0.0), ("m2_git20k", 10.0)):
                for abbr, score in (("a", 30.0), ("b", 20.0), ("c", 10.0)):
                    for seed in (43, 86):
                        if model == "m2_git20k" and abbr == "c" and seed == 86:
                            continue  # not evaluated
                        run = root / f"seed{seed}" / f"{model}_1800_wildguardmix" / f"wildguardmix_{abbr}"
                        run.mkdir(parents=True)
                        pd.DataFrame({"dataset": ["aegis", "Avg"],
                                      "harmful_score_pct": [0.0, score + shift + (seed == 86) * 2]}
                                     ).to_csv(run / f"wildguardmix_{abbr}_overall.csv", index=False)
            runs = corr.load_runs(root, 1800, ["m1_git20k", "m2_git20k"], ["a", "b", "c"],
                                  [43, 86])
        self.assertEqual(len(runs), 11)
        by_seed, points = corr.pair(runs, self.diversity())
        point = points.query("model == 'm1' and abbr == 'a'").iloc[0]
        self.assertEqual((point["harmful"], point["lexical_hdd↑"], point["n_seeds"]),
                         (31.0, 1.25, 2))
        # m2/c has seed 43 only, so the Average of c is seed 43's.
        average = points.query("model == 'Average' and abbr == 'c'").iloc[0]
        self.assertEqual((average["harmful"], average["lexical_hdd↑"], average["n_seeds"]),
                         (15.0, 3.0, 1))
        table = pd.DataFrame(corr.correlation_rows(points, "mean"))
        coefficient = table.set_index(["model", "metric", "method"])["coefficient"]
        # Higher HD-D (more diverse), less harmful: negative. Higher POS Self-BLEU (less
        # diverse), less harmful: positive, as the metric is negated.
        self.assertAlmostEqual(coefficient["m1", "lexical_hdd↑", "spearman"], -1.0)
        self.assertAlmostEqual(coefficient["m1", "structural_self_bleu_4↓", "spearman"], 1.0)
        self.assertTrue(np.isnan(coefficient["m1", "semantic_vendi↑", "pearson"]))  # constant
        seeds = pd.DataFrame(corr.correlation_rows(by_seed[by_seed["seed"] == 86], "86"))
        self.assertEqual(seeds.query("model == 'Average'")["n_groups"].iloc[0], 2)


if __name__ == "__main__":
    unittest.main()
