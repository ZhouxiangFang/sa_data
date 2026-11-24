"""Stage 2 selection: incremental objectives equal the metrics, pools stay within one stratum."""

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd
import helper
import select_data as sd

DEVICE = "cpu"


class FakeFeatures:
    """Random POS tags and words (with repeated rows) and unit embeddings."""

    def __init__(self, count=60, seed=0):
        rng = random.Random(seed)
        tags = [[rng.choice(("DT", "NN", "VB", "JJ", "IN")) for _ in range(rng.randint(2, 12))]
                for _ in range(count - 3)]
        self.tags = tags + [tags[0], tags[1], ["NN"]]
        self.words = [[tag.lower() + str(rng.randint(0, 3)) for tag in seq] for seq in self.tags]
        x = np.random.default_rng(seed).normal(size=(count, 6))
        x[count - 3] = x[0]  # an identical row
        self.x = x / np.linalg.norm(x, axis=1, keepdims=True)
        self.counts = {}

    def ngrams(self, kind, n):
        return sd.ngram_counts(getattr(self, kind), n)

    def embeddings(self, kind):
        return self.x


class ObjectiveTests(unittest.TestCase):
    def test_every_objective_scores_the_metric_of_the_subset_plus_the_candidate(self):
        features = FakeFeatures()
        members, candidates = list(range(0, 40, 2)), np.array([1, 3, 57, 58, 59, 41])
        for metric in sd.METRICS:
            if metric == "semantic_cluster_inertia":
                continue  # a k-means search, checked below
            with self.subTest(metric=metric):
                objective, reference = sd.build_objective(metric, features, 600, 0, DEVICE)
                for member in members:
                    objective.add(member)
                got = np.asarray(sd.torch.as_tensor(objective.scores(candidates)).double())
                expected = [reference(members + [int(c)]) for c in candidates]
                np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    def test_cluster_inertia_never_scores_above_a_fresh_fit(self):
        features = FakeFeatures()
        objective, reference = sd.build_objective("semantic_cluster_inertia", features, 600, 0,
                                                  DEVICE)
        members = list(range(30))
        for member in members:
            objective.add(member)
        self.assertLessEqual(objective.value(), reference(members) + 1e-9)
        scores = objective.scores(np.arange(30, 40)).numpy()
        self.assertTrue(np.isfinite(scores).all())


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.strata = sd.Strata(["vanilla"] * 70 + ["adversarial"] * 30)
        self.x = FakeFeatures(100).x

    def test_quotas_follow_the_natural_shares(self):
        self.assertEqual(self.strata.quotas(10).tolist(), [7, 3])
        self.assertEqual(self.strata.quotas(15).tolist(), [11, 4])  # 10.5 / 4.5: first on ties
        initial = self.strata.initial(10, 43)
        self.assertEqual(np.bincount(self.strata.labels[initial]).tolist(), [7, 3])

    def test_each_stratum_is_selected_alone_from_a_fresh_pool_each_step(self):
        calls = []

        class Recorder:
            def __init__(self):
                self.members = []

            def add(self, i):
                self.members.append(i)

            def scores(self, rows):
                calls.append((np.asarray(rows), list(self.members)))
                return np.asarray(rows, dtype=float)  # prefers the highest row

        build = lambda metric, features, size, seed, device: (Recorder(), lambda rows: rows[-1])
        initial = self.strata.initial(10, 43)
        with mock.patch.object(sd, "build_objective", build):
            selected, scores, orders = sd.select("lexical_hdd", None, initial, 40, self.strata,
                                                 43, 5, DEVICE)
        self.assertEqual(len(set(selected)), 40)
        labels = self.strata.labels[selected]
        self.assertEqual(np.bincount(labels).tolist(), [28, 12])
        self.assertTrue((np.diff(labels) >= 0).all())  # stratum by stratum
        self.assertEqual(orders[:7] + orders[28:31], [0] * 10)
        self.assertEqual(orders[7:10], [1, 2, 3])
        for rows, members in calls:
            stratum = set(self.strata.labels[rows]) | set(self.strata.labels[members])
            self.assertEqual(len(stratum), 1)  # the pool and the members: one stratum only
            self.assertLessEqual(len(rows), 5)
            self.assertFalse(set(members) & set(rows.tolist()))  # never a selected row
        self.assertGreater(len({tuple(rows) for rows, _ in calls}), 20)  # redrawn every step

    def test_mixed_ablation_scores_both_strata_but_draws_each_pool_from_one(self):
        calls = []

        class Recorder:
            def __init__(self):
                self.members = []

            def add(self, i):
                self.members.append(i)

            def scores(self, rows):
                calls.append((np.asarray(rows), list(self.members)))
                return np.asarray(rows, dtype=float)

        build = lambda metric, features, size, seed, device: (Recorder(), lambda rows: rows[-1])
        initial = self.strata.initial(10, 43)
        with mock.patch.object(sd, "build_objective", build):
            selected, _, orders = sd.select("lexical_hdd", None, initial, 40, self.strata, 43, 5,
                                            DEVICE, mixed=True)
        self.assertEqual(np.bincount(self.strata.labels[selected]).tolist(), [28, 12])
        self.assertEqual(orders, [0] * 10 + list(range(1, 31)))
        for rows, members in calls:
            self.assertEqual(len(set(self.strata.labels[rows])), 1)  # each pool: one stratum
            self.assertEqual(len(set(self.strata.labels[members])), 2)  # scored on both

    def test_baselines_fill_each_stratum_from_its_own_rows(self):
        features = mock.Mock(embeddings=lambda kind: self.x)
        initial = self.strata.initial(12, 43)
        for name in sd.BASELINES:
            with self.subTest(name=name):
                selected, scores, orders = sd.select(name, features, initial, 30, self.strata,
                                                     43, None, DEVICE)
                self.assertEqual(sorted(row for row, order in zip(selected, orders) if not order),
                                 sorted(initial))
                self.assertEqual(len(set(selected)), 30)
                self.assertEqual(np.bincount(self.strata.labels[selected]).tolist(), [21, 9])
                self.assertEqual(len(scores), 30)


class PoolTests(unittest.TestCase):
    def test_test_prompts_and_long_prompts_leave_and_repeats_stay(self):
        rows = pd.DataFrame({"prompt": ["a", "a", "Test  Prompt", "b" * 50, "c"],
                             "response": ["r", "r", "r", "r", "r"],
                             "stratum": ["vanilla", "vanilla", "vanilla", "adversarial",
                                         "adversarial"]})

        class Tokenizer:
            def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
                return [conversation[0]["content"] for conversation in conversations]

            def __call__(self, texts, **kwargs):
                return {"length": [len(text) for text in texts]}

        tests = pd.DataFrame({"prompt": ["test prompt"]})
        with mock.patch.dict(sd.LOADERS, {"alert_dpo": lambda: rows}), \
                mock.patch.object(sd, "load_test_set", lambda name: tests):
            pool = sd.load_pool("alert_dpo", "safe", Tokenizer(), 10)
            self.assertEqual(pool["prompt"].tolist(), ["a", "a", "c"])
            with self.assertRaises(SystemExit):
                sd.load_pool("alert_dpo", "refusal", Tokenizer(), 10)

    def test_training_reads_the_selected_subset(self):
        class Tokenizer:
            def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
                return [conversation[0]["content"] for conversation in conversations]

            def __call__(self, texts, **kwargs):
                return {"length": [len(text) for text in texts]}

        benign = pd.DataFrame({"prompt": [f"b{i}" for i in range(9)], "response": ["ok"] * 9})
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(helper, "DATA_DIR", Path(directory)):
            path = helper.selection_csv(43, "nemotron", "semantic_vendi", 3)
            self.assertEqual(path, Path(directory) / "selected/seed43/nemotron/semantic_vendi_3.csv")
            path.parent.mkdir(parents=True)
            pd.DataFrame({"row": [5, 1, 1], "prompt": ["x", "None", "None"],
                          "response": ["no"] * 3}).to_csv(path, index=False)
            self.assertEqual(helper.selection_methods(43, "nemotron", 3), ["semantic_vendi"])
            mixture, info = helper.training_examples(
                "semantic_vendi", 43, Tokenizer(), 9, selected="nemotron", samples_dir=None,
                benign_pool=benign, general_pool=benign.assign(prompt=lambda d: "g" + d.prompt))
            self.assertEqual(info["files"]["harmful"], path)
            self.assertEqual(sorted(mixture["prompt"].tolist()).count("None"), 2)
            self.assertIn("x", set(mixture["prompt"]))
            self.assertEqual(len(mixture), 9)
        self.assertEqual(helper.checkpoint_name("m", 1800, "qdit", 86, "alert_dpo"),
                         "m_1800_alert_dpo_qdit_seed86")


if __name__ == "__main__":
    unittest.main()
