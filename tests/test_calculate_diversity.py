import math
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd


def random_seqs(rng, count=40, vocabulary=("DT", "NN", "VB", "JJ", "IN", ".")):
    seqs = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 12))] for _ in range(count)]
    # Duplicates and very short sequences exercise tied maxima and smoothing.
    return seqs + [seqs[0], ["NN"], ["DT", "NN"]]


class MetricTests(unittest.TestCase):
    def test_the_stage_1_metrics(self):
        self.assertEqual(cd.METRICS, (
            "lexical_hdd↑", "lexical_distinct_2↑",
            "semantic_vendi↑", "semantic_distsum_cosine↑", "semantic_cluster_inertia↑",
            "structural_self_bleu_4↓", "structural_self_bleu_5↓",
            "structural_distinct_5↑", "structural_distinct_6↑", "structural_pos_vendi↑",
            "structural_knn_cosine_1↑", "structural_knn_cosine_3↑"))
        seqs = random_seqs(random.Random(2))
        x = np.random.default_rng(1).normal(size=(len(seqs), 8))
        scores = {**cd.text_metrics(seqs, seqs), **cd.embedding_metrics(x, x, 0)}
        self.assertEqual(set(scores), set(cd.METRICS))
        self.assertEqual(scores["structural_distinct_6↑"], cd.distinct_n(seqs, 6))
        self.assertEqual(scores["structural_self_bleu_4↓"], cd.pos_self_bleu(seqs)[3])

    def test_pos_self_bleu_matches_nltk_for_each_order(self):
        seqs = random_seqs(random.Random(0))
        smoothing = SmoothingFunction().method1
        expected = [np.mean([
            sentence_bleu(seqs[:i] + seqs[i + 1:], seq, weights=(1 / k,) * k,
                          smoothing_function=smoothing)
            for i, seq in enumerate(seqs)
        ]) for k in range(1, 6)]
        np.testing.assert_allclose(cd.pos_self_bleu(seqs, 5), expected, rtol=1e-12)

    def test_hdd_matches_hypergeometric_formula(self):
        seqs = [["a", "b", "a", "c"] * 6, ["d", "a", "e"] * 9]
        counts = Counter(token for seq in seqs for token in seq)
        total = sum(counts.values())
        expected = sum(1 - math.comb(total - f, 42) / math.comb(total, 42)
                       for f in counts.values()) / 42
        self.assertAlmostEqual(cd.hdd(seqs), expected, places=12)
        self.assertTrue(math.isnan(cd.hdd([["a"] * 41])))

    def test_distinct_n_pools_examples(self):
        seqs = [["a", "b", "c"], ["a", "b", "d"], ["e"]]
        self.assertEqual(cd.distinct_n(seqs, 2), 3 / 4)
        # Sequences shorter than n contribute no n-grams to either count.
        seqs = [list("abcdef"), list("abcdefg"), ["e"]]
        self.assertEqual(cd.distinct_n(seqs, 5), 3 / 5)
        self.assertEqual(cd.distinct_n(seqs, 6), 2 / 3)
        self.assertTrue(math.isnan(cd.distinct_n([list("abcde")], 6)))

    def test_embedding_metrics_match_their_definitions(self):
        x = np.random.default_rng(0).normal(size=(30, 8))
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        pairs = [(i, j) for i in range(len(x)) for j in range(len(x)) if i != j]
        self.assertAlmostEqual(cd.distsum_cosine(x),
                               np.mean([1 - unit[i] @ unit[j] for i, j in pairs]), places=12)
        distances = 1 - unit @ unit.T
        np.fill_diagonal(distances, np.inf)
        for k in (1, 3):
            self.assertAlmostEqual(cd.knn_cosine_distance(x, k),
                                   np.sort(distances, axis=1)[:, :k].mean(), places=12)
        eigenvalues = np.linalg.eigvalsh(unit @ unit.T / len(x))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        self.assertAlmostEqual(cd.vendi_score(x),
                               np.exp(-np.sum(eigenvalues * np.log(eigenvalues))), places=8)
        self.assertAlmostEqual(cd.vendi_score(np.ones((5, 3))), 1.0, places=10)
        self.assertAlmostEqual(cd.vendi_score(np.eye(4)), 4.0, places=10)
        self.assertAlmostEqual(cd.cluster_inertia(x, 1, 0),
                               np.sum((x - x.mean(axis=0)) ** 2), places=8)

    def test_seed_means_average_each_group_over_its_seeds(self):
        by_seed = pd.DataFrame({"subcategory": ["A", "A", "B"], "abbr": ["a", "a", "b"],
                                "seed": [86, 43, 43], "count": [9, 9, 5],
                                "harmful_sample_sha256": ["x", "y", "z"],
                                "lexical_hdd↑": [0.2, 0.4, 0.5]})
        means = cd.seed_means(by_seed)
        self.assertEqual(means.columns.tolist(),
                         ["subcategory", "abbr", "count", "seeds", "lexical_hdd↑"])
        self.assertEqual(means["seeds"].tolist(), ["43 86", "43"])
        self.assertEqual(means["lexical_hdd↑"].tolist(), [0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
