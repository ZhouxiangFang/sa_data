import math
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd


def random_seqs(rng, count=40, vocabulary=('DT', 'NN', 'VB', 'JJ', 'IN', '.')):
    seqs = [[rng.choice(vocabulary) for _ in range(rng.randint(1, 12))] for _ in range(count)]
    # Duplicates and very short sequences exercise tied maxima and smoothing.
    return seqs + [seqs[0], ['NN'], ['DT', 'NN']]


class MetricTests(unittest.TestCase):
    def test_self_bleu_matches_nltk_for_each_order(self):
        seqs = random_seqs(random.Random(0))
        smoothing = SmoothingFunction().method1
        expected = [np.mean([
            sentence_bleu(seqs[:i] + seqs[i + 1:], seq, weights=(1 / k,) * k,
                          smoothing_function=smoothing)
            for i, seq in enumerate(seqs)
        ]) for k in range(1, 6)]
        np.testing.assert_allclose(cd.self_bleu(seqs, 5), expected, rtol=1e-12)

    def test_hdd_matches_hypergeometric_formula(self):
        seqs = [['a', 'b', 'a', 'c'] * 6, ['d', 'a', 'e'] * 9]
        counts = Counter(token for seq in seqs for token in seq)
        total = sum(counts.values())
        expected = sum(1 - math.comb(total - f, 42) / math.comb(total, 42)
                       for f in counts.values()) / 42
        self.assertAlmostEqual(cd.hdd(seqs), expected, places=12)
        self.assertTrue(math.isnan(cd.hdd([['a'] * 41])))

    def test_distinct_n_pools_examples(self):
        seqs = [['a', 'b', 'c'], ['a', 'b', 'd'], ['e']]
        self.assertEqual(cd.distinct_n(seqs, 1), 5 / 7)
        self.assertEqual(cd.distinct_n(seqs, 2), 3 / 4)

    def test_ngram_vendi_matches_dense_kernel(self):
        seqs = random_seqs(random.Random(1))
        kernels = []
        for n in cd.POS_NGRAM_ORDERS:
            grams = sorted({gram for seq in seqs for gram in nltk.ngrams(seq, n)})
            counts = np.array([[Counter(nltk.ngrams(seq, n))[gram] for gram in grams]
                               for seq in seqs], dtype=float)
            norms = np.linalg.norm(counts, axis=1, keepdims=True)
            counts = np.divide(counts, norms, out=np.zeros_like(counts), where=norms > 0)
            kernels.append(counts @ counts.T)
        eigenvalues = np.linalg.eigvalsh(np.mean(kernels, axis=0) / len(seqs))
        eigenvalues = eigenvalues[eigenvalues > 0]
        expected = np.exp(-np.sum(eigenvalues * np.log(eigenvalues)))
        self.assertAlmostEqual(cd.ngram_vendi_score(seqs), expected, places=10)
        self.assertAlmostEqual(cd.ngram_vendi_score([['DT', 'NN', 'VB', 'NN']] * 5), 1.0,
                               places=10)

    def test_embedding_metrics_match_pairwise_definitions(self):
        x = np.random.default_rng(0).normal(size=(30, 8))
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        pairs = [(i, j) for i in range(len(x)) for j in range(len(x)) if i != j]
        self.assertAlmostEqual(cd.distsum_cosine(x),
                               np.mean([1 - unit[i] @ unit[j] for i, j in pairs]), places=12)
        distances = 1 - unit @ unit.T
        np.fill_diagonal(distances, np.inf)
        for k in (1, 3):
            self.assertAlmostEqual(cd.knn_cosine_distance(x, k),
                                   np.sort(distances, axis=1)[:, k - 1].mean(), places=12)
        self.assertAlmostEqual(cd.radius(x), np.prod(x.std(axis=0)) ** (1 / 8), places=12)
        self.assertAlmostEqual(cd.cluster_inertia(x, 1, 0),
                               np.sum((x - x.mean(axis=0)) ** 2), places=8)
        self.assertAlmostEqual(cd.cluster_inertia(x, len(x), 0), 0.0, places=8)


if __name__ == '__main__':
    unittest.main()
