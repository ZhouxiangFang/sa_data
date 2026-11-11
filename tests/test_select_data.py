import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calculate_diversity as cd
import select_data as sd

TAGS = ('DT', 'NN', 'VB', 'JJ', 'IN', '.')
WORDS = tuple('abcdefghijklmnop')


def random_features(seed, count=60):
    rng = random.Random(seed)
    tags = [[rng.choice(TAGS) for _ in range(rng.randint(1, 14))] for _ in range(count)]
    words = [[rng.choice(WORDS) for _ in range(rng.randint(0, 12))] for _ in range(count)]
    # Repeats and very short sequences exercise tied maxima, smoothing and zero eigenvalues.
    tags[5], tags[6], tags[7] = list(tags[0]), ['NN'], ['DT', 'NN']
    words[5] = list(words[0])
    return sd.Features([str(i) for i in range(count)], tags, words)


class FakeEmbedder:
    """Fixed random embeddings, correlated so that the metrics differ between rows."""
    model_id = 'fake'

    def __init__(self, count, seed, dimension=24):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(count, dimension)) + 2 * rng.normal(size=(1, dimension))
        base[9] = base[3]  # two identical embeddings
        self.semantic = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype(np.float32)
        other = rng.normal(size=(count, dimension)) + rng.normal(size=(1, dimension))
        self.structural = (other / np.linalg.norm(other, axis=1, keepdims=True)).astype(np.float32)

    def encode_texts(self, texts):
        return self.semantic

    def encode(self, pos_seqs):
        return self.structural


def brute_force_step(reference, selected, count, minimize):
    """Metric from scratch on the subset plus each remaining row; nan where undefined."""
    values = np.full(count, np.nan)
    for row in range(count):
        if row not in selected:
            values[row] = reference(selected + [row])
    best = np.nanmin(values) if minimize else np.nanmax(values)
    return values, best


class GreedyTests(unittest.TestCase):
    def check_metric(self, metric, seed=0, count=60, initial_size=12, size=24):
        features = random_features(seed, count)
        embedder = FakeEmbedder(count, seed)
        objective, reference = sd.build_objective(metric, features, embedder, size, 'cpu')
        initial = sorted(random.Random(seed).sample(range(count), initial_size))
        minimize = sd.minimized(metric)
        selected, scores = sd.greedy_select(objective, initial, size, minimize=minimize)
        self.assertEqual(selected[:initial_size], initial)
        self.assertEqual(len(set(selected)), size)
        for step, score in enumerate(scores):
            before = selected[:initial_size + step]
            values, best = brute_force_step(reference, before, count, minimize)
            row = selected[initial_size + step]
            # The pick's score is the metric from scratch, and no candidate beats it.
            self.assertTrue(math.isclose(score, values[row], rel_tol=1e-9, abs_tol=1e-12),
                            (metric, step, score, values[row]))
            self.assertTrue(math.isclose(values[row], best, rel_tol=1e-9, abs_tol=1e-12),
                            (metric, step, values[row], best))
            # Ties go to the lowest row index, up to rounding between the two computations.
            ties = np.flatnonzero(np.isclose(values, best, rtol=1e-9, atol=1e-12))
            self.assertIn(row, ties)

    def test_every_default_metric_matches_brute_force(self):
        for metric in sd.DEFAULT_METRICS:
            if metric == 'semantic_cluster_inertia':
                continue  # not a closed-form metric; see ClusterInertiaTests
            for seed in (0, 1):
                with self.subTest(metric=metric, seed=seed):
                    self.check_metric(metric, seed)

    def test_exact_ties_go_to_the_lowest_row(self):
        class Constant:
            def add(self, row):
                pass

            def scores(self):
                return torch.tensor([1.0, 2.0, 2.0, 2.0, 0.5])

        selected, scores = sd.greedy_select(Constant(), [2], 3)
        self.assertEqual(selected, [2, 1, 3])
        self.assertEqual(scores, [2.0, 2.0])


class ClusterInertiaTests(unittest.TestCase):
    def test_scores_match_warm_started_kmeans(self):
        from sklearn.cluster import KMeans

        for seed, clusters in ((0, 4), (1, 6)):
            x = FakeEmbedder(120, seed).semantic.astype(np.float64)
            objective = sd.ClusterInertia(x, clusters, seed, 'cpu', block=50)
            members = list(range(0, 80, 2))
            for row in members:
                objective.add(row)
            scores = objective.scores().numpy()
            start = objective.centroids.numpy()
            # The kept solution is at least as good as the metric's own fit.
            self.assertLessEqual(objective.value(),
                                 cd.cluster_inertia(x[members], clusters, seed) * (1 + 1e-12))
            for row in range(120):
                expected = KMeans(n_clusters=clusters, init=start, n_init=1, tol=0, max_iter=300,
                                  algorithm='lloyd').fit(x[members + [row]]).inertia_
                self.assertTrue(math.isclose(scores[row], expected, rel_tol=1e-9),
                                (seed, row, scores[row], expected))

    def test_greedy_scores_are_the_kept_solutions(self):
        x = FakeEmbedder(80, 2).semantic.astype(np.float64)
        objective = sd.ClusterInertia(x, 4, 2, 'cpu')
        selected, scores = sd.greedy_select(objective, list(range(10)), 20)
        self.assertEqual(len(set(selected)), 20)
        for step, score in enumerate(scores):
            fresh = cd.cluster_inertia(x[selected[:11 + step]], 4, 2)
            self.assertLessEqual(score, fresh * (1 + 1e-12))
            self.assertGreater(score, 0.9 * fresh)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.x = FakeEmbedder(90, 4).semantic
        unit = self.x.astype(np.float64)
        self.unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
        self.similarity = self.unit @ self.unit.T
        self.initial = [1, 8, 20, 33, 47, 60]

    def select(self, name, size=30, **options):
        selected, scores = sd.build_baseline(name, self.x, size, 'cpu', **options)(self.initial)
        self.assertEqual(selected[:len(self.initial)], self.initial)
        self.assertEqual(len(set(selected)), size)
        self.assertEqual(len(scores), size - len(self.initial))
        return selected, scores

    def test_kcenter_greedy_adds_the_farthest_row(self):
        selected, scores = self.select('kcenter_greedy')
        for step, score in enumerate(scores):
            before = selected[:len(self.initial) + step]
            nearest = (1 - self.similarity[:, before]).min(axis=1)
            nearest[before] = -1
            self.assertAlmostEqual(score, nearest.max(), places=12)
            self.assertAlmostEqual(nearest[selected[len(before)]], nearest.max(), places=12)

    def test_qdit_adds_the_largest_facility_location_gain(self):
        selected, scores = self.select('qdit')
        for step, score in enumerate(scores):
            before = selected[:len(self.initial) + step]
            values = np.array([self.similarity[:, before + [row]].max(axis=1).sum()
                               for row in range(len(self.x))])
            values[before] = -np.inf
            self.assertAlmostEqual(score, values.max(), places=9)
            self.assertAlmostEqual(values[selected[len(before)]], values.max(), places=9)

    def test_novelselect_matches_the_released_code(self):
        # distance_weighted and bi_density_weighted_distance of NovelSum's novelselect.py.
        def distance_weighted(matrix, power=1.0):
            _, sorted_indices = torch.sort(matrix, dim=1)
            steps = torch.arange(1, matrix.shape[1] + 1, dtype=matrix.dtype)
            weights = (1 / torch.pow(steps, power)).expand(matrix.shape[0], -1)
            weighted = torch.zeros_like(matrix)
            weighted.scatter_(1, sorted_indices, weights)
            return matrix * weighted

        unit = torch.from_numpy(self.unit)
        squared = torch.cdist(torch.from_numpy(self.x).double(), torch.from_numpy(self.x).double()) ** 2
        density = np.empty(len(self.x))
        for row in range(len(self.x)):
            # Row 9 repeats row 3: the index holds distinct vectors and drops the closest one.
            others = np.sort(np.delete(squared[row].numpy(), [3, 9] if row in (3, 9) else [row, 9]))
            density[row] = 1 / (others[:10].mean() + 1e-9) ** 0.5
        density = torch.from_numpy(density)
        selected, scores = self.select('novelselect')
        for step, score in enumerate(scores):
            before = selected[:len(self.initial) + step]
            distance = 1 - unit @ unit[before].T
            novelty = distance_weighted(
                distance * (density.unsqueeze(1) + density[before].unsqueeze(0))).mean(dim=1)
            novelty[before] = -torch.inf
            self.assertTrue(math.isclose(score, float(novelty.max()), rel_tol=1e-6),
                            (step, score, float(novelty.max())))
            self.assertEqual(selected[len(before)], int(novelty.argmax()))

    def test_farthest_takes_the_largest_total_distances(self):
        selected, scores = self.select('farthest')
        total = (1 - self.similarity).sum(axis=1)
        rest = [row for row in np.argsort(-total, kind='stable') if row not in self.initial]
        self.assertEqual(sorted(selected[len(self.initial):]), sorted(int(r) for r in rest[:24]))
        np.testing.assert_allclose(scores, total[selected[len(self.initial):]], rtol=1e-9)

    def test_repr_filter_keeps_rows_below_one_threshold(self):
        selected, scores = self.select('repr_filter', seed=5)
        closest = [self.similarity[row, selected[:position]].max()
                   for position, row in enumerate(selected) if position >= len(self.initial)]
        np.testing.assert_allclose(scores, closest, atol=1e-6)
        again, _ = self.select('repr_filter', seed=5)
        self.assertEqual(selected, again)
        other, _ = self.select('repr_filter', seed=6)
        self.assertNotEqual(selected, other)

    def test_kmeans_draws_evenly_from_clusters(self):
        from sklearn.cluster import KMeans

        selected, scores = self.select('kmeans', size=36, n_clusters=3, seed=7)
        labels = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(
            self.x.astype(np.float64))
        self.assertEqual(scores, [float(labels[row]) for row in selected[len(self.initial):]])
        held = np.bincount(labels[selected], minlength=3)
        start = np.bincount(labels[self.initial], minlength=3)
        sizes = np.bincount(labels, minlength=3)
        for cluster in range(3):
            # A cluster holds its share unless it is too small or started above it.
            self.assertGreaterEqual(held[cluster], min(12, sizes[cluster]))
            if start[cluster] <= 12 and sizes.min() >= 12:
                self.assertEqual(held[cluster], 12)


class BorderedVendiTests(unittest.TestCase):
    def scores(self, kernel, members, normalize):
        count = len(kernel)
        objective = sd.Vendi(lambda i: kernel[:, i], np.diag(kernel).copy(), len(members) + 1,
                             normalize)
        for row in members:
            objective.add(row)
        expected = []
        for row in range(count):
            rows = members + [row]
            eigenvalues = np.linalg.eigvalsh(kernel[np.ix_(rows, rows)] / len(rows))
            eigenvalues = eigenvalues[eigenvalues > 1e-13]
            if normalize:
                eigenvalues = eigenvalues / eigenvalues.sum()
            expected.append(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))
        return objective.scores(), np.asarray(expected)

    def test_matches_dense_eigenvalues(self):
        rng = np.random.default_rng(0)
        for count, dimension, members in ((80, 64, 30), (300, 16, 40), (200, 256, 150)):
            x = rng.normal(size=(count, dimension)) + rng.normal(size=(1, dimension))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x[1] = x[0]  # a repeated member: a zero eigenvalue
            kernel = x @ x.T
            for normalize in (True, False):
                got, expected = self.scores(kernel, list(range(members)), normalize)
                np.testing.assert_allclose(got, expected, rtol=1e-9)

    def test_orthogonal_and_repeated_candidates(self):
        # Members e1, e1, e2; candidates: a repeat of e1, the orthogonal e3, a mixture.
        x = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                      [1, 0, 0, 0], [0, 0, 1, 0], [0.6, 0, 0.8, 0]], dtype=float)
        got, expected = self.scores(x @ x.T, [0, 1, 2], True)
        np.testing.assert_allclose(got[3:], expected[3:], rtol=1e-9)


class NgramCountTests(unittest.TestCase):
    def test_counts_match_nltk(self):
        import nltk
        from collections import Counter

        seqs = random_features(3).tags
        for n in (1, 2, 3, 5):
            counts = sd.ngram_counts(seqs, n)
            for row, seq in enumerate(seqs):
                expected = sorted(Counter(nltk.ngrams(seq, n)).values())
                self.assertEqual(sorted(counts[row].data.tolist()), expected)
            distinct = len({gram for seq in seqs for gram in nltk.ngrams(seq, n)})
            self.assertEqual(int((np.asarray(counts.sum(axis=0)) > 0).sum()), distinct)


if __name__ == '__main__':
    unittest.main()
