"""Selection of a diverse training subset: exact greedy selection under each diversity
metric, and the selection strategies NovelSum (Yang et al., 2025) compares, as baselines.

For each dataset the candidate pool is every harmful prompt with a safe response,
from every subcategory (helper.selection_pool), without the prompts align.py would
skip (at least --max_length tokens). Every row is a candidate, so a prompt with
several safe responses (WildGuardMix: a refusal and a harmless compliance; Aegis: the
original model answer and NVIDIA's added refusals, often repeated verbatim) can be
picked once per response. The pool options are
align.py's: --response_label_filter refusal (wildguardmix only) keeps only refusals, and
--dedup keeps the first row of each prompt; with either, results go to
data/seed<seed>/<dataset>_<tag>/ (e.g. wildguardmix_refusal) and training reads them
with the same options (chili/run.sh --selected --response_label_filter refusal).

Strata (--stratify adversarial, the default): where the pool has an adversarial
flag (wildguardmix), vanilla and adversarial prompts keep their pool shares in
every subset: both the initial examples and the final subset take
round(n * stratum share) rows from each stratum (largest remainders). A selection
step only considers candidates from strata that are not yet full, so a metric is
still computed on the whole subset. Pools without the flag form one stratum.

Selection starts from --init_size examples (default 50) drawn uniformly within each
stratum with --seed (default 43); every metric, embedding model and baseline starts
from the same examples. Each step then adds the candidate whose addition gives the best
value of the metric on the whole subset (the largest value, or for POS Self-BLEU the
smallest), until the subset has --size examples (default 600). Every candidate
is scored exactly at every step, with the metric as calculate_diversity.py
defines it; ties go to the earliest pool row.

Metrics are named as the columns of the diversity CSVs, without the arrow. The
default is the 12 the diversity figures show (plot_bars_diversity.FIGURE_METRICS):
  lexical_hdd  lexical_distinct_2  lexical_distinct_3
  semantic_vendi  semantic_distsum_cosine  semantic_cluster_inertia
  structural_self_bleu_4  structural_self_bleu_5  structural_distinct_1_4
  structural_pos_vendi  structural_knn_cosine_1  structural_knn_cosine_3
--metrics also takes lexical_distinct_<n>, semantic_radius, structural_self_bleu_<n>,
structural_ngram_vendi and structural_knn_cosine_<k>. The embedding metrics use
--embedding_models (default qwen3-embedding-0.6b), on the prompts (semantic_*) or
on the POS strings (structural_pos_vendi, structural_knn_cosine_*).
semantic_cluster_inertia is the result of a k-means fit (--n_clusters, default 12,
as calculate_diversity.py), so it has no closed form. Each step keeps the best
k-means solution found for the current subset; a candidate's score is the inertia
k-means (Lloyd's iterations) converges to on the subset plus the candidate,
started from that solution. All candidates start from the same centroids, so
their scores differ only through the candidate. The solution kept for the next
step is the better of the picked candidate's and a fresh fit as the metric runs it
(10 k-means++ starts, seeded with --seed), and its inertia is the `score`; it can
be slightly below the value calculate_diversity.py reports for the same subset,
which keeps only the fresh fit.

Baselines (--baselines, default all) select with a strategy of their own instead
of a metric, from the same pool and initial examples, to the same size and stratum
quotas. The NovelSum strategies run on the prompt embeddings (semantic_<name>) of
every embedding model; distances are cosine. (structural_pos_<name> runs one on
the POS string embeddings instead, only when named in full.)
  kcenter_greedy  adds the row farthest from its nearest selected row (Sener &
                  Savarese, 2018). score: that distance.
  qdit            adds the row that most raises Facility Location, the sum over the
                  pool of each row's largest similarity to a selected row (Bukharin
                  et al., 2024, diversity term only). score: Facility Location.
  novelselect     adds the row with the largest novelty, as NovelSum's released
                  novelselect.py: distances to the selected rows, each times the sum
                  of the two rows' densities, sorted ascending and weighted 1 / rank,
                  then averaged. A row's density is (mean squared L2 distance to its 10
                  nearest distinct pool rows) ** -0.5. score: the novelty.
  farthest        ranks the pool once by total distance to all other rows and takes
                  the most distant rows. score: that total.
  repr_filter     visits the pool in random order and keeps a row if its similarity
                  to every kept row of its stratum is below a threshold (Liu et al.,
                  2023). Each stratum's threshold is the lowest one that still fills
                  its quota (found by bisection), since a fixed value does not transfer
                  between embedding models. score: the row's largest similarity when kept.
  kmeans          clusters each stratum with k-means (--n_clusters) and draws the
                  same number of rows from each cluster at random; the initial
                  examples count toward their clusters. score: the row's cluster.
and two that need no embedding (named random and evenly, as the training baselines,
but drawn from this pool and these initial examples):
  random          draws the rest of each stratum's quota uniformly. score: empty.
  evenly          fills each stratum's quota across its subcategories as evenly as
                  possible: each step draws a random row from the subcategory with the
                  fewest rows so far (the smaller subcategory on ties), so subcategories
                  that run out pass their share on. An initial example counts toward
                  the least-filled of its subcategories. score: empty.
kcenter_greedy, qdit and novelselect are deterministic given the initial examples;
repr_filter, kmeans, random and evenly also draw from --seed.

Results go to data/seed<seed>/<dataset>[_<tag>]/<metric>_<size>.csv, or
<metric>_<embedding model>_<size>.csv for the metrics and baselines that use an
embedding model (semantic_*, structural_pos_*, structural_knn_cosine_*). A CSV
holds the pool rows of the subset in selection order, with pool_index (the row's
index in the loaded dataset), stratum (vanilla / adversarial, or all), order (0
for the initial examples, then 1, 2, ...) and score (for a metric, its value on
the subset once that row was added; for a baseline, see above, empty for the
initial examples). Existing CSVs are kept unless --overwrite is given, so an
interrupted run resumes. A CSV without the stratum column of this run (made before
stratified selection, or with another --stratify) stops the run until it is
moved away, since training lists every CSV in the folder.

How each step stays exact and fast (all candidates are scored at once):
  distsum, radius   running sums of the embeddings
  cluster inertia   a candidate first moves one centroid, so only the distances to
                    centroids that moved are recomputed in each Lloyd iteration
  KNN distance      each member's k nearest distances, updated by the candidate
  distinct-n, HD-D  sparse n-gram counts against the subset's counts
  POS Self-BLEU     a candidate changes a member's clipped matches only for the
                    n-grams the member has more often than any other member; these
                    are summed for all (candidate, member) pairs by one sparse product
  Vendi Score       the subset's kernel is diagonalized once per step; a candidate
                    then borders a diagonal matrix, whose eigenvalues are the roots
                    of a secular equation, found for all candidates in parallel

Examples:
    python select_data.py                       # three datasets, 12 metrics + 8 baselines
    python select_data.py --datasets aegis --metrics semantic_vendi lexical_hdd
    python select_data.py --embedding_models bge-m3 --size 300 --seed 44
    python select_data.py --metrics --baselines qdit random     # only two baselines
    python select_data.py --response_label_filter refusal              # wildguardmix refusals only
"""

import argparse
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import get_num_threads, njit, prange, set_num_threads
from scipy import sparse
from sklearn import preprocessing
from sklearn.cluster import KMeans

import calculate_diversity as cd
from helper import (
    DATA_DIR,
    DEFAULT_MAX_LENGTH,
    DEFAULT_POOL,
    DEFAULT_SEED,
    add_pool_arguments,
    formatted_prompt_lengths,
    resolve_pool_options,
    selection_pool,
)

DATASETS = cd.SUPPORTED['train']
EMBEDDING_MODELS = {cd.embedding_name(model_id): model_id for model_id in cd.EMBEDDING_MODEL_IDS}
DEFAULT_EMBEDDING_MODELS = ('qwen3-embedding-0.6b',)
# Every metric calculate_diversity.py reports (with its default orders and k).
ALL_METRICS = (
    'lexical_hdd',
    *(f'lexical_distinct_{n}' for n in cd.LEXICAL_DISTINCT_ORDERS),
    'semantic_vendi', 'semantic_distsum_cosine', 'semantic_cluster_inertia', 'semantic_radius',
    *(f'structural_self_bleu_{n}' for n in range(1, 6)),
    'structural_distinct_1_4', 'structural_ngram_vendi', 'structural_pos_vendi',
    *(f'structural_knn_cosine_{k}' for k in cd.KNN_KS),
)
# The metrics the diversity figures show (plot_bars_diversity.FIGURE_METRICS; a test
# keeps the two lists equal without importing the plotting code here).
DEFAULT_METRICS = (
    'lexical_hdd', 'lexical_distinct_2', 'lexical_distinct_3',
    'semantic_vendi', 'semantic_distsum_cosine', 'semantic_cluster_inertia',
    'structural_self_bleu_4', 'structural_self_bleu_5', 'structural_distinct_1_4',
    'structural_pos_vendi', 'structural_knn_cosine_1', 'structural_knn_cosine_3',
)
# NovelSum's strategies, run on embeddings, and the two pool baselines, which need none.
EMBEDDING_BASELINES = ('novelselect', 'kcenter_greedy', 'repr_filter', 'qdit', 'kmeans',
                       'farthest')
POOL_BASELINES = ('random', 'evenly')
BASELINES = EMBEDDING_BASELINES + POOL_BASELINES
BASELINE_KINDS = {'semantic_': 'semantic', 'structural_pos_': 'structural'}
METRIC_PATTERN = re.compile(
    r'lexical_hdd|lexical_distinct_([1-9])|semantic_(?:vendi|distsum_cosine|cluster_inertia|radius)'
    r'|structural_self_bleu_([1-9])|structural_distinct_1_4|structural_ngram_vendi'
    r'|structural_pos_vendi|structural_knn_cosine_([1-9]\d*)')


def needs_embedding(metric):
    return metric.startswith(('semantic_', 'structural_pos_', 'structural_knn_cosine_'))


def baseline_parts(job):
    """(kind, baseline name) of a baseline job, or None for a metric. The kind is
    'semantic' or 'structural' (the embeddings it runs on), or 'pool' for random/evenly."""
    if job in POOL_BASELINES:
        return 'pool', job
    for prefix, kind in BASELINE_KINDS.items():
        if job.startswith(prefix) and job[len(prefix):] in EMBEDDING_BASELINES:
            return kind, job[len(prefix):]
    return None


def minimized(metric):
    """POS Self-BLEU is the only metric where lower values mean more diversity."""
    return 'self_bleu' in metric


def metric_number(metric):
    return int(metric.rsplit('_', 1)[1])


def output_path(output_dir, seed, dataset, metric, embedding, size, options=DEFAULT_POOL):
    """<output_dir>/seed<seed>/<dataset>[_<options tag>]/<metric>[_<embedding>]_<size>.csv,
    as helper.selection_csv names it for training."""
    name = f'{metric}_{embedding}_{size}.csv' if embedding else f'{metric}_{size}.csv'
    return Path(output_dir) / f'seed{seed}' / options.folder(dataset) / name


class Strata:
    """Pool rows split into strata whose shares every subset keeps (vanilla and
    adversarial prompts): a subset of n rows holds quotas(n) rows of each stratum."""

    def __init__(self, labels, names):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.names = list(names)
        self.sizes = np.bincount(self.labels, minlength=len(self.names))

    @classmethod
    def single(cls, count, name='all'):
        return cls(np.zeros(count, dtype=np.int64), [name])

    def __len__(self):
        return len(self.names)

    def rows(self, stratum):
        return np.flatnonzero(self.labels == stratum)

    def quotas(self, total):
        """Rows per stratum in a subset of `total`: the pool shares, rounded by largest
        remainders (ties to the first stratum)."""
        exact = self.sizes * total / self.sizes.sum()
        quotas = np.floor(exact).astype(np.int64)
        for stratum in np.argsort(-(exact - quotas), kind='stable')[:total - quotas.sum()]:
            quotas[stratum] += 1
        if (quotas > self.sizes).any():
            raise ValueError(f'A subset of {total} needs {quotas.tolist()} rows from strata '
                             f'{self.names} of sizes {self.sizes.tolist()}')
        return quotas

    def room(self, selected, total):
        """Rows each stratum still needs for a subset of `total` holding `selected`."""
        room = self.quotas(total) - np.bincount(self.labels[list(selected)],
                                                minlength=len(self.names))
        if (room < 0).any():
            raise ValueError(f'The rows {list(selected)[:5]}... exceed the stratum quotas '
                             f'{self.quotas(total).tolist()} of a subset of {total}')
        return room

    def initial(self, size, seed):
        """`size` rows drawn uniformly within each stratum, in pool order."""
        rng = np.random.default_rng(seed)
        quotas = self.quotas(size)
        return sorted(int(row) for stratum in range(len(self))
                      for row in rng.choice(self.rows(stratum), quotas[stratum], replace=False))

    def counts(self, rows):
        return dict(zip(self.names, np.bincount(self.labels[list(rows)],
                                                minlength=len(self.names)).tolist()))


# ---------------------------------------------------------------- n-gram counts

def ngram_counts(seqs, n):
    """CSR matrix (sequence x distinct n-gram) of n-gram counts."""
    lengths = np.fromiter(map(len, seqs), np.int64, len(seqs))
    vocabulary = {}
    tokens = np.fromiter((vocabulary.setdefault(token, len(vocabulary))
                          for seq in seqs for token in seq), np.int64, int(lengths.sum()))
    grams = np.maximum(lengths - n + 1, 0)
    rows = np.repeat(np.arange(len(seqs)), grams)
    starts = np.cumsum(lengths) - lengths
    # Position of each n-gram's first token in `tokens`.
    position = np.repeat(starts - (np.cumsum(grams) - grams), grams) + np.arange(int(grams.sum()))
    ids = tokens[position]
    for offset in range(1, n):
        _, ids = np.unique(ids * len(vocabulary) + tokens[position + offset], return_inverse=True)
    width = int(ids.max()) + 1 if len(ids) else 1
    counts = sparse.coo_matrix((np.ones(len(ids), np.int64), (rows, ids)),
                               shape=(len(seqs), width)).tocsr()
    counts.sort_indices()
    return counts


def torch_csr(matrix, dtype, device):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data).to(dtype), size=matrix.shape, device=device)


# ------------------------------------------------------------------- objectives
# An objective scores, for every pool row at once, the metric on the current
# subset plus that row (`scores`), and adds a row to the subset (`add`).

class DistSum:
    """Mean cosine distance over pairs, from the sum of the subset's unit vectors."""

    def __init__(self, embeddings, device):
        unit = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
        self.x = torch.from_numpy(unit).to(device)
        self.total = torch.zeros(self.x.shape[1], dtype=torch.float64, device=device)
        self.n = 0

    def add(self, i):
        self.total += self.x[i]
        self.n += 1

    def scores(self):
        n = self.n + 1
        # ||total + x||^2 for unit x.
        squared = self.total @ self.total + 2 * (self.x @ self.total) + 1
        return 1 - (squared - n) / (n * (n - 1))


class Radius:
    """Geometric mean of per-dimension standard deviations, from running moments."""

    def __init__(self, embeddings, device):
        self.x = torch.from_numpy(np.asarray(embeddings, dtype=np.float64)).to(device)
        self.squares = self.x * self.x
        self.sum = torch.zeros(self.x.shape[1], dtype=torch.float64, device=device)
        self.square_sum = torch.zeros_like(self.sum)
        self.n = 0

    def add(self, i):
        self.sum += self.x[i]
        self.square_sum += self.squares[i]
        self.n += 1

    def scores(self):
        n = self.n + 1
        mean = (self.sum + self.x) / n
        variance = ((self.square_sum + self.squares) / n - mean * mean).clamp_(min=0)
        # A zero variance gives log 0 = -inf and a radius of 0, as in the metric.
        return torch.exp(0.5 * torch.log(variance).mean(dim=1))


class ClusterInertia:
    """k-means inertia, each candidate refit by Lloyd's iterations from the subset's solution."""

    def __init__(self, embeddings, n_clusters, seed, device, block=4096):
        self.x = torch.from_numpy(np.asarray(embeddings, dtype=np.float64)).to(device)
        self.norms = (self.x * self.x).sum(dim=1)
        self.k, self.seed, self.block = n_clusters, seed, block
        self.members, self.centroids, self.inertia, self.stale = [], None, None, False

    def add(self, i):
        self.members.append(i)
        self.stale = True

    def lloyd(self, points, centroids):
        """Iterate to a fixed assignment; an emptied cluster keeps its centroid."""
        norms = (points * points).sum(dim=1, keepdim=True)
        assignment = None
        for _ in range(300):
            distances = norms - 2 * points @ centroids.T + (centroids * centroids).sum(dim=1)
            closest = distances.argmin(dim=1)
            if assignment is not None and torch.equal(closest, assignment):
                break
            assignment = closest
            member = torch.nn.functional.one_hot(assignment, self.k).double()
            counts = member.sum(dim=0).unsqueeze(1)
            centroids = torch.where(counts > 0, member.T @ points / counts.clamp(min=1), centroids)
        return centroids, float(distances.gather(1, assignment.unsqueeze(1)).sum())

    def refit(self):
        """Best solution for the current subset: the previous one refined, or a fresh fit."""
        if not self.stale:
            return
        points = self.x[self.members]
        fresh = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(
            points.cpu().numpy())
        solutions = [self.lloyd(points, torch.from_numpy(fresh.cluster_centers_).to(points))]
        if self.centroids is not None:
            solutions.append(self.lloyd(points, self.centroids))
        self.centroids, self.inertia = min(solutions, key=lambda solution: solution[1])
        self.stale = False

    def value(self):
        self.refit()
        return self.inertia

    def scores(self):
        self.refit()
        return torch.cat([self.score_block(torch.arange(start, min(start + self.block, len(self.x)),
                                                        device=self.x.device))
                          for start in range(0, len(self.x), self.block)])

    def score_block(self, candidates):
        k, x, x_norms = self.k, self.x[candidates], self.norms[candidates]
        points, point_norms = self.x[self.members], self.norms[self.members]
        centroids = self.centroids
        centroid_norms = (centroids * centroids).sum(dim=1)
        base = point_norms.unsqueeze(1) - 2 * points @ centroids.T + centroid_norms
        base_assignment = base.argmin(dim=1)
        member = torch.nn.functional.one_hot(base_assignment, k).double()
        count = len(candidates)
        rows = torch.arange(count, device=x.device)
        # Per candidate: member-to-centroid distances, assignments, cluster sums and sizes.
        distances = base.unsqueeze(0).repeat(count, 1, 1)
        assignment = base_assignment.unsqueeze(0).repeat(count, 1)
        sums = (member.T @ points).unsqueeze(0).repeat(count, 1, 1)
        sizes = member.sum(dim=0).unsqueeze(0).repeat(count, 1)
        own_distances = x_norms.unsqueeze(1) - 2 * x @ centroids.T + centroid_norms
        own = own_distances.argmin(dim=1)
        sums[rows, own] += x
        sizes[rows, own] += 1
        changed = rows * k + own  # (candidate, cluster) pairs whose centroid moved
        for _ in range(300):
            row, cluster = changed // k, changed % k
            filled = sizes[row, cluster] > 0
            row, cluster = row[filled], cluster[filled]
            moved = sums[row, cluster] / sizes[row, cluster].unsqueeze(1)
            moved_norms = (moved * moved).sum(dim=1)
            distances[row, :, cluster] = (point_norms.unsqueeze(0) - 2 * moved @ points.T
                                          + moved_norms.unsqueeze(1))
            own_distances[row, cluster] = x_norms[row] - 2 * (x[row] * moved).sum(dim=1) + moved_norms
            active = torch.unique(row)
            closest = distances[active].argmin(dim=2)
            switch = (closest != assignment[active]).nonzero()
            switch_row, switch_point = active[switch[:, 0]], switch[:, 1]
            old, new = assignment[switch_row, switch_point], closest[switch[:, 0], switch[:, 1]]
            own_closest = own_distances[active].argmin(dim=1)
            own_switch = own_closest != own[active]
            own_row, own_new = active[own_switch], own_closest[own_switch]
            own_old = own[own_row]
            for rows_, clusters, vectors, sign in (
                    (switch_row, old, points[switch_point], -1), (switch_row, new, points[switch_point], 1),
                    (own_row, own_old, x[own_row], -1), (own_row, own_new, x[own_row], 1)):
                sums.index_put_((rows_, clusters), sign * vectors, accumulate=True)
                sizes.index_put_((rows_, clusters), torch.full_like(clusters, sign, dtype=sizes.dtype),
                                 accumulate=True)
            assignment[switch_row, switch_point] = new
            own[own_row] = own_new
            changed = torch.unique(torch.cat([switch_row * k + old, switch_row * k + new,
                                              own_row * k + own_old, own_row * k + own_new]))
            if not len(changed):
                break
        return (distances.gather(2, assignment.unsqueeze(2)).sum(dim=(1, 2))
                + own_distances.gather(1, own.unsqueeze(1)).squeeze(1))


class KnnCosine:
    """Mean cosine distance to the k nearest other examples."""

    def __init__(self, embeddings, k, size, device):
        unit = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
        self.x = torch.from_numpy(unit).to(device)
        self.k = k
        # distances[c, j]: distance from pool row c to the j-th member.
        self.distances = torch.empty(len(unit), size, dtype=torch.float64, device=device)
        self.members = []

    def add(self, i):
        self.distances[:, len(self.members)] = (1 - self.x @ self.x[i]).clamp_(min=0)
        self.members.append(i)

    def scores(self):
        n, k = len(self.members), self.k
        distances = self.distances[:, :n]
        among = distances[self.members].clone()
        among.fill_diagonal_(math.inf)
        nearest = among.topk(k, dim=1, largest=False).values
        # A candidate closer than a member's k-th neighbor replaces that neighbor.
        saved = (nearest[:, -1].unsqueeze(0) - distances).clamp_(min=0).sum(dim=1)
        own = distances.topk(k, dim=1, largest=False).values.sum(dim=1)
        return (nearest.sum() - saved + own) / ((n + 1) * k)


class DistinctN:
    """Unique / total n-grams of the pooled subset, averaged over `count_matrices`."""

    def __init__(self, count_matrices, device):
        self.orders = []
        for counts in count_matrices:
            self.orders.append({
                'counts': counts,
                'present': torch_csr((counts > 0).astype(np.float64), torch.float64, device),
                'distinct': torch.from_numpy(counts.getnnz(axis=1).astype(np.float64)).to(device),
                'total': torch.from_numpy(np.asarray(counts.sum(axis=1)).ravel()
                                          .astype(np.float64)).to(device),
                'seen': torch.zeros(counts.shape[1], dtype=torch.float64, device=device),
                'subset_total': 0.0,
            })

    def add(self, i):
        for order in self.orders:
            counts = order['counts']
            grams = counts.indices[counts.indptr[i]:counts.indptr[i + 1]]
            order['seen'][torch.from_numpy(grams.astype(np.int64)).to(order['seen'].device)] = 1
            order['subset_total'] += float(order['total'][i])

    def scores(self):
        value = 0
        for order in self.orders:
            seen = order['seen']
            new = order['distinct'] - order['present'] @ seen
            # 0 / 0 (no n-gram of this order) is nan, as in the metric.
            value = value + (seen.sum() + new) / (order['subset_total'] + order['total'])
        return value / len(self.orders)


class Hdd:
    """HD-D of the pooled subset: sum over types of P(type in `draws` draws) / draws."""

    def __init__(self, counts, device, draws=cd.HDD_DRAWS):
        coo = counts.tocoo()
        self.rows = torch.from_numpy(coo.row.astype(np.int64)).to(device)
        self.types = torch.from_numpy(coo.col.astype(np.int64)).to(device)
        self.counts = torch.from_numpy(coo.data.astype(np.float64)).to(device)
        self.matrix = counts
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        unique_lengths, inverse = np.unique(lengths, return_inverse=True)
        self.lengths = torch.from_numpy(lengths).to(device)
        self.unique_lengths = torch.from_numpy(unique_lengths).to(device)
        self.length_index = torch.from_numpy(inverse.astype(np.int64)).to(device)
        self.frequency = torch.zeros(counts.shape[1], dtype=torch.float64, device=device)
        self.total = 0.0
        self.draws = draws
        self.offsets = torch.arange(draws, dtype=torch.float64, device=device)

    def add(self, i):
        counts = self.matrix
        entries = slice(counts.indptr[i], counts.indptr[i + 1])
        types = torch.from_numpy(counts.indices[entries].astype(np.int64)).to(self.frequency.device)
        self.frequency[types] += torch.from_numpy(
            counts.data[entries].astype(np.float64)).to(self.frequency.device)
        self.total += float(self.lengths[i])

    def present(self, total, frequency):
        """1 - C(total - f, draws) / C(total, draws), elementwise (broadcast inputs)."""
        ratio = (frequency.unsqueeze(-1) / (total.unsqueeze(-1) - self.offsets)).clamp_(max=1)
        # prod_j (1 - f / (total - j)); a factor of 0 (f > total - draws) gives exactly 1.
        return -torch.expm1(torch.log1p(-ratio).sum(dim=-1))

    def scores(self):
        totals = self.total + self.unique_lengths
        # Types the candidate does not contain depend on it only through its length.
        frequencies, multiplicity = torch.unique(self.frequency[self.frequency > 0],
                                                 return_counts=True)
        base = (self.present(totals.unsqueeze(1), frequencies.unsqueeze(0))
                * multiplicity).sum(dim=1)[self.length_index]
        total = self.total + self.lengths[self.rows]
        frequency = self.frequency[self.types]
        change = self.present(total, frequency + self.counts) - self.present(total, frequency)
        value = base.index_add(0, self.rows, change) / self.draws
        return value.masked_fill(self.total + self.lengths < self.draws, math.nan)


class PosSelfBleu:
    """POS Self-BLEU-k (NLTK, method1 smoothing) of the subset with each candidate added."""

    def __init__(self, count_matrices, lengths, size, device):
        self.k, self.size, self.device = len(count_matrices), size, device
        self.lengths = torch.from_numpy(np.asarray(lengths, dtype=np.float64)).to(device)
        self.members = []
        self.orders = [self._order(counts) for counts in count_matrices]

    def _order(self, counts):
        device = self.device
        coo = counts.tocoo()
        # Column (g, t) of `reach` marks the rows whose count of n-gram g is above t, so
        # min(a, b) of two counts is the number of shared columns. A column only one pool
        # row reaches cannot be shared by a candidate and a member, and is dropped.
        highest = np.asarray(counts.max(axis=0).todense()).ravel().astype(np.int64)
        offset = np.cumsum(highest) - highest
        repeat = coo.data.astype(np.int64)
        level = np.arange(int(repeat.sum())) - np.repeat(np.cumsum(repeat) - repeat, repeat)
        column = np.repeat(offset[coo.col], repeat) + level
        kept = np.bincount(column, minlength=int(highest.sum())) >= 2
        compact = np.where(kept, np.cumsum(kept) - 1, -1)
        entries = kept[column]
        width = max(1, int(kept.sum()))
        reach = sparse.csr_matrix(
            (np.ones(int(entries.sum()), np.float32),
             (np.repeat(coo.row, repeat)[entries], compact[column[entries]])),
            shape=(counts.shape[0], width))
        column_gram = np.repeat(np.arange(len(highest)), highest)[kept]
        column_level = (np.arange(int(highest.sum())) - np.repeat(offset, highest))[kept]
        as_tensor = lambda array: torch.from_numpy(np.asarray(array, dtype=np.int64)).to(device)
        return {
            'counts': counts,
            'reach': torch_csr(reach, torch.float32, device),
            'offset': as_tensor(offset), 'compact': as_tensor(compact),
            'column_gram': as_tensor(column_gram), 'column_level': as_tensor(column_level),
            # The two largest per-member counts of each n-gram.
            'first': torch.zeros(len(highest), dtype=torch.int64, device=device),
            'second': torch.zeros(len(highest), dtype=torch.int64, device=device),
            'member': [], 'gram': [], 'count': [],
            # weights[(g, t), h] = 1 where member h gains a match if a candidate reaches (g, t).
            'weights': torch.zeros(width, self.size, dtype=torch.float32, device=device),
            'set': None,
        }

    def add(self, i):
        position = len(self.members)
        for order in self.orders:
            counts = order['counts']
            entries = slice(counts.indptr[i], counts.indptr[i + 1])
            gram = torch.from_numpy(counts.indices[entries].astype(np.int64)).to(self.device)
            count = torch.from_numpy(counts.data[entries].astype(np.int64)).to(self.device)
            first, second = order['first'][gram], order['second'][gram]
            order['first'][gram] = torch.maximum(first, count)
            order['second'][gram] = torch.where(count >= first, first,
                                                torch.maximum(second, count))
            order['member'].append(torch.full_like(gram, position))
            order['gram'].append(gram)
            order['count'].append(count)
        self.members.append(i)

    def scores(self):
        n = len(self.members)
        member_length = self.lengths[self.members]
        # Closest reference length, the shorter one on ties (as NLTK's closest_ref_length).
        gap = (member_length.unsqueeze(1) - member_length.unsqueeze(0)).abs()
        key = gap * 1e7 + member_length.unsqueeze(0)
        key.fill_diagonal_(math.inf)
        closest = member_length[key.argmin(dim=1)]
        candidate_length = self.lengths.unsqueeze(1)
        gap = (candidate_length - member_length.unsqueeze(0)).abs()
        closest_gap = (closest - member_length).abs()
        reference = torch.where(
            (gap < closest_gap) | ((gap == closest_gap) & (candidate_length < closest)),
            candidate_length, closest)
        penalty = torch.where(member_length > reference, 1.0,
                              torch.exp(1 - reference / member_length))
        own_reference = member_length[(gap * 1e7 + member_length.unsqueeze(0)).argmin(dim=1)]
        own_penalty = torch.where(self.lengths > own_reference, 1.0,
                                  torch.exp(1 - own_reference / self.lengths))

        log_precision = torch.zeros(len(self.lengths), n, dtype=torch.float64, device=self.device)
        own_log_precision = torch.zeros(len(self.lengths), dtype=torch.float64, device=self.device)
        for index, order in enumerate(self.orders):
            member = torch.cat(order['member'])
            gram = torch.cat(order['gram'])
            count = torch.cat(order['count'])
            first, second = order['first'][gram], order['second'][gram]
            # Largest count of the n-gram among the other members.
            others = torch.where(count == first, second, first)
            matches = torch.zeros(n, dtype=torch.float64, device=self.device).index_add_(
                0, member, torch.minimum(count, others).double())
            # Levels others < t <= count: a candidate reaching level t adds one match.
            short = count - others
            lacking = short > 0
            repeat = short[lacking]
            level = (torch.arange(int(repeat.sum()), device=self.device)
                     - torch.repeat_interleave(repeat.cumsum(0) - repeat, repeat))
            column = order['compact'][torch.repeat_interleave(
                order['offset'][gram[lacking]] + others[lacking], repeat) + level]
            flat = (column * self.size + torch.repeat_interleave(member[lacking], repeat))[column >= 0]
            weights = order['weights']
            if order['set'] is not None:
                weights.view(-1)[order['set']] = 0
            weights.view(-1)[flat] = 1
            order['set'] = flat
            gained = torch.sparse.mm(order['reach'], weights)[:, :n]
            matched = matches.unsqueeze(0) + gained.double()
            own_matched = (order['reach'] @ (order['column_level'] < order['first'][
                order['column_gram']]).float()).double()
            if index == 0:
                # NLTK scores a hypothesis with no matching unigram as 0.
                alive, own_alive = matched > 0, own_matched > 0
            possible = (member_length - index).clamp(min=1)
            log_precision += torch.log(torch.where(matched > 0, matched, 0.1) / possible)
            own_possible = (self.lengths - index).clamp(min=1)
            own_log_precision += torch.log(
                torch.where(own_matched > 0, own_matched, 0.1) / own_possible)
        total = (penalty * torch.exp(log_precision / self.k) * alive).sum(dim=1)
        total += own_penalty * torch.exp(own_log_precision / self.k) * own_alive
        return total / (n + 1)


# ------------------------------------------------------------------- baselines

def unit_tensor(embeddings, device):
    unit = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    return torch.from_numpy(unit).to(device)


class KCenterGreedy:
    """Score: cosine distance to the nearest selected row."""

    def __init__(self, embeddings, device):
        self.x = unit_tensor(embeddings, device)
        self.nearest = torch.full((len(self.x),), math.inf, dtype=torch.float64, device=device)

    def add(self, i):
        self.nearest = torch.minimum(self.nearest, (1 - self.x @ self.x[i]).clamp_(min=0))

    def scores(self):
        return self.nearest


class FacilityLocation:
    """Score: sum over the pool of the largest cosine similarity to a selected row."""

    def __init__(self, embeddings, device, block=2048):
        self.x = unit_tensor(embeddings, device)
        self.similarity = self.x @ self.x.T
        self.covered = torch.full((len(self.x),), -math.inf, dtype=torch.float64, device=device)
        self.block = block

    def add(self, i):
        self.covered = torch.maximum(self.covered, self.similarity[i])

    def scores(self):
        gain = torch.cat([(rows - self.covered).clamp_(min=0).sum(dim=1)
                          for rows in self.similarity.split(self.block)])
        return self.covered.sum() + gain


def local_density(embeddings, neighbors, power, device, regularization=1e-9):
    """NovelSum's density: (mean squared L2 distance to the nearest distinct rows) ** -power."""
    distinct, inverse = np.unique(np.asarray(embeddings, dtype=np.float32), axis=0,
                                  return_inverse=True)
    points = torch.from_numpy(distinct).double().to(device)
    norms = (points * points).sum(dim=1)
    neighbors = min(neighbors, len(points) - 1)
    nearest = torch.cat([
        (norms[start:start + 2048].unsqueeze(1) - 2 * points[start:start + 2048] @ points.T + norms)
        # The closest row is the row itself.
        .topk(neighbors + 1, dim=1, largest=False).values[:, 1:].clamp_(min=0)
        for start in range(0, len(points), 2048)])
    density = 1 / (nearest.mean(dim=1) + regularization) ** power
    return density[torch.from_numpy(np.asarray(inverse).ravel()).to(device)]


class NovelSelect:
    """Score: NovelSum's novelty of a row given the selected rows (novelselect.py)."""

    def __init__(self, embeddings, size, device, neighbors=10, density_power=0.5,
                 distance_power=1.0):
        self.x = unit_tensor(embeddings, device)
        self.density = local_density(embeddings, neighbors, density_power, device)
        # weighted[c, j]: density-weighted distance from row c to the j-th selected row.
        self.weighted = torch.empty(len(self.x), size, dtype=torch.float64, device=device)
        self.rank_weights = 1 / torch.arange(1, size + 1, dtype=torch.float64,
                                             device=device) ** distance_power
        self.n = 0

    def add(self, i):
        self.weighted[:, self.n] = (1 - self.x @ self.x[i]) * (self.density + self.density[i])
        self.n += 1

    def scores(self):
        ordered = self.weighted[:, :self.n].sort(dim=1).values
        return (ordered * self.rank_weights[:self.n]).mean(dim=1)


def take_in_order(rows, initial, size, strata):
    """Rows of `rows`, in turn, that are not initial and whose stratum still has room."""
    room, chosen, picked = strata.room(initial, size), set(initial), []
    for row in rows:
        row = int(row)
        if row not in chosen and room[strata.labels[row]] > 0:
            picked.append(row)
            room[strata.labels[row]] -= 1
            if not room.any():
                break
    if room.any():
        raise ValueError(f'The strata ran out of rows; still needed {room.tolist()}')
    return picked


def farthest_select(embeddings, initial, size, device, strata):
    """The initial rows, then the rows with the largest total cosine distance to the pool."""
    x = unit_tensor(embeddings, device)
    total = (len(x) - x @ x.sum(dim=0)).cpu().numpy()
    picked = take_in_order(np.argsort(-total, kind='stable'), initial, size, strata)
    return list(initial) + picked, [float(total[row]) for row in picked]


def random_select(initial, size, seed, strata):
    """The initial rows, then rows drawn uniformly until every stratum is full."""
    order = np.random.default_rng((seed, 3)).permutation(len(strata.labels))
    picked = take_in_order(order, initial, size, strata)
    return list(initial) + picked, [math.nan] * len(picked)


def evenly_select(groups, initial, size, seed, strata):
    """Each stratum's quota spread as evenly as possible over its subcategories.

    `groups` holds each row's subcategories. Each step draws a random row of the
    subcategory with the fewest rows so far that still has rows (on ties the
    subcategory with fewer rows in the stratum, then the name), and the row counts
    toward that subcategory. An initial row counts toward the least-filled of its own.
    """
    rng = np.random.default_rng((seed, 4))
    room, taken, picked = strata.room(initial, size), set(initial), []
    for stratum in range(len(strata)):
        rows = strata.rows(stratum)
        names = sorted({group for row in rows for group in groups[row]})
        members = {name: [int(row) for row in
                          rng.permutation([row for row in rows if name in groups[row]])]
                   for name in names}
        available = {name: len(found) for name, found in members.items()}
        filled = dict.fromkeys(names, 0)

        def order(name):
            return filled[name], available[name], name

        for row in initial:
            if strata.labels[row] == stratum:
                filled[min(groups[row], key=order)] += 1
        for _ in range(room[stratum]):
            for name in names:
                while members[name] and members[name][-1] in taken:
                    members[name].pop()
            open_names = [name for name in names if members[name]]
            if not open_names:
                raise ValueError(f'Stratum {strata.names[stratum]} ran out of rows')
            name = min(open_names, key=order)
            row = members[name].pop()
            taken.add(row)
            picked.append(row)
            filled[name] += 1
    return list(initial) + picked, [math.nan] * len(picked)


@njit(cache=True)
def filter_rows(similarity, order, initial, threshold, limit):
    """Rows of `order` kept in turn: similarity to every kept row is below `threshold`."""
    kept = np.empty(len(initial) + limit, dtype=np.int64)
    kept[:len(initial)] = initial
    count = len(initial)
    closest = np.empty(limit)
    for row in order:
        if count == len(kept):
            break
        largest = -np.inf
        for j in range(count):
            if similarity[row, kept[j]] > largest:
                largest = similarity[row, kept[j]]
        if largest < threshold:
            closest[count - len(initial)] = largest
            kept[count] = row
            count += 1
    return kept[len(initial):count], closest[:count - len(initial)]


def repr_filter_threshold(similarity, order, initial, limit):
    """filter_rows at the lowest threshold that keeps `limit` rows (bisection)."""
    low, high = -1.0, 1.0 + 1e-6  # nothing passes at -1; everything but exact repeats above 1
    best = filter_rows(similarity, order, initial, high, limit)
    threshold = high
    if len(best[0]) < limit:
        raise ValueError('Repr Filter: too few rows have distinct embeddings')
    for _ in range(40):
        middle = (low + high) / 2
        result = filter_rows(similarity, order, initial, middle, limit)
        if len(result[0]) == limit:
            best, threshold, high = result, middle, middle
        else:
            low = middle
    return best, threshold


def repr_filter_select(embeddings, initial, size, seed, device, strata):
    """Repr Filter in a random order: each stratum at the lowest threshold that fills it.

    A row is compared with the kept rows of its own stratum. The rows of all strata
    are returned in the order the pool was visited."""
    x = unit_tensor(embeddings, device)
    similarity = (x @ x.T).float().cpu().numpy()
    initial = np.asarray(initial, dtype=np.int64)
    order = np.random.default_rng((seed, 1)).permutation(len(x))
    order = order[~np.isin(order, initial)]
    room = strata.room(initial, size)
    picked, closest = [], []
    for stratum in range(len(strata)):
        if room[stratum] == 0:
            continue
        inside = strata.labels == stratum
        (kept, nearest), threshold = repr_filter_threshold(
            similarity, order[inside[order]], initial[inside[initial]], int(room[stratum]))
        label = '' if len(strata) == 1 else f' ({strata.names[stratum]})'
        print(f'    Repr Filter threshold{label}: cosine similarity below {threshold:.4f}')
        picked += kept.tolist()
        closest += nearest.tolist()
    position = np.empty(len(x), dtype=np.int64)
    position[order] = np.arange(len(order))
    ranking = np.argsort(position[picked], kind='stable')
    return initial.tolist() + [picked[i] for i in ranking], [closest[i] for i in ranking]


def kmeans_select(embeddings, initial, size, n_clusters, seed, strata):
    """In each stratum, the same number of rows from each of its k-means clusters, drawn
    at random. score: the row's cluster within its stratum."""
    x = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.default_rng((seed, 2))
    quotas, chosen = strata.quotas(size), set(initial)
    picked, scores = [], []
    for stratum in range(len(strata)):
        rows = strata.rows(stratum)
        clusters = min(n_clusters, len(rows))
        labels = np.full(len(x), -1)
        labels[rows] = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(x[rows])
        waiting = [[int(row) for row in rng.permutation(rows[labels[rows] == cluster])
                    if row not in chosen] for cluster in range(clusters)]
        own = [row for row in initial if strata.labels[row] == stratum]
        held = np.bincount(labels[own], minlength=clusters)
        target, count = quotas[stratum], len(own)
        share = [target // clusters + (cluster < target % clusters) for cluster in range(clusters)]
        # Fill every cluster to its share, then spread what small clusters could not supply.
        for limit in (share, [target] * clusters):
            progress = True
            while count < target and progress:
                progress = False
                for cluster in range(clusters):
                    if waiting[cluster] and held[cluster] < limit[cluster] and count < target:
                        picked.append(waiting[cluster].pop())
                        scores.append(float(cluster))
                        held[cluster] += 1
                        count += 1
                        progress = True
    return list(initial) + picked, scores


def build_baseline(name, embeddings, size, device, seed=DEFAULT_SEED, n_clusters=12,
                   strata=None, groups=None):
    """select(initial, label) -> (rows in selection order, score per added row).

    `embeddings` (None for random and evenly) and `groups` (each row's subcategories,
    for evenly) describe the pool; `strata` (default: one) fixes each stratum's rows."""
    if strata is None:
        strata = Strata.single(len(embeddings if embeddings is not None else groups))
    greedy = {'kcenter_greedy': lambda: KCenterGreedy(embeddings, device),
              'qdit': lambda: FacilityLocation(embeddings, device),
              'novelselect': lambda: NovelSelect(embeddings, size, device)}
    if name in greedy:
        return lambda initial, label='': greedy_select(greedy[name](), initial, size,
                                                       label=label, strata=strata)
    if name == 'farthest':
        return lambda initial, label='': farthest_select(embeddings, initial, size, device, strata)
    if name == 'repr_filter':
        return lambda initial, label='': repr_filter_select(embeddings, initial, size, seed,
                                                            device, strata)
    if name == 'kmeans':
        return lambda initial, label='': kmeans_select(embeddings, initial, size, n_clusters,
                                                       seed, strata)
    if name == 'random':
        return lambda initial, label='': random_select(initial, size, seed, strata)
    if name == 'evenly':
        return lambda initial, label='': evenly_select(groups, initial, size, seed, strata)
    raise ValueError(f'Unknown baseline: {name!r}')


@njit(parallel=True, fastmath=True, cache=True)
def bordered_vendi(eigenvalues, projections, diagonal, normalize, out):
    """Vendi Score of diag(eigenvalues) bordered by each row of `projections`.

    Candidate c gives the matrix [[diag(eigenvalues), z], [z', diagonal[c]]] with
    z = projections[c]; its eigenvalues are the roots of
        f(x) = diagonal[c] - x - sum_i z_i^2 / (eigenvalues_i - x),
    one in each interval between consecutive (ascending) eigenvalues and one beyond
    each end. f decreases between poles, so each root is bracketed and refined
    with a rational model that matches f and f' (as LAPACK's dlaed4), falling back to
    bisection whenever a step leaves the bracket. The score is exp(entropy) of
    the roots / (n + 1), over the positive roots, renormalized if `normalize`.
    """
    n = eigenvalues.size
    for c in prange(projections.shape[0]):
        if not np.isfinite(diagonal[c]):
            out[c] = np.nan
            continue
        squares = np.empty(n)
        norm = 0.0
        for i in range(n):
            squares[i] = projections[c, i] * projections[c, i]
            norm += squares[i]
        norm = math.sqrt(norm)
        corner = diagonal[c]
        tolerance = 1e-15 * (max(abs(eigenvalues[0]), abs(eigenvalues[n - 1]))
                             + abs(corner) + norm)
        roots = np.empty(n + 1)
        for j in range(n + 1):
            if j == 0:
                low = min(eigenvalues[0], corner) - norm - tolerance
                high = eigenvalues[0]
            elif j == n:
                low = eigenvalues[n - 1]
                high = max(eigenvalues[n - 1], corner) + norm + tolerance
            else:
                low = eigenvalues[j - 1]
                high = eigenvalues[j]
            left, right = low, high
            x = 0.5 * (low + high)
            if high - low > tolerance:
                for _ in range(200):
                    below = 0.0
                    below_slope = 0.0
                    for i in range(j):
                        inverse = 1.0 / (x - eigenvalues[i])
                        term = squares[i] * inverse
                        below += term
                        below_slope += term * inverse
                    above = 0.0
                    above_slope = 0.0
                    for i in range(j, n):
                        inverse = 1.0 / (eigenvalues[i] - x)
                        term = squares[i] * inverse
                        above += term
                        above_slope += term * inverse
                    f = corner - x + below - above
                    if f > 0:
                        low = x
                    elif f < 0:
                        high = x
                    else:
                        break
                    if j == 0:
                        gap = right - x
                        weight = gap * gap * above_slope
                        q = f + weight / gap - gap
                        root = math.sqrt(q * q + 4 * weight)
                        step = 2 * weight / (q + root) if q > 0 else 0.5 * (root - q)
                        proposal = right - step
                    elif j == n:
                        gap = x - left
                        weight = gap * gap * below_slope
                        q = f - weight / gap + gap
                        root = math.sqrt(q * q + 4 * weight)
                        step = 0.5 * (q + root) if q >= 0 else 2 * weight / (root - q)
                        proposal = left + step
                    else:
                        gap_left = x - left
                        gap_right = right - x
                        weight_left = gap_left * gap_left * (below_slope + 0.5)
                        weight_right = gap_right * gap_right * (above_slope + 0.5)
                        constant = f - weight_left / gap_left + weight_right / gap_right
                        width = right - left
                        p = weight_left + weight_right - constant * width
                        root = math.sqrt(max(p * p + 4 * constant * weight_left * width, 0.0))
                        if p > 0:
                            step = 2 * weight_left * width / (p + root)
                        else:
                            step = (root - p) / (2 * constant)
                        proposal = left + step
                    if not (proposal > low and proposal < high):
                        proposal = 0.5 * (low + high)
                        if not (proposal > low and proposal < high):
                            x = proposal
                            break
                    moved = abs(proposal - x)
                    x = proposal
                    if moved <= 4e-16 * abs(x) + tolerance:
                        break
            roots[j] = x
        mass = 0.0
        for j in range(n + 1):
            if roots[j] > 0:
                mass += roots[j]
        scale = 1.0 / mass if normalize else 1.0 / (n + 1)
        entropy = 0.0
        for j in range(n + 1):
            if roots[j] > 0:
                value = roots[j] * scale
                entropy -= value * math.log(value)
        out[c] = math.exp(entropy)


class Vendi:
    """Vendi Score of a kernel given by its columns (`column(i)`) and diagonal."""

    def __init__(self, column, diagonal, size, normalize):
        self.column, self.normalize = column, normalize
        self.diagonal = np.ascontiguousarray(diagonal, dtype=np.float64)
        # kernel[j] is the kernel column of the j-th member.
        self.kernel = np.empty((size, len(self.diagonal)))
        self.members = []

    def add(self, i):
        self.kernel[len(self.members)] = self.column(i)
        self.members.append(i)

    def scores(self):
        n = len(self.members)
        kernel = self.kernel[:n]
        among = kernel[:, self.members]
        eigenvalues, eigenvectors = np.linalg.eigh((among + among.T) / 2)
        projections = np.ascontiguousarray((eigenvectors.T @ kernel).T)
        out = np.empty(len(self.diagonal))
        bordered_vendi(eigenvalues, projections, self.diagonal, self.normalize, out)
        return out


def embedding_vendi(embeddings, size):
    """Vendi objective with the cosine kernel of `embeddings`."""
    unit = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    return Vendi(lambda i: unit @ unit[i], np.einsum('ij,ij->i', unit, unit), size, normalize=True)


def ngram_vendi(count_matrices, size):
    """Vendi objective with the n-gram kernel of calculate_diversity.ngram_vendi_score."""
    normalized = [preprocessing.normalize(counts.astype(np.float64)) for counts in count_matrices]
    transposed = [counts.T.tocsr() for counts in normalized]

    def column(i):
        return sum(np.asarray((counts @ columns[:, i]).todense()).ravel()
                   for counts, columns in zip(normalized, transposed)) / len(normalized)

    diagonal = sum(np.asarray(counts.multiply(counts).sum(axis=1)).ravel()
                   for counts in normalized) / len(normalized)
    # The reference divides by N without renormalizing the eigenvalues.
    return Vendi(column, diagonal, size, normalize=False)


# ------------------------------------------------------------------- selection

class Features:
    """Tags, words, n-gram counts and embeddings of the pool prompts, computed on demand."""

    def __init__(self, prompts, tags, words):
        self.prompts, self.tags, self.words = prompts, tags, words
        self.count_cache, self.embedding_cache = {}, {}

    def counts(self, kind, n):
        if (kind, n) not in self.count_cache:
            self.count_cache[kind, n] = ngram_counts(getattr(self, kind), n)
        return self.count_cache[kind, n]

    def embeddings(self, embedder, kind):
        """Unit embeddings of the raw prompts ('semantic') or POS strings ('structural')."""
        key = embedder.model_id, kind
        if key not in self.embedding_cache:
            self.embedding_cache[key] = (embedder.encode_texts(self.prompts) if kind == 'semantic'
                                         else embedder.encode(self.tags))
        return self.embedding_cache[key]


def build_objective(metric, features, embedder, size, device, seed=DEFAULT_SEED, n_clusters=12):
    """(objective, reference): the greedy objective and the metric as calculate_diversity
    computes it, as a function of pool row indices."""
    tags, words = features.tags, features.words
    pick = lambda items: lambda rows: [items[i] for i in rows]
    if metric == 'lexical_hdd':
        subset = pick(words)
        return Hdd(features.counts('words', 1), device), lambda rows: cd.hdd(subset(rows))
    if metric.startswith('lexical_distinct_'):
        n, subset = metric_number(metric), pick(words)
        return (DistinctN([features.counts('words', n)], device),
                lambda rows: cd.distinct_n(subset(rows), n))
    if metric == 'structural_distinct_1_4':
        subset = pick(tags)
        return (DistinctN([features.counts('tags', n) for n in cd.POS_NGRAM_ORDERS], device),
                lambda rows: float(np.mean([cd.distinct_n(subset(rows), n)
                                            for n in cd.POS_NGRAM_ORDERS])))
    if metric.startswith('structural_self_bleu_'):
        k, subset = metric_number(metric), pick(tags)
        return (PosSelfBleu([features.counts('tags', n) for n in range(1, k + 1)],
                         [len(seq) for seq in tags], size, device),
                lambda rows: cd.pos_self_bleu(subset(rows), k)[k - 1])
    if metric == 'structural_ngram_vendi':
        subset = pick(tags)
        return (ngram_vendi([features.counts('tags', n) for n in cd.POS_NGRAM_ORDERS], size),
                lambda rows: cd.ngram_vendi_score(subset(rows)))
    kind = 'semantic' if metric.startswith('semantic_') else 'structural'
    embeddings = features.embeddings(embedder, kind)
    if metric in ('semantic_vendi', 'structural_pos_vendi'):
        return embedding_vendi(embeddings, size), lambda rows: cd.vendi_score(embeddings[rows])
    if metric == 'semantic_distsum_cosine':
        return DistSum(embeddings, device), lambda rows: cd.distsum_cosine(embeddings[rows])
    if metric == 'semantic_cluster_inertia':
        return (ClusterInertia(embeddings, n_clusters, seed, device),
                lambda rows: cd.cluster_inertia(embeddings[rows], n_clusters, seed))
    if metric == 'semantic_radius':
        return Radius(embeddings, device), lambda rows: cd.radius(embeddings[rows])
    k = metric_number(metric)
    return (KnnCosine(embeddings, k, size, device),
            lambda rows: cd.knn_cosine_distance(embeddings[rows], k))


def greedy_select(objective, initial, size, minimize=False, label='', strata=None):
    """Grow `initial` to `size` rows, adding the best-scoring row each step.

    With `strata`, only rows of strata below their quota for `size` rows are
    candidates. Returns the rows in selection order and the metric after each
    addition (one value per added row): the picked row's score, or
    objective.value() where the objective has one. Ties go to the lowest row index.
    """
    selected = list(initial)
    for row in selected:
        objective.add(row)
    taken, scores, start = None, [], time.time()
    while len(selected) < size:
        values = torch.as_tensor(objective.scores())
        if taken is None:
            taken = torch.zeros(len(values), dtype=torch.bool, device=values.device)
            taken[selected] = True
            if strata is not None:
                stratum = torch.as_tensor(strata.labels, device=values.device)
                room = torch.as_tensor(strata.room(selected, size), device=values.device)
        if minimize:
            values = -values
        blocked = taken | values.isnan()
        if strata is not None:
            blocked |= (room <= 0)[stratum]
        values = values.masked_fill(blocked, -math.inf)
        best = values.max()
        if best == -math.inf:
            raise ValueError(f'{label}: no candidate has a defined score at size {len(selected)}')
        row = int(torch.nonzero(values == best)[0])
        objective.add(row)
        taken[row] = True
        if strata is not None:
            room[stratum[row]] -= 1
        selected.append(row)
        if hasattr(objective, 'value'):
            scores.append(objective.value())
        else:
            scores.append(-float(best) if minimize else float(best))
        if label and (len(selected) % 100 == 0 or len(selected) == size):
            print(f'    {label}: {len(selected)}/{size}, score {scores[-1]:.6f}, '
                  f'{time.time() - start:.0f}s', flush=True)
    return selected, scores


def load_pool(dataset, tokenizer, max_length=DEFAULT_MAX_LENGTH, options=DEFAULT_POOL):
    """Harmful prompts with an eligible response from every subcategory
    (helper.selection_pool with these pool options), short enough to train on. Without
    options.dedup a prompt with several eligible responses is several rows."""
    pool = selection_pool(dataset, options)
    rows = len(pool)
    lengths = np.asarray(formatted_prompt_lengths(pool['prompt'].tolist(), tokenizer))
    pool = pool[lengths < max_length]
    print(f'  Pool ({options.response_label_filter} responses'
          f'{", one row per prompt" if options.dedup else ""}): {rows:,} rows, {len(pool):,} '
          f'under {max_length:,} tokens ({pool["prompt"].nunique():,} distinct prompts)')
    return pool.rename_axis('pool_index').reset_index()


def pool_strata(pool, stratify):
    """Vanilla/adversarial strata where the pool has the adversarial flag, else one stratum."""
    if stratify == 'adversarial' and 'adversarial' in pool:
        flags = pool['adversarial'].astype(str).str.strip().str.lower().map({'false': 0, 'true': 1})
        if flags.isna().any():
            raise ValueError(f'Unexpected adversarial values: {sorted(set(pool["adversarial"]))}')
        return Strata(flags.to_numpy(), ['vanilla', 'adversarial'])
    return Strata.single(len(pool))


def row_groups(pool):
    """Each row's subcategories (sub_abbrs), as tuples; '' for a row without one."""
    return [tuple(groups) if isinstance(groups, (list, tuple, np.ndarray)) and len(groups)
            else ('',) for groups in pool['sub_abbrs']]


def stale_csvs(folder, size, names=None):
    """CSVs of `size` rows in `folder` without a stratum column (selected before
    stratification) or, given `names`, holding a stratum outside them (another --stratify)."""
    stale = []
    for path in sorted(Path(folder).glob(f'*_{size}.csv')):
        try:
            strata = pd.read_csv(path, usecols=['stratum'], keep_default_na=False)['stratum']
        except ValueError:  # no stratum column
            stale.append(path)
            continue
        if names is not None and not set(strata) <= set(names):
            stale.append(path)
    return stale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', type=str.lower, choices=DATASETS,
                        help=f'Datasets to select from (default: {" ".join(cd.DEFAULT_DATASETS)}; '
                             'wildguardmix with --response_label_filter refusal)')
    parser.add_argument('--embedding_models', '--embedding_model', dest='embedding_models',
                        nargs='+', default=list(DEFAULT_EMBEDDING_MODELS),
                        help='Embedding models for the metrics that need one: '
                             f'{" ".join(EMBEDDING_MODELS)} or a sentence-transformers model ID '
                             f'(default: {" ".join(DEFAULT_EMBEDDING_MODELS)})')
    parser.add_argument('--metrics', nargs='*', default=list(DEFAULT_METRICS),
                        help='Metrics to select with; give no value for none (default: the 12 '
                             f'the diversity figures show: {" ".join(DEFAULT_METRICS)}; also '
                             f'{" ".join(m for m in ALL_METRICS if m not in DEFAULT_METRICS)})')
    parser.add_argument('--baselines', nargs='*', default=list(BASELINES),
                        help='Baseline strategies; NovelSum\'s run on the prompt embeddings '
                             '(semantic_<name>), structural_pos_<name> runs one on POS '
                             'embeddings; give no value for none (default: all): '
                             f'{" ".join(BASELINES)}')
    parser.add_argument('--stratify', choices=('adversarial', 'none'), default='adversarial',
                        help='adversarial (default): keep the pool shares of vanilla and '
                             'adversarial prompts in the initial examples and every subset, '
                             'where the dataset flags them; none: one stratum')
    parser.add_argument('--size', type=int, default=600, help='Subset size (default: 600)')
    parser.add_argument('--init_size', type=int, default=50,
                        help='Random examples the selection starts from (default: 50)')
    parser.add_argument('--n_clusters', type=int, default=12,
                        help='k-means clusters for semantic_cluster_inertia (as in '
                             'calculate_diversity.py) and the kmeans baseline (default: 12)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Seed of the initial examples (default: {DEFAULT_SEED})')
    parser.add_argument('--output_dir', type=Path, default=Path(DATA_DIR),
                        help='Results go to <output_dir>/seed<seed>/<dataset>/ (default: data/)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Select again where the CSV exists (default: keep it)')
    parser.add_argument('--model', default='qwen2.5-ins',
                        help='Tokenizer model for the --max_length filter, as in align.py')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument('--device', default=None,
                        help='Device for embeddings and scoring, e.g. cuda:0 or cpu '
                             '(default: cuda if available)')
    parser.add_argument('--threads', type=int, default=min(32, os.cpu_count() or 1),
                        help='CPU threads for the Vendi Score metrics (default: 32)')
    parser.add_argument('--num_proc', type=int, default=8, help='spaCy processes (default: 8)')
    parser.add_argument('--embed_dtype', choices=('bfloat16', 'float16', 'float32'),
                        default='float32')
    parser.add_argument('--embed_batch_size', type=int, default=32)
    add_pool_arguments(parser)
    args = parser.parse_args()
    resolve_pool_options(parser, args, cd.DEFAULT_DATASETS)
    for metric in args.metrics:
        if not METRIC_PATTERN.fullmatch(metric):
            parser.error(f'Unknown metric {metric!r}; choose from: {" ".join(DEFAULT_METRICS)}')
    if len(set(args.metrics)) != len(args.metrics):
        parser.error('--metrics must be distinct')
    jobs = []
    for baseline in args.baselines:
        if baseline in EMBEDDING_BASELINES:
            jobs.append('semantic_' + baseline)
        elif baseline_parts(baseline):
            jobs.append(baseline)
        else:
            parser.error(f'Unknown baseline {baseline!r}; choose from: {" ".join(BASELINES)}')
    args.baselines = list(dict.fromkeys(jobs))
    if not args.metrics and not args.baselines:
        parser.error('Nothing to select: give --metrics or --baselines')
    if not 2 <= args.init_size < args.size:
        parser.error('--init_size must be at least 2 and below --size')
    neighbors = [metric_number(m) for m in args.metrics if m.startswith('structural_knn_cosine_')]
    if neighbors and max(neighbors) >= args.init_size:
        parser.error('--init_size must exceed every KNN neighbor count')
    if not 1 <= args.n_clusters <= args.init_size:
        parser.error('--n_clusters must be in [1, init_size]')
    if min(args.threads, args.num_proc, args.max_length) < 1:
        parser.error('--threads, --num_proc and --max_length must be positive')
    args.embedding_models = [EMBEDDING_MODELS.get(model.lower(), model)
                             for model in args.embedding_models]
    if len({cd.embedding_name(m) for m in args.embedding_models}) != len(args.embedding_models):
        parser.error('--embedding_models must have distinct model names')
    return args


def main():
    args = parse_args()
    set_num_threads(min(args.threads, get_num_threads()))
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    embedders = {
        cd.embedding_name(model_id): cd.PosEmbedder(
            batch_size=args.embed_batch_size, device=device, model_id=model_id,
            dtype=args.embed_dtype)
        for model_id in args.embedding_models
    }
    tokenizer = None
    for dataset in args.datasets:
        jobs = [(metric, name) for metric in args.metrics + args.baselines
                for name in (embedders if needs_embedding(metric) else [None])]
        paths = {job: output_path(args.output_dir, args.seed, dataset, *job, args.size,
                                  args.options)
                 for job in jobs}
        folder = output_path(args.output_dir, args.seed, dataset, 'any', None, args.size,
                             args.options).parent
        # Training lists every CSV in the folder, so earlier versions must not stay there.
        stale = stale_csvs(folder, args.size)
        others = [path for path in stale if path not in paths.values()]
        if others or (stale and not args.overwrite):
            raise SystemExit(
                f'{dataset}: {len(stale)} CSVs in {folder} were selected before stratified '
                f'selection (no stratum column), e.g. {stale[0].name}; training lists every CSV '
                'there, so move them away first'
                + ('.' if others else ' (or pass --overwrite to select them again).'))
        pending = [job for job in jobs if args.overwrite or not paths[job].exists()]
        print(f'{dataset}: {len(pending)} of {len(jobs)} selections to run '
              f'(size {args.size}, seed {args.seed})', flush=True)
        if not pending:
            continue
        if tokenizer is None:
            from transformers import AutoTokenizer
            from helper import model_mapping

            tokenizer = AutoTokenizer.from_pretrained(
                model_mapping.get(args.model, args.model), trust_remote_code=True)
        pool = load_pool(dataset, tokenizer, args.max_length, args.options)
        prompts = list(dict.fromkeys(pool['prompt']))
        print(f'  POS-tagging {len(prompts):,} distinct prompts...', flush=True)
        tags_of = dict(zip(prompts, cd.tag_corpus(prompts, num_proc=args.num_proc)))
        tagged = [tags_of[prompt] for prompt in pool['prompt']]
        # The metrics need at least one POS tag per prompt; as calculate_diversity.py,
        # stop rather than drop such prompts.
        untagged = [prompt for prompt, (tags, _) in tags_of.items() if not tags]
        if untagged:
            raise SystemExit(f'{dataset}: {len(untagged)} prompts have no POS tags, e.g. '
                             f'{untagged[0]!r}; the metrics cannot score them')
        if len(pool) < args.size:
            raise SystemExit(f'{dataset}: the pool has {len(pool):,} rows, fewer than '
                             f'--size {args.size}')
        strata = pool_strata(pool, args.stratify)
        mismatched = [path for path in stale_csvs(folder, args.size, strata.names)
                      if path not in {paths[job] for job in pending}]
        if mismatched:
            raise SystemExit(
                f'{dataset}: {len(mismatched)} CSVs in {folder} hold other strata than '
                f'{", ".join(strata.names)} (another --stratify), e.g. {mismatched[0].name}; '
                'move them away first.')
        features = Features(pool['prompt'].tolist(), [tags for tags, _ in tagged],
                            [words for _, words in tagged])
        initial = strata.initial(args.init_size, args.seed)
        print(f'  Strata (pool / initial / subset): ' + ', '.join(
            f'{name} {size:,} / {start} / {final}' for name, size, start, final in zip(
                strata.names, strata.sizes, strata.quotas(args.init_size),
                strata.quotas(args.size))), flush=True)
        order = [0] * args.init_size + list(range(1, args.size - args.init_size + 1))

        def save(selected, scores, path, label):
            if len(selected) != args.size or len(set(selected)) != args.size:
                raise SystemExit(f'{label}: selected {len(set(selected))} distinct rows')
            if strata.counts(selected) != dict(zip(strata.names, strata.quotas(args.size).tolist())):
                raise SystemExit(f'{label}: strata {strata.counts(selected)} differ from the '
                                 f'quotas {strata.quotas(args.size).tolist()}')
            temporary = path.with_name(path.name + '.tmp')
            pool.iloc[selected].assign(
                stratum=[strata.names[stratum] for stratum in strata.labels[selected]],
                order=order, score=scores).to_csv(temporary, index=False)
            temporary.replace(path)

        for metric, name in pending:
            label = f'{dataset} {metric}' + (f' ({name})' if name else '')
            start = time.time()
            path = paths[metric, name]
            path.parent.mkdir(parents=True, exist_ok=True)
            if baseline_parts(metric):
                kind, baseline = baseline_parts(metric)
                embeddings = None if kind == 'pool' else features.embeddings(embedders[name], kind)
                select = build_baseline(baseline, embeddings, args.size, device, seed=args.seed,
                                        n_clusters=args.n_clusters, strata=strata,
                                        groups=row_groups(pool))
                selected, scores = select(initial, label)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                save(selected, [math.nan] * args.init_size + scores, path, label)
                print(f'  {label}: done in {time.time() - start:.0f}s; saved {path}', flush=True)
                continue
            objective, reference = build_objective(
                metric, features, embedders.get(name), args.size, device,
                seed=args.seed, n_clusters=args.n_clusters)
            selected, scores = greedy_select(objective, initial, args.size,
                                             minimize=minimized(metric), label=label,
                                             strata=strata)
            # A k-means objective reports the best solution it found, which a fresh fit may miss.
            searched = hasattr(objective, 'value')
            del objective
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            first, last = reference(initial), reference(selected)
            if searched:
                last = scores[-1]
            # The incremental score and the metric computed from scratch must agree.
            elif not math.isclose(scores[-1], last, rel_tol=1e-6, abs_tol=1e-9):
                raise SystemExit(f'{label}: final score {scores[-1]!r} differs from the '
                                 f'metric on the selected subset, {last!r}')
            save(selected, [first] * args.init_size + scores, path, label)
            print(f'  {label}: {first:.6f} -> {last:.6f} in {time.time() - start:.0f}s; '
                  f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
