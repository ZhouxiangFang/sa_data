"""Stage 2: select a diverse harmful training subset from one safety dataset.

    python select_data.py --dataset wildjailbreak --seed 43
    python select_data.py --dataset wildguardmix --response_label_filter refusal
    python select_data.py --dataset alert_dpo --methods semantic_vendi random

Datasets: harmful prompts with a safe response, split into vanilla and adversarial strata.
Prompts equal to a test prompt (any of the seven test sets, case and spaces ignored) and prompts
of --max_length or more tokens are dropped; nothing else is (rows with the same prompt and
response stay separate rows, and each can be selected once).
  wildguardmix   WildGuardMix train: response labeled unharmful (--response_label_filter
                 refusal: also labeled a refusal); strata from its adversarial flag
  wildjailbreak  WildJailbreak train: vanilla_harmful / adversarial_harmful and their completions
  alert_dpo      ALERT-DPO: alert_dpo (vanilla) / alert_dpo_adversarial, the chosen response
  nemotron       Nemotron-SFT-Safety-v2, English rows with a refusal policy (the final answer,
                 no reasoning); strata by prompt source (NEMOTRON_STRATA)
Vanilla and adversarial rows are never mixed: a subset of --size rows takes from each stratum its
natural share (largest remainders), selected from that stratum alone by every method. Every
method starts from the same --init_size random rows (the same shares; --seed).

Methods (--methods, default all):
  metrics   lexical_hdd lexical_distinct_2 semantic_vendi semantic_distsum_cosine
            semantic_cluster_inertia structural_self_bleu_4 structural_self_bleu_5
            structural_distinct_5 structural_distinct_6 structural_pos_vendi
            structural_knn_cosine_1 structural_knn_cosine_3
            Greedy per stratum, as calculate_diversity.py defines each metric (k-means: one
            cluster per 50 rows of the stratum's share): each step draws a fresh pool of
            --candidates (5k) of the stratum's rows not yet selected and adds the one that gives
            the best metric on the stratum's selected rows (the largest; the smallest POS
            Self-BLEU), until the stratum has its share. Ties go to the lowest row. The final
            value is checked against the metric computed from scratch.
  baselines over the whole of each stratum, each stratum on its own, on the prompt embeddings:
    random          uniform draws
    qdit            adds the row that most raises Facility Location (sum over the stratum of each
                    row's largest cosine similarity to a selected row; Bukharin et al., 2024,
                    diversity term)
    kcenter_greedy  adds the row farthest (cosine) from its nearest selected row
    novelselect     adds the row of largest NovelSum novelty (Yang et al., 2025): its distances
                    to the selected rows times the two rows' densities, sorted, weighted 1/rank
    repr_filter     visits the rows in random order, keeping one if its cosine similarity to
                    every kept row is below a threshold (Liu et al., 2024), the lowest threshold
                    that fills the stratum's share (bisection)
    kmeans          k-means (one cluster per 50 rows of the stratum's share) and an equal
                    number of random rows from each cluster
Embeddings: Qwen/Qwen3-Embedding-0.6B in float32 (calculate_diversity.embed), of the prompts
(semantic_*, baselines) or their spaCy POS strings (structural_pos_vendi, structural_knn_*).
Tags and embeddings of a dataset are cached in --cache_dir and reused by every seed.

Ablation (--mixed_diversity, metrics only): each metric is computed on the whole subset, both
strata together; each step's pool still comes from one stratum (the one furthest below its
share). Its subsets go to <dataset>[_refusal]_mixed/.

Output: data/stage2_selected/seed<seed>/<dataset>[_refusal]/<method>_<size>.csv (helper.selection_csv),
stratum by stratum in selection order: pool row, stratum, order (within the stratum; 0: initial
rows), score (the stratum's metric after the row was added, or the baseline's own score),
prompt, response and the dataset's own fields: WildGuardMix subcategory and
response_refusal_label; WildJailbreak vanilla (the request an adversarial prompt rewrites; the
release has no tactics); ALERT-DPO id, category and attack_type; Nemotron uuid, prompt_source,
categories, subcategories, adversarial_persona and response_policy. Existing CSVs are kept
unless --overwrite.
"""

import argparse
import json
import math
import os
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
from helper import (DEFAULT_MAX_LENGTH, DEFAULT_SEED, SELECTION_DATASETS, add_pool_argument,
                    clean_pairs, eligible_rows, load_dataset, load_test_set, load_train,
                    model_mapping, overrefusal_test_datasets, pool_tag, prompt_lengths,
                    safety_test_datasets, selection_csv)

METRICS = tuple(metric.rstrip("↑↓") for metric in cd.METRICS)
BASELINES = ("random", "qdit", "kcenter_greedy", "novelselect", "repr_filter", "kmeans")
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
# Nemotron prompt sources: Aegis v2 prompts are plain requests; the others are jailbreaks,
# persona/tactic role-play (Gretel), safe-sounding context (SSCR) or jailbreak templates.
NEMOTRON_STRATA = {"Aegis v2": "vanilla", "sdg_jailbreaks": "adversarial",
                   "Gretel Safety Alignment v1": "adversarial", "sdg_sscr": "adversarial",
                   "HarmfulTasks": "adversarial"}
NEMOTRON_REFUSALS = ("refusal_with_explanation_redirection", "refusal_no_redirection",
                     "hard_refusal_with_helplines")


# ------------------------------------------------------------------------------- datasets
# Each loader returns prompt, response and stratum (vanilla / adversarial) per row, then the
# dataset's own fields for later analysis, saved with every selected row.

def load_wildguardmix(response_label_filter):
    rows = eligible_rows(load_train(), response_label_filter)
    adversarial = rows["adversarial"].astype(str).str.lower().map({"true": True, "false": False})
    return rows.assign(stratum=np.where(adversarial, "adversarial", "vanilla"))[
        ["prompt", "response", "stratum", "subcategory", "response_refusal_label"]]


def load_wildjailbreak():
    """vanilla: the plain request an adversarial prompt rewrites (the release has no tactics)."""
    df = load_dataset("allenai/wildjailbreak", "train", split="train", delimiter="\t",
                      keep_default_na=False).to_pandas()
    df = df[df["data_type"].isin(["vanilla_harmful", "adversarial_harmful"])]
    adversarial = df["data_type"] == "adversarial_harmful"
    return pd.DataFrame({"prompt": np.where(adversarial, df["adversarial"], df["vanilla"]),
                         "response": df["completion"],
                         "stratum": np.where(adversarial, "adversarial", "vanilla"),
                         "vanilla": np.where(adversarial, df["vanilla"], "")})


def load_alert_dpo():
    parts = [load_dataset("Babelscape/ALERT_DPO", config, split="train").to_pandas()
             .assign(stratum=stratum)
             for config, stratum in (("alert_dpo", "vanilla"),
                                     ("alert_dpo_adversarial", "adversarial"))]
    return pd.concat(parts, ignore_index=True).rename(columns={"chosen": "response"})[
        ["prompt", "response", "stratum", "id", "category", "attack_type"]]


def load_nemotron():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nvidia/Nemotron-SFT-Safety-v2", "data/train.jsonl",
                           repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["translation_languages"] is None and row["response_policy"] in NEMOTRON_REFUSALS:
                user, assistant = row["messages"]
                meta = row["metadata"]
                rows.append({"prompt": user["content"], "response": assistant["content"],
                             "stratum": NEMOTRON_STRATA[meta["source_l1"]],
                             "uuid": row["uuid"], "prompt_source": meta["source_l1"],
                             **{key: meta.get(key) for key in (
                                 "categories", "subcategories", "adversarial_persona")},
                             "response_policy": row["response_policy"]})
    return pd.DataFrame(rows)


LOADERS = {"wildguardmix": load_wildguardmix, "wildjailbreak": load_wildjailbreak,
           "alert_dpo": load_alert_dpo, "nemotron": load_nemotron}
assert tuple(LOADERS) == SELECTION_DATASETS


def normalized(text):
    return " ".join(str(text).lower().split())


def load_pool(dataset, response_label_filter, tokenizer, max_length):
    """The dataset's candidate rows (see the module doc), index 0..N-1."""
    if pool_tag(response_label_filter) and dataset != "wildguardmix":
        raise SystemExit(f"--response_label_filter refusal needs refusal labels; {dataset} has none")
    loader = LOADERS[dataset]
    pool = loader(response_label_filter) if dataset == "wildguardmix" else loader()
    pool = clean_pairs(pool)
    counts = [len(pool)]
    tests = {normalized(prompt) for name in safety_test_datasets + overrefusal_test_datasets
             for prompt in load_test_set(name)["prompt"]}
    pool = pool[~pool["prompt"].map(normalized).isin(tests)]
    counts.append(len(pool))
    lengths = np.concatenate([prompt_lengths(pool["prompt"].iloc[start:start + 4096], tokenizer)
                              for start in range(0, len(pool), 4096)])
    pool = pool[lengths < max_length].reset_index(drop=True)
    print(f"{dataset} pool: {counts[0]:,} rows, {counts[1]:,} without test prompts, "
          f"{len(pool):,} under {max_length:,} tokens; "
          + ", ".join(f"{name} {size:,}" for name, size in pool["stratum"].value_counts().items()),
          flush=True)
    return pool


class Strata:
    """Pool rows split into strata (vanilla, adversarial) whose natural shares a subset keeps."""

    def __init__(self, names):
        self.names = sorted(set(names), key=["vanilla", "adversarial"].index)
        self.labels = np.asarray([self.names.index(name) for name in names], dtype=np.int64)
        self.sizes = np.bincount(self.labels, minlength=len(self.names))

    def quotas(self, total):
        """Rows per stratum in a subset of `total`: the shares, rounded by largest remainders."""
        exact = self.sizes * total / self.sizes.sum()
        quotas = np.floor(exact).astype(np.int64)
        for stratum in np.argsort(-(exact - quotas), kind="stable")[:total - quotas.sum()]:
            quotas[stratum] += 1
        return quotas

    def rows(self, stratum):
        return np.flatnonzero(self.labels == stratum)

    def initial(self, size, seed):
        """`size` rows drawn uniformly within each stratum by its share, in pool order."""
        rng = np.random.default_rng(seed)
        return sorted(int(row) for stratum, quota in enumerate(self.quotas(size))
                      for row in rng.choice(self.rows(stratum), quota, replace=False))


# --------------------------------------------------------------------------------- features

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


class Features:
    """POS tags, words and embeddings of the pool prompts, cached per dataset in cache_dir."""

    def __init__(self, prompts, cache_dir, num_proc):
        self.prompts, self.counts = prompts, {}
        self.cache_dir = Path(cache_dir)
        texts = self.cache_dir / "texts.parquet"
        cached = pd.read_parquet(texts) if texts.exists() else None
        if cached is None or not set(prompts) <= set(cached["prompt"]):
            unique = list(dict.fromkeys(prompts))
            print(f"POS-tagging {len(unique):,} prompts...", flush=True)
            tagged = cd.tag_texts(unique, num_proc)
            cached = pd.DataFrame({"prompt": unique,
                                   "tags": [" ".join(tagged[p][0]) for p in unique],
                                   "words": [" ".join(tagged[p][1]) for p in unique]})
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(texts)
            for stale in self.cache_dir.glob("*.npy"):
                stale.unlink()
        self.cached = cached
        index = dict(zip(cached["prompt"], range(len(cached))))
        self.rows = np.asarray([index[prompt] for prompt in prompts])
        self.tags = [cached["tags"].iat[i].split() for i in self.rows]
        self.words = [cached["words"].iat[i].split() for i in self.rows]
        untagged = sum(not tags for tags in self.tags)
        if untagged:
            raise SystemExit(f"{untagged} prompts have no POS tags; the metrics cannot score them")

    def ngrams(self, kind, n):
        if (kind, n) not in self.counts:
            self.counts[kind, n] = ngram_counts(getattr(self, kind), n)
        return self.counts[kind, n]

    def embeddings(self, kind):
        """float64 unit embeddings of the prompts (semantic) or POS strings (structural)."""
        path = self.cache_dir / f"{kind}.npy"
        if not path.exists():
            texts = self.cached["prompt" if kind == "semantic" else "tags"].tolist()
            print(f"Embedding {len(texts):,} {kind} texts with {EMBEDDER}...", flush=True)
            vectors = cd.embed(EMBEDDER, list(dict.fromkeys(texts)))
            np.save(path, np.stack([vectors[text] for text in texts]).astype(np.float32))
        return preprocessing.normalize(np.load(path, mmap_mode="r")[self.rows].astype(np.float64))


def tensor(array, device, dtype=torch.float64):
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


# ------------------------------------------------------------------------------- objectives
# An objective adds a pool row to the subset (add) and scores, for given candidate rows, the
# metric of the subset with each candidate added (scores).

class DistSum:
    """Mean cosine distance over pairs, from the sum of the subset's unit vectors."""

    def __init__(self, x, device):
        self.x, self.n = tensor(x, device), 0
        self.total = torch.zeros(x.shape[1], dtype=torch.float64, device=device)

    def add(self, i):
        self.total += self.x[i]
        self.n += 1

    def scores(self, rows):
        n = self.n + 1
        squared = self.total @ self.total + 2 * (self.x[rows] @ self.total) + 1
        return 1 - (squared - n) / (n * (n - 1))


class KnnCosine:
    """Mean cosine distance to the k nearest other members."""

    def __init__(self, x, k, device):
        self.x, self.k, self.members = tensor(x, device), k, []

    def add(self, i):
        self.members.append(i)

    def scores(self, rows):
        members = self.x[self.members]
        among = (1 - members @ members.T).clamp_(min=0)
        among.fill_diagonal_(math.inf)
        nearest = among.topk(self.k, dim=1, largest=False).values
        distances = (1 - self.x[rows] @ members.T).clamp_(min=0)
        # A candidate closer than a member's k-th neighbor replaces that neighbor.
        saved = (nearest[:, -1].unsqueeze(0) - distances).clamp_(min=0).sum(dim=1)
        own = distances.topk(self.k, dim=1, largest=False).values.sum(dim=1)
        return (nearest.sum() - saved + own) / ((len(self.members) + 1) * self.k)


class ClusterInertia:
    """k-means inertia, each candidate refit by Lloyd's iterations from the subset's solution.

    Each step keeps the best solution found for the subset (the previous one refined, or a
    fresh fit as calculate_diversity runs it); a candidate's score is the inertia Lloyd's
    iterations reach from it with the candidate added, so it can be slightly below a fresh fit.
    """

    def __init__(self, x, n_clusters, seed, device, block=4096):
        self.x = tensor(x, device)
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

    def value(self):
        if self.stale:
            points = self.x[self.members]
            fresh = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(
                points.cpu().numpy())
            solutions = [self.lloyd(points, torch.from_numpy(fresh.cluster_centers_).to(points))]
            if self.centroids is not None:
                solutions.append(self.lloyd(points, self.centroids))
            self.centroids, self.inertia = min(solutions, key=lambda solution: solution[1])
            self.stale = False
        return self.inertia

    def scores(self, rows):
        self.value()
        return torch.cat([self.score_block(rows[start:start + self.block])
                          for start in range(0, len(rows), self.block)])

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
            own_distances[row, cluster] = (x_norms[row] - 2 * (x[row] * moved).sum(dim=1)
                                           + moved_norms)
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
                    (switch_row, old, points[switch_point], -1),
                    (switch_row, new, points[switch_point], 1),
                    (own_row, own_old, x[own_row], -1), (own_row, own_new, x[own_row], 1)):
                sums.index_put_((rows_, clusters), sign * vectors, accumulate=True)
                sizes.index_put_((rows_, clusters),
                                 torch.full_like(clusters, sign, dtype=sizes.dtype),
                                 accumulate=True)
            assignment[switch_row, switch_point] = new
            own[own_row] = own_new
            changed = torch.unique(torch.cat([switch_row * k + old, switch_row * k + new,
                                              own_row * k + own_old, own_row * k + own_new]))
            if not len(changed):
                break
        return (distances.gather(2, assignment.unsqueeze(2)).sum(dim=(1, 2))
                + own_distances.gather(1, own.unsqueeze(1)).squeeze(1))


class DistinctN:
    """Unique / total n-grams of the pooled subset."""

    def __init__(self, counts):
        self.present = (counts > 0).astype(np.float64).tocsr()
        self.distinct = counts.getnnz(axis=1).astype(np.float64)
        self.total = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        self.seen = np.zeros(counts.shape[1])
        self.subset_total = 0.0

    def add(self, i):
        self.seen[self.present.indices[self.present.indptr[i]:self.present.indptr[i + 1]]] = 1
        self.subset_total += self.total[i]

    def scores(self, rows):
        new = self.distinct[rows] - self.present[rows] @ self.seen
        with np.errstate(invalid="ignore", divide="ignore"):  # 0 / 0 is nan, as the metric
            return (self.seen.sum() + new) / (self.subset_total + self.total[rows])


class Hdd:
    """HD-D of the pooled subset: sum over types of P(type in `draws` draws) / draws."""

    def __init__(self, counts, device, draws=cd.HDD_DRAWS):
        self.matrix, self.device, self.draws = counts, device, draws
        self.lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        self.frequency = torch.zeros(counts.shape[1], dtype=torch.float64, device=device)
        self.total = 0.0
        self.offsets = torch.arange(draws, dtype=torch.float64, device=device)

    def add(self, i):
        entries = slice(self.matrix.indptr[i], self.matrix.indptr[i + 1])
        self.frequency[tensor(self.matrix.indices[entries], self.device, torch.int64)] += tensor(
            self.matrix.data[entries], self.device)
        self.total += self.lengths[i]

    def present(self, total, frequency):
        """1 - C(total - f, draws) / C(total, draws), elementwise (broadcast inputs)."""
        ratio = (frequency.unsqueeze(-1) / (total.unsqueeze(-1) - self.offsets)).clamp_(max=1)
        return -torch.expm1(torch.log1p(-ratio).sum(dim=-1))

    def scores(self, rows):
        lengths = tensor(self.lengths[rows], self.device)
        unique_lengths, inverse = torch.unique(lengths, return_inverse=True)
        # Types the candidate does not contain depend on it only through its length.
        frequencies, multiplicity = torch.unique(self.frequency[self.frequency > 0],
                                                 return_counts=True)
        base = (self.present((self.total + unique_lengths).unsqueeze(1), frequencies.unsqueeze(0))
                * multiplicity).sum(dim=1)[inverse]
        entries = self.matrix[rows].tocoo()
        row = tensor(entries.row, self.device, torch.int64)
        frequency = self.frequency[tensor(entries.col, self.device, torch.int64)]
        total = self.total + lengths[row]
        added = tensor(entries.data, self.device)
        change = self.present(total, frequency + added) - self.present(total, frequency)
        value = base.index_add(0, row, change) / self.draws
        return value.masked_fill(self.total + lengths < self.draws, math.nan)


class PosSelfBleu:
    """POS Self-BLEU-k (NLTK sentence BLEU, method1 smoothing) of the subset with each candidate.

    A member's clipped matches for an n-gram are min(its count, the largest count among the
    other members). A candidate with count a adds max(0, min(b, a) - o) for a member with count
    b above the others' largest o; only the member with the strictly largest count of an n-gram
    can gain, so each n-gram has at most one such member.
    """

    def __init__(self, count_matrices, lengths, device):
        self.orders, self.device, self.k = count_matrices, device, len(count_matrices)
        self.lengths = tensor(lengths, device)
        self.first = [torch.zeros(c.shape[1], dtype=torch.int64, device=device)
                      for c in count_matrices]
        self.second = [torch.zeros_like(first) for first in self.first]
        self.entries = [[] for _ in count_matrices]
        self.members = []

    def add(self, i):
        for index, counts in enumerate(self.orders):
            entries = slice(counts.indptr[i], counts.indptr[i + 1])
            gram = tensor(counts.indices[entries], self.device, torch.int64)
            count = tensor(counts.data[entries], self.device, torch.int64)
            first, second = self.first[index][gram], self.second[index][gram]
            self.first[index][gram] = torch.maximum(first, count)
            self.second[index][gram] = torch.where(count >= first, first,
                                                   torch.maximum(second, count))
            self.entries[index].append((torch.full_like(gram, len(self.members)), gram, count))
        self.members.append(i)

    def scores(self, rows):
        n, device = len(self.members), self.device
        member_length = self.lengths[self.members]
        length = self.lengths[tensor(rows, device, torch.int64)]
        # Closest reference length, the shorter on ties (NLTK's closest_ref_length).
        gap = (member_length.unsqueeze(1) - member_length.unsqueeze(0)).abs()
        key = gap * 1e7 + member_length.unsqueeze(0)
        key.fill_diagonal_(math.inf)
        closest = member_length[key.argmin(dim=1)]
        gap = (length.unsqueeze(1) - member_length.unsqueeze(0)).abs()
        closest_gap = (closest - member_length).abs()
        reference = torch.where((gap < closest_gap)
                                | ((gap == closest_gap) & (length.unsqueeze(1) < closest)),
                                length.unsqueeze(1), closest)
        penalty = torch.where(member_length > reference, 1.0,
                              torch.exp(1 - reference / member_length))
        own_reference = member_length[(gap * 1e7 + member_length.unsqueeze(0)).argmin(dim=1)]
        own_penalty = torch.where(length > own_reference, 1.0, torch.exp(1 - own_reference / length))
        log_precision = torch.zeros(len(rows), n, dtype=torch.float64, device=device)
        own_log_precision = torch.zeros(len(rows), dtype=torch.float64, device=device)
        for index, counts in enumerate(self.orders):
            member, gram, count = (torch.cat(part) for part in zip(*self.entries[index]))
            first, second = self.first[index], self.second[index]
            others = torch.where(count == first[gram], second[gram], first[gram])
            matches = torch.zeros(n, dtype=torch.float64, device=device).index_add_(
                0, member, torch.minimum(count, others).double())
            lacking = count > others
            order = gram[lacking].argsort()
            lacking_gram = gram[lacking][order]
            owner, low, high = member[lacking][order], others[lacking][order], count[lacking][order]
            entries = counts[rows].tocoo()
            row = tensor(entries.row, device, torch.int64)
            cgram = tensor(entries.col, device, torch.int64)
            ccount = tensor(entries.data, device, torch.int64)
            position = torch.searchsorted(lacking_gram, cgram).clamp_(max=max(len(lacking_gram) - 1, 0))
            hit = (lacking_gram[position] == cgram) if len(lacking_gram) else torch.zeros_like(
                cgram, dtype=torch.bool)
            position = position[hit]
            gain = (ccount[hit] - low[position]).clamp(min=0).minimum(high[position] - low[position])
            matched = matches.unsqueeze(0).repeat(len(rows), 1).index_put_(
                (row[hit], owner[position]), gain.double(), accumulate=True)
            own_matched = torch.zeros(len(rows), dtype=torch.float64, device=device).index_add_(
                0, row, torch.minimum(ccount, first[cgram]).double())
            if index == 0:  # NLTK scores a hypothesis without a matching unigram as 0
                alive, own_alive = matched > 0, own_matched > 0
            possible = (member_length - index).clamp(min=1)
            log_precision += torch.log(torch.where(matched > 0, matched, 0.1) / possible)
            own_possible = (length - index).clamp(min=1)
            own_log_precision += torch.log(
                torch.where(own_matched > 0, own_matched, 0.1) / own_possible)
        total = (penalty * torch.exp(log_precision / self.k) * alive).sum(dim=1)
        total += own_penalty * torch.exp(own_log_precision / self.k) * own_alive
        return total / (n + 1)


@njit(parallel=True, fastmath=True, cache=True)
def bordered_vendi(eigenvalues, projections, out):
    """Vendi Score (renormalized eigenvalues) of diag(eigenvalues) bordered by each row z of
    `projections` and a corner of 1 (a unit vector's cosine with itself).

    The eigenvalues of [[diag(eigenvalues), z], [z', 1]] are the roots of
        f(x) = 1 - x - sum_i z_i^2 / (eigenvalues_i - x),
    one in each interval between consecutive (ascending) eigenvalues and one beyond each end.
    f decreases between poles, so each root is bracketed and refined with a rational model
    matching f and f' (as LAPACK's dlaed4), falling back to bisection whenever a step leaves
    the bracket.
    """
    n = eigenvalues.size
    corner = 1.0
    for c in prange(projections.shape[0]):
        squares = np.empty(n)
        norm = 0.0
        for i in range(n):
            squares[i] = projections[c, i] * projections[c, i]
            norm += squares[i]
        norm = math.sqrt(norm)
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
        entropy = 0.0
        for j in range(n + 1):
            if roots[j] > 0:
                value = roots[j] / mass
                entropy -= value * math.log(value)
        out[c] = math.exp(entropy)


class Vendi:
    """Vendi Score with the cosine kernel of unit embeddings."""

    def __init__(self, x):
        self.x, self.members = x, []

    def add(self, i):
        self.members.append(i)

    def scores(self, rows):
        members = self.x[self.members]
        among = members @ members.T
        eigenvalues, eigenvectors = np.linalg.eigh((among + among.T) / 2)
        projections = np.ascontiguousarray(self.x[rows] @ members.T @ eigenvectors)
        out = np.empty(len(rows))
        bordered_vendi(eigenvalues, projections, out)
        return out


# --------------------------------------------------------------------- baseline objectives

class KCenterGreedy:
    """Score: cosine distance to the nearest selected row."""

    def __init__(self, x, device):
        self.x = tensor(x, device)
        self.nearest = torch.full((len(x),), math.inf, dtype=torch.float64, device=device)

    def add(self, i):
        self.nearest = torch.minimum(self.nearest, (1 - self.x @ self.x[i]).clamp_(min=0))

    def scores(self, rows):
        return self.nearest[rows]


class FacilityLocation:
    """Score: sum over the stratum of each row's largest cosine similarity to a selected row."""

    def __init__(self, x, device, block=1024):
        x = tensor(x, device, torch.float32)
        self.similarity = x @ x.T  # float32: 27 GB for WildJailbreak's 83k adversarial rows
        self.covered = torch.full((len(x),), -math.inf, dtype=torch.float64, device=device)
        self.block = block

    def add(self, i):
        self.covered = torch.maximum(self.covered, self.similarity[i].double())

    def scores(self, rows):
        covered = self.covered.clamp(min=-1)  # nothing covered yet: every row gains fully
        gain = torch.cat([(self.similarity[rows[start:start + self.block]].double() - covered)
                          .clamp_(min=0).sum(dim=1)
                          for start in range(0, len(rows), self.block)])
        return covered.sum() + gain


class NovelSelect:
    """Score: NovelSum's novelty of a row given the selected rows (novelselect.py)."""

    def __init__(self, x, size, device, neighbors=10, density_power=0.5):
        self.x = tensor(x, device)
        # Density: (mean squared L2 distance to the nearest distinct rows) ** -0.5.
        distinct, inverse = np.unique(x.astype(np.float32), axis=0, return_inverse=True)
        points = tensor(distinct, device)
        norms = (points * points).sum(dim=1)
        k = min(neighbors, len(points) - 1)
        nearest = torch.cat([
            (norms[s:s + 2048].unsqueeze(1) - 2 * points[s:s + 2048] @ points.T + norms)
            .topk(k + 1, dim=1, largest=False).values[:, 1:].clamp_(min=0)
            for s in range(0, len(points), 2048)])
        self.density = (1 / (nearest.mean(dim=1) + 1e-9) ** density_power)[
            tensor(np.asarray(inverse).ravel(), device, torch.int64)]
        # weighted[c, j]: density-weighted distance from row c to the j-th selected row.
        self.weighted = torch.empty(len(x), size, dtype=torch.float64, device=device)
        self.rank_weights = 1 / torch.arange(1, size + 1, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, i):
        self.weighted[:, self.n] = (1 - self.x @ self.x[i]) * (self.density + self.density[i])
        self.n += 1

    def scores(self, rows):
        ordered = self.weighted[rows, :self.n].sort(dim=1).values
        return (ordered * self.rank_weights[:self.n]).mean(dim=1)


# -------------------------------------------------------------------------------- selection

def greedy_select(objective, initial, strata_rows, quotas, rngs, candidates=None,
                  minimize=False, label=""):
    """Grow `initial` until each stratum (sorted row arrays `strata_rows`) holds its quota. Each
    step takes the stratum furthest below its quota, draws a fresh pool of up to `candidates` of
    its rows not yet selected (None: all of them; with its own rng) and adds the best-scoring
    one (ties: the lowest row). Returns the rows and, per added row, objective.value() or its
    score."""
    selected = list(initial)
    for row in selected:
        objective.add(row)
    free = [~np.isin(rows, selected) for rows in strata_rows]
    quotas = np.asarray(quotas)
    room = quotas - np.asarray([(~mask).sum() for mask in free])
    size, scores, start = len(selected) + room.sum(), [], time.time()
    while room.any():
        stratum = int(np.argmax(room / np.maximum(quotas, 1)))
        rows = strata_rows[stratum]
        pool = rows[free[stratum]]
        if candidates is not None and len(pool) > candidates:
            pool = np.sort(rngs[stratum].choice(pool, candidates, replace=False))
        values = torch.as_tensor(objective.scores(pool)).double().cpu().numpy()
        values = -values if minimize else values.copy()
        values[np.isnan(values)] = -math.inf
        best = int(np.argmax(values))
        if values[best] == -math.inf:
            raise SystemExit(f"{label}: no candidate has a defined score at size {len(selected)}")
        row = int(pool[best])
        objective.add(row)
        free[stratum][np.searchsorted(rows, row)] = False
        room[stratum] -= 1
        selected.append(row)
        scores.append(objective.value() if hasattr(objective, "value")
                      else (-values[best] if minimize else values[best]))
        if label and (len(selected) % 100 == 0 or len(selected) == size):
            print(f"  {label}: {len(selected)}/{size}, score {scores[-1]:.6f}, "
                  f"{time.time() - start:.0f}s", flush=True)
    return selected, scores


@njit(cache=True)
def filter_rows(x, order, initial, threshold, limit):
    """Rows of `order` kept in turn: cosine similarity to every kept row below `threshold`."""
    kept = np.empty(len(initial) + limit, dtype=np.int64)
    kept[:len(initial)] = initial
    count = len(initial)
    closest = np.empty(limit)
    for row in order:
        if count == len(kept):
            break
        largest = -np.inf
        for j in range(count):
            similarity = np.dot(x[row], x[kept[j]])
            if similarity > largest:
                largest = similarity
                if largest >= threshold:
                    break
        if largest < threshold:
            closest[count - len(initial)] = largest
            kept[count] = row
            count += 1
    return kept[len(initial):count], closest[:count - len(initial)]


def repr_filter(x, initial, quota, rng):
    """Repr Filter from `initial` up to `quota` rows, at the lowest threshold that fills it
    (bisection)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    order = rng.permutation(len(x))
    order = order[~np.isin(order, initial)]
    initial = np.asarray(initial, dtype=np.int64)
    quota -= len(initial)
    low, high = -1.0, 1.0 + 1e-6
    best = filter_rows(x, order, initial, high, quota)
    if len(best[0]) < quota:
        raise SystemExit("Repr Filter: too few rows with distinct embeddings")
    for _ in range(30):
        middle = (low + high) / 2
        result = filter_rows(x, order, initial, middle, quota)
        if len(result[0]) == quota:
            best, high = result, middle
        else:
            low = middle
    print(f"  Repr Filter threshold: cosine similarity below {high:.4f}", flush=True)
    return initial.tolist() + best[0].tolist(), best[1].tolist()


def kmeans_select(x, initial, quota, n_clusters, seed, rng):
    """An equal number of random rows from each k-means cluster (initial rows count)."""
    clusters = min(n_clusters, len(x))
    labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(x)
    waiting = [[int(row) for row in rng.permutation(np.flatnonzero(labels == cluster))
                if row not in set(initial)] for cluster in range(clusters)]
    held = np.bincount(labels[initial], minlength=clusters) if len(initial) else np.zeros(
        clusters, dtype=np.int64)
    selected, scores = list(initial), []
    share = [quota // clusters + (cluster < quota % clusters) for cluster in range(clusters)]
    # Fill every cluster to its share, then spread what small clusters could not supply.
    for limit in (share, [quota] * clusters):
        progress = True
        while len(selected) < quota and progress:
            progress = False
            for cluster in range(clusters):
                if waiting[cluster] and held[cluster] < limit[cluster] and len(selected) < quota:
                    selected.append(waiting[cluster].pop())
                    scores.append(float(cluster))
                    held[cluster] += 1
                    progress = True
    return selected, scores


def select_baseline(name, x, initial, quota, seed, rng, device):
    """A baseline on one stratum (local row indices): its rows and a score per added row."""
    if name == "random":
        rest = rng.permutation(np.setdiff1d(np.arange(len(x)), initial))[:quota - len(initial)]
        return list(initial) + rest.tolist(), [math.nan] * len(rest)
    if name == "repr_filter":
        return repr_filter(x, initial, quota, rng)
    if name == "kmeans":
        return kmeans_select(x, initial, quota, max(1, quota // cd.EXAMPLES_PER_CLUSTER), seed,
                             rng)
    objective = {"qdit": lambda: FacilityLocation(x, device),
                 "kcenter_greedy": lambda: KCenterGreedy(x, device),
                 "novelselect": lambda: NovelSelect(x, quota, device)}[name]()
    return greedy_select(objective, initial, [np.arange(len(x))], [quota], [rng], label=name)


def metric_select(method, features, initial, strata_rows, quotas, rngs, candidates, seed,
                  device, label):
    """Greedy selection under one metric, computed on all the given strata's rows together;
    returns the rows and the metric on the initial rows, then after each added row."""
    objective, reference = build_objective(method, features, int(sum(quotas)), seed, device)
    selected, added = greedy_select(objective, initial, strata_rows, quotas, rngs, candidates,
                                    minimize="self_bleu" in method, label=label)
    del objective
    first, last = reference(initial), reference(selected)
    # The incremental score must equal the metric from scratch (k-means keeps the best solution
    # it found, which a fresh fit may miss).
    if method != "semantic_cluster_inertia" and not math.isclose(
            added[-1], last, rel_tol=1e-6, abs_tol=1e-9):
        raise SystemExit(f"{label}: final score {added[-1]!r} differs from the metric on the "
                         f"selected rows, {last!r}")
    print(f"  {label}: {first:.6f} -> {added[-1]:.6f}", flush=True)
    return selected, [first] * len(initial) + added


def select(method, features, initial, size, strata, seed, candidates, device, mixed=False):
    """(rows, score per row, order) of one method. Each stratum is selected on its own: a
    metric is the stratum's own metric (its initial rows' value, then the value after each
    added row; order counts within the stratum); a baseline's score is empty for initial rows.
    With mixed (an ablation, metrics only), one metric on the whole subset: each step still
    draws its pool from one stratum, the one furthest below its share."""
    quotas = strata.quotas(size)
    if mixed:
        rows = [strata.rows(stratum) for stratum in range(len(quotas))]
        rngs = [np.random.default_rng((seed, stratum)) for stratum in range(len(quotas))]
        selected, scores = metric_select(method, features, initial, rows, quotas, rngs,
                                         candidates, seed, device, f"{method} (mixed)")
        return selected, scores, [0] * len(initial) + list(range(1, size - len(initial) + 1))
    selected, scores, orders = [], [], []
    for stratum, quota in enumerate(quotas):
        rows = strata.rows(stratum)
        start = [int(row) for row in rows if row in set(initial)]
        label = f"{method} ({strata.names[stratum]})"
        rng = np.random.default_rng((seed, stratum))
        if method in BASELINES:
            local = [int(i) for i in np.flatnonzero(np.isin(rows, start))]
            picked, added = select_baseline(method, features.embeddings("semantic")[rows], local,
                                            quota, seed, rng, device)
            picked, added = [int(rows[i]) for i in picked], [math.nan] * len(start) + added
        else:
            picked, added = metric_select(method, features, start, [rows], [quota], [rng],
                                          candidates, seed, device, label)
        if device == "cuda":
            torch.cuda.empty_cache()
        selected += picked
        scores += added
        orders += [0] * len(start) + list(range(1, quota - len(start) + 1))
    return selected, scores, orders


def build_objective(metric, features, size, seed, device):
    """(objective, metric computed from scratch on pool rows) for a set of `size` rows."""
    n_clusters = max(1, size // cd.EXAMPLES_PER_CLUSTER)
    pick = lambda items: lambda rows: [items[i] for i in rows]
    words, tags = pick(features.words), pick(features.tags)
    if metric == "lexical_hdd":
        return Hdd(features.ngrams("words", 1), device), lambda rows: cd.hdd(words(rows))
    if metric == "lexical_distinct_2":
        return DistinctN(features.ngrams("words", 2)), lambda rows: cd.distinct_n(words(rows), 2)
    if metric.startswith("structural_distinct_"):
        n = int(metric[-1])
        return DistinctN(features.ngrams("tags", n)), lambda rows: cd.distinct_n(tags(rows), n)
    if metric.startswith("structural_self_bleu_"):
        k = int(metric[-1])
        objective = PosSelfBleu([features.ngrams("tags", n) for n in range(1, k + 1)],
                                [len(seq) for seq in features.tags], device)
        return objective, lambda rows: cd.pos_self_bleu(tags(rows), k)[k - 1]
    x = features.embeddings("semantic" if metric.startswith("semantic_") else "structural")
    if metric in ("semantic_vendi", "structural_pos_vendi"):
        return Vendi(x), lambda rows: cd.vendi_score(x[rows])
    if metric == "semantic_distsum_cosine":
        return DistSum(x, device), lambda rows: cd.distsum_cosine(x[rows])
    if metric == "semantic_cluster_inertia":
        return (ClusterInertia(x, n_clusters, seed, device),
                lambda rows: cd.cluster_inertia(x[rows], n_clusters, seed))
    k = int(metric[-1])
    return KnnCosine(x, k, device), lambda rows: cd.knn_cosine_distance(x[rows], k)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=SELECTION_DATASETS, required=True)
    add_pool_argument(parser)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--methods", nargs="+", choices=METRICS + BASELINES,
                        default=list(METRICS + BASELINES))
    parser.add_argument("--mixed_diversity", action="store_true",
                        help="Ablation: each metric on the whole subset, both strata together "
                             "(the pools stay per stratum); metrics only")
    parser.add_argument("--size", type=int, default=600, help="Subset size (default: 600)")
    parser.add_argument("--init_size", type=int, default=50,
                        help="Random rows every method starts from (default: 50)")
    parser.add_argument("--candidates", type=int, default=5000,
                        help="Size of the pool a metric step draws from one stratum (default: 5,000)")
    parser.add_argument("--model", default="qwen2.5_git20k", help="Tokenizer for --max_length")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--cache_dir", type=Path, default=Path("/scratch/zf28/selection_cache"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--num_proc", type=int, default=8, help="POS-tagging processes")
    args = parser.parse_args()
    if not 0 < args.init_size < args.size:
        parser.error("--init_size must be positive and below --size")
    if args.mixed_diversity:
        if set(args.methods) & set(BASELINES) and args.methods != list(METRICS + BASELINES):
            parser.error("--mixed_diversity applies to the metrics only")
        args.methods = [method for method in args.methods if method in METRICS]
    set_num_threads(min(args.threads, get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = {method: selection_csv(args.seed, args.dataset, method, args.size,
                                   args.response_label_filter, args.mixed_diversity)
             for method in args.methods}
    pending = [method for method in args.methods if args.overwrite or not paths[method].exists()]
    print(f"{args.dataset}, seed {args.seed}: {len(pending)} of {len(args.methods)} selections "
          "to run", flush=True)
    if not pending:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_mapping.get(args.model, args.model))
    pool = load_pool(args.dataset, args.response_label_filter, tokenizer, args.max_length)
    strata = Strata(pool["stratum"].tolist())
    initial = strata.initial(args.init_size, args.seed)
    print("Strata (pool / initial / subset): " + ", ".join(
        f"{name} {size:,} / {start} / {final}" for name, size, start, final in zip(
            strata.names, strata.sizes, strata.quotas(args.init_size),
            strata.quotas(args.size))), flush=True)
    for name, start, quota in zip(strata.names, strata.quotas(args.init_size),
                                  strata.quotas(args.size)):
        # KNN-3 needs 4 rows, k-means one per cluster.
        if start < max(4, quota // cd.EXAMPLES_PER_CLUSTER):
            parser.error(f"--init_size gives {name} {start} initial rows; it needs "
                         f"{max(4, quota // cd.EXAMPLES_PER_CLUSTER)}")
    features = Features(pool["prompt"].tolist(), args.cache_dir / args.dataset, args.num_proc)
    for method in pending:
        start = time.time()
        selected, scores, orders = select(method, features, initial, args.size, strata,
                                          args.seed, args.candidates, device,
                                          args.mixed_diversity)
        counts = np.bincount(strata.labels[selected], minlength=len(strata.names))
        if len(set(selected)) != args.size or (counts != strata.quotas(args.size)).any():
            raise SystemExit(f"{method}: {len(set(selected))} rows, strata {counts.tolist()}")
        subset = pool.iloc[selected].rename_axis("row").reset_index().assign(
            order=orders, score=scores)
        path = paths[method]
        path.parent.mkdir(parents=True, exist_ok=True)
        subset[["row", "stratum", "order", "score", *pool.columns.drop("stratum")]].to_csv(
            path.with_name(path.name + ".tmp"), index=False)
        path.with_name(path.name + ".tmp").replace(path)
        print(f"{method}: {time.time() - start:.0f}s; saved {path}", flush=True)


if __name__ == "__main__":
    main()
