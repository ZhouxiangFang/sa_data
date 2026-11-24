"""Diversity of each run's harmful training sample, per training group and seed.

    python calculate_diversity.py --seeds 43 86 129
    python calculate_diversity.py --seeds 43 --response_label_filter refusal --num_train 500 \
        --results_dir ../results/correlations/refusal_500

Scores the very examples each run trained on: helper.harmful_sample reads the seed's sample file
(samples/stage1/seed<seed>/wildguardmix[_refusal]/<abbr>_<n>.jsonl) or draws and saves it
exactly as training would. One row per training group (every subcategory with at least
--num_train eligible rows, and the random and evenly baselines) and seed. --num_train is the
harmful count of the runs (600 for the default 1,800-example mixture).

Metrics (↑: higher = more diverse, ↓: lower = more diverse):
  lexical_hdd↑               HD-D of the pooled words (42 draws; McCarthy & Jarvis, 2010)
  lexical_distinct_2↑        Distinct-2: unique / total word bigrams, pooled over the sample
  semantic_vendi↑            Vendi Score, cosine kernel, prompt embeddings
  semantic_distsum_cosine↑   DistSum: mean cosine distance over pairs (NovelSum)
  semantic_cluster_inertia↑  k-means inertia of the prompt embeddings, one cluster per 50 prompts
  structural_self_bleu_4/5↓  POS Self-BLEU-4/5 (NLTK sentence BLEU, method1 smoothing)
  structural_distinct_5/6↑   POS Distinct-5/6: unique / total POS 5-/6-grams, pooled
  structural_pos_vendi↑      POS-embedding Vendi: Vendi Score of the POS strings' embeddings
  structural_knn_cosine_1/3↑ POS KNN-1/3: mean cosine distance to the 1/3 nearest POS embeddings
Words are spaCy tokens without punctuation or spaces, lowercased; POS sequences are spaCy's
fine-grained tags (en_core_web_sm), joined by spaces for embedding. Embeddings are unit-normalized.
Tags and embeddings cover the whole eligible pool, so no value depends on which seeds or groups
are scored together.

Per embedding model (default BAAI/bge-m3 and Qwen/Qwen3-Embedding-0.6B), in the setting's
scores folder (--results_dir, default results/correlations/default; another --num_train or
--response_label_filter needs its own, as range/common.sh's setting_setup names it):
  <results_dir>/diversity/<embedder>/seed<seed>.csv   one row per group
  <results_dir>/diversity/<embedder>/mean.csv         mean over every seed scored so far (its
                                                      seeds column lists them)
The lexical and POS n-gram metrics use no embedding and are the same in every embedder's CSV.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import brevity_penalty
from scipy.stats import hypergeom
from sklearn import preprocessing
from sklearn.cluster import KMeans

from helper import (BASELINE_ABBRS, DATASET, DEFAULT_SEED, SAMPLES_DIR, STAGE1_DIR,
                    add_pool_argument, eligible_rows, harmful_pool, harmful_sample,
                    load_train, model_mapping, sample_digest, seed_dir, subcategory_stats,
                    training_groups)

EMBEDDING_MODEL_IDS = ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B")
EXAMPLES_PER_CLUSTER = 50  # NovelSum's k-means granularity: 12 clusters for 600 prompts
HDD_DRAWS = 42
# Every metric column; the arrow is the direction of more diversity.
METRICS = (
    "lexical_hdd↑", "lexical_distinct_2↑",
    "semantic_vendi↑", "semantic_distsum_cosine↑", "semantic_cluster_inertia↑",
    "structural_self_bleu_4↓", "structural_self_bleu_5↓",
    "structural_distinct_5↑", "structural_distinct_6↑", "structural_pos_vendi↑",
    "structural_knn_cosine_1↑", "structural_knn_cosine_3↑",
)


def embedding_name(model_id):
    """Results folder of an embedding model, e.g. BAAI/bge-m3 -> bge-m3."""
    return model_id.rstrip("/").split("/")[-1].lower()


def diversity_csv(results_dir, embedder, seed=None):
    """The mean CSV of an embedder in a setting's scores folder, or with seed that seed's CSV."""
    return (Path(results_dir) / "diversity" / embedder
            / (f"{seed_dir(seed)}.csv" if seed is not None else "mean.csv"))


def hdd(seqs, draws=HDD_DRAWS):
    """HD-D of the pooled tokens (lexicalrichness' hdd): each type's probability of appearing
    in `draws` tokens drawn without replacement, summed and divided by `draws`."""
    counts = Counter(token for seq in seqs for token in seq)
    total = sum(counts.values())
    if total < draws:
        return math.nan
    frequencies = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    return float(np.sum(1 - hypergeom.pmf(0, total, frequencies, draws)) / draws)


def distinct_n(seqs, n):
    """Unique / total n-grams in the pooled sample (sequences shorter than n add none)."""
    total = sum(max(0, len(seq) - n + 1) for seq in seqs)
    if not total:
        return math.nan
    return len({gram for seq in seqs for gram in nltk.ngrams(seq, n)}) / total


def pos_self_bleu(seqs, max_n=5):
    """POS Self-BLEU-1..max_n: entry k - 1 is the mean NLTK sentence_bleu of each sequence
    against all others with weights (1 / k,) * k and method1 smoothing.

    Exact, but without rebuilding the references for every hypothesis: for each n-gram the
    two largest per-sequence counts let the hypothesis be left out. Tied maxima are kept, so
    identical sequences still count as each other's references.
    """
    seqs = [seq for seq in seqs if seq]
    if len(seqs) < 2:
        return [math.nan] * max_n
    counts = [[Counter(nltk.ngrams(seq, n)) for n in range(1, max_n + 1)] for seq in seqs]
    maxima = [{} for _ in range(max_n)]
    for sequence_counts in counts:
        for n, grams in enumerate(sequence_counts):
            for gram, count in grams.items():
                first, second = maxima[n].get(gram, (0, 0))
                maxima[n][gram] = (count, first) if count >= first else (first, max(second, count))
    lengths = Counter(map(len, seqs))
    scores = []
    for seq, sequence_counts in zip(seqs, counts):
        log_precisions = []
        for n, grams in enumerate(sequence_counts):
            matches = 0
            for gram, count in grams.items():
                first, second = maxima[n][gram]
                matches += min(count, second if count == first else first)
            if n == 0 and matches == 0:
                break  # NLTK returns zero when no unigram matches.
            log_precisions.append(math.log((matches or 0.1) / max(1, sum(grams.values()))))
        if not log_precisions:
            scores.append([0.0] * max_n)
            continue
        reference_length = min(
            (length for length, count in lengths.items() if count > (length == len(seq))),
            key=lambda length: (abs(length - len(seq)), length))
        penalty = brevity_penalty(reference_length, len(seq))
        scores.append([penalty * math.exp(math.fsum(log_precisions[:k]) / k)
                       for k in range(1, max_n + 1)])
    return [sum(column) / len(scores) for column in zip(*scores)]


def vendi_score(embeddings):
    """exp(entropy of the eigenvalues of K / N), K the cosine similarity matrix: 1 for
    identical embeddings, N for N orthogonal ones (github.com/vertaix/Vendi-Score)."""
    x = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    n, dimension = x.shape
    # The smaller Gram matrix; both share their nonzero eigenvalues.
    gram = x @ x.T if n <= dimension else x.T @ x
    eigenvalues = np.linalg.eigvalsh(gram / n)
    eigenvalues = eigenvalues[eigenvalues > 0]  # tiny negatives are rounding error
    eigenvalues /= eigenvalues.sum()
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def distsum_cosine(embeddings):
    """Mean 1 - cos(x_i, x_j) over the N (N - 1) ordered pairs i != j."""
    x = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    n = len(x)
    total = x.sum(axis=0)
    # The cosines of all pairs sum to ||sum_i x_i||^2, the n self-pairs to n.
    return float(1 - (total @ total - n) / (n * (n - 1)))


def cluster_inertia(embeddings, n_clusters, seed):
    """k-means sum of squared distances to the nearest centroid (best of 10 inits)."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return float(kmeans.fit(np.asarray(embeddings, dtype=np.float64)).inertia_)


def knn_cosine_distance(embeddings, k):
    """Mean cosine distance from each example to its k nearest other examples."""
    x = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    distances = 1 - x @ x.T
    np.fill_diagonal(distances, np.inf)
    return float(np.partition(distances, k - 1, axis=1)[:, :k].clip(min=0).mean())


def text_metrics(words, tags):
    """The metrics that use no embedding, from each prompt's words and POS tags."""
    bleu = pos_self_bleu(tags, 5)
    return {"lexical_hdd↑": hdd(words), "lexical_distinct_2↑": distinct_n(words, 2),
            "structural_self_bleu_4↓": bleu[3], "structural_self_bleu_5↓": bleu[4],
            "structural_distinct_5↑": distinct_n(tags, 5),
            "structural_distinct_6↑": distinct_n(tags, 6)}


def embedding_metrics(semantic, structural, seed):
    """The metrics of the prompt embeddings (semantic) and the POS embeddings (structural)."""
    return {"semantic_vendi↑": vendi_score(semantic),
            "semantic_distsum_cosine↑": distsum_cosine(semantic),
            "semantic_cluster_inertia↑": cluster_inertia(
                semantic, max(1, len(semantic) // EXAMPLES_PER_CLUSTER), seed),
            "structural_pos_vendi↑": vendi_score(structural),
            "structural_knn_cosine_1↑": knn_cosine_distance(structural, 1),
            "structural_knn_cosine_3↑": knn_cosine_distance(structural, 3)}


def tag_texts(texts, num_proc=4):
    """{text: (POS tags, lowercased words without punctuation or spaces)}."""
    import spacy

    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    docs = nlp.pipe(texts, n_process=num_proc, batch_size=1000)
    return {text: ([token.tag_ for token in doc],
                   [token.lower_ for token in doc if not (token.is_punct or token.is_space)])
            for text, doc in zip(texts, docs)}


def embed(model_id, texts, batch_size=32):
    """{text: unit-normalized embedding} with a sentence-transformers model in float32."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, model_kwargs={"torch_dtype": "float32"})
    pooling = [module.get_config_dict() for module in model
               if type(module).__name__ == "Pooling"]
    # Last-token pooling (Qwen3) reads the final position, so pad on the left.
    if pooling and pooling[0].get("pooling_mode_lasttoken"):
        model.tokenizer.padding_side = "left"
    vectors = model.encode(texts, batch_size=batch_size, prompt="", convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    return dict(zip(texts, vectors))


def seed_means(by_seed):
    """One row per group: each metric's mean over its seeds, which `seeds` lists."""
    metrics = [column for column in by_seed if column in METRICS]
    rows = []
    for _, group in by_seed.groupby("abbr", sort=False):
        row = group.iloc[0].drop(["seed", "harmful_sample_sha256", *metrics],
                                 errors="ignore").to_dict()
        row["seeds"] = " ".join(str(int(seed)) for seed in sorted(group["seed"]))
        row.update({metric: round(float(group[metric].mean()), 6) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--num_train", type=int, default=600,
                        help="Harmful examples per run (default: 600, a third of 1,800)")
    add_pool_argument(parser)
    parser.add_argument("--model", default="qwen2.5_git20k",
                        help="Tokenizer that checks prompt lengths, as in training")
    parser.add_argument("--embedding_models", nargs="+", default=list(EMBEDDING_MODEL_IDS))
    parser.add_argument("--samples_dir", type=Path, default=SAMPLES_DIR,
                        help="Folder of the stage1/ samples (default: samples/)")
    parser.add_argument("--results_dir", type=Path,
                        help="The setting's scores folder (default: results/correlations/default)")
    parser.add_argument("--num_proc", type=int, default=8, help="POS-tagging processes")
    args = parser.parse_args()
    if args.results_dir is None:
        if args.num_train != 600 or args.response_label_filter != "safe":
            parser.error("another --num_train or --response_label_filter needs --results_dir")
        args.results_dir = STAGE1_DIR
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_mapping.get(args.model, args.model))
    df = load_train()
    names = {entry["abbr"]: name for name, entry in subcategory_stats(DATASET).items()}
    groups = [*BASELINE_ABBRS, *training_groups(df, args.num_train, args.response_label_filter)]
    samples = {}
    for abbr in groups:
        for seed in args.seeds:
            sample, path, overlong = harmful_sample(
                df, abbr, args.num_train, seed, tokenizer, args.response_label_filter,
                samples_dir=args.samples_dir)
            if overlong:
                print(f"Warning: {abbr}, seed {seed}: {overlong} overlong prompt(s) skipped")
            samples[abbr, seed] = sample["prompt"].tolist(), sample_digest(path)

    pool = list(dict.fromkeys(eligible_rows(df, args.response_label_filter)["prompt"]))
    print(f"POS-tagging the {len(pool):,} distinct prompts of the eligible pool...")
    tagged = tag_texts(pool, args.num_proc)
    pos_strings = list(dict.fromkeys(" ".join(tags) for tags, _ in tagged.values()))
    rows = []
    for (abbr, seed), (prompts, digest) in samples.items():
        rows.append({
            "subcategory": names.get(abbr, abbr), "abbr": abbr, "seed": seed,
            "count": len(harmful_pool(df, abbr, args.num_train, args.response_label_filter)),
            "scored_on": len(prompts), "harmful_sample_sha256": digest,
            **text_metrics([tagged[prompt][1] for prompt in prompts],
                           [tagged[prompt][0] for prompt in prompts]),
        })
    for model_id in args.embedding_models:
        print(f"Embedding {len(pool):,} prompts and {len(pos_strings):,} POS strings "
              f"with {model_id}...")
        vectors = embed(model_id, pool + pos_strings)
        scored = []
        for row, (prompts, _) in zip(rows, samples.values()):
            semantic = np.asarray([vectors[prompt] for prompt in prompts])
            structural = np.asarray([vectors[" ".join(tagged[prompt][0])] for prompt in prompts])
            scored.append({**row, **embedding_metrics(semantic, structural, row["seed"])})
        result = pd.DataFrame(scored)
        result = result[[*result.columns.drop(list(METRICS)), *METRICS]].round(6)
        name = embedding_name(model_id)
        for seed, seed_rows in result.groupby("seed", sort=False):
            path = diversity_csv(args.results_dir, name, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            seed_rows.to_csv(path, index=False)
            print(f"Saved {path}")
        # The mean covers every seed scored so far, so seeds can be scored one job at a time.
        mean_csv = diversity_csv(args.results_dir, name)
        seed_csvs = sorted(mean_csv.parent.glob("seed*.csv"))
        means = seed_means(pd.concat([pd.read_csv(path) for path in seed_csvs],
                                     ignore_index=True))
        mean_csv.parent.mkdir(parents=True, exist_ok=True)
        means.to_csv(mean_csv, index=False)
        print(means.drop(columns=["subcategory"]).to_string(index=False))
        print(f"Saved {mean_csv}")


if __name__ == "__main__":
    main()
