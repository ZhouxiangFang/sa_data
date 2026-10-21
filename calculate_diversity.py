"""Lexical, semantic and structural diversity of alignment samples, per subcategory.

Training samples use exactly the filtering, tokenizer length limit, rounding,
shuffle and seeds used by align.py. CLI samples total 600 examples by default.
Set --harmful_rate below 1 to include WildJailbreak benign examples. To reproduce
an alignment run, use its model, num_train, harmful_rate, benign_data_type,
max_length and seed. Additional rounds use seed + round (default: 10 rounds).

All source-pool prompts, including benign data when used, are tagged with spaCy
before sampling; tags are reused across groups and rounds. Each metric reports
the mean across rounds, using the same sample for all metrics within a round.
The CSV's sample_rounds column records the number of rounds for each row.
Each embedding model in --embedding_models (default: BAAI/bge-m3 and
Qwen/Qwen3-Embedding-0.6B, in float32) gets its own CSV under
<output_dir>/<model name>/csv/, e.g. results/diversity/bge-m3/csv/, beside the
figs/ folder that plot_bars_diversity.py writes. Metrics that use no
embedding are computed once and repeated in every model's CSV. Each model is
loaded once and its embeddings are reused across groups and datasets.
Qwen3-Embedding-4B and -8B results (bfloat16) follow the same layout.

Every metric column is prefixed with the aspect of diversity it measures:
  lexical_     lowercased words (spaCy tokens without punctuation or whitespace)
    hdd               HD-D of the pooled sample (McCarthy & Jarvis, 2010; 42 draws)
    distinct_2        unique / total word bigrams, pooled across examples
  semantic_    unit-normalized embeddings of the raw prompts
    vendi             Vendi Score with a cosine kernel
    distsum_cosine    DistSum / (N (N - 1)): mean 1 - cos(x_i, x_j) over pairs i != j
    cluster_inertia   k-means (--n_clusters) sum of squared distances to centroids
    radius            geometric mean of per-dimension standard deviations
  structural_  spaCy fine-grained POS tag sequences (e.g. "DT NN VBZ")
    self_bleu_1..N    Self-BLEU-n with uniform 1..n-gram weights (N = --max_n)
    distinct_1_4      mean over n = 1..4 of pooled unique / total POS n-grams
    ngram_vendi       Vendi Score with the n-gram kernel (n = 1..4), as in
                      vendi_score.text_utils.ngram_vendi_score
    pos_vendi         Vendi Score, cosine kernel, of embeddings of POS strings
    knn_cosine        mean cosine distance to the k-th nearest POS embedding (--knn_k)
Each metric column ends in ↑ if higher values mean more diversity, or ↓ if lower
values do (Self-BLEU only).
DistSum, cluster inertia, radius and KNN distance follow NovelSum (Yang et al.,
2025), with DistSum averaged rather than summed over pairs. The embedding models
are trained on unit-normalized embeddings and their raw norms carry no trained
meaning, so all metrics use unit vectors. Squared-L2 DistSum would then equal 2 * distsum_cosine,
so it is not reported.
Test splits remain available for prompt-only analysis, without training filters.

Examples:
    python calculate_diversity.py --datasets wildguardmix aegis gretel
    python calculate_diversity.py --datasets aegis --harmful_rate 0.5
    python calculate_diversity.py --datasets wildguardmix --abbr cyberattack
    python calculate_diversity.py --datasets gretel --abbr information_hazards
"""

import argparse
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import brevity_penalty
from scipy import sparse
from scipy.stats import hypergeom
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from helper import (
    BENIGN_DATA_TYPES,
    build_training_data,
    load_wildjailbreak_benign,
    select_harmful_subcategory,
    safety_test_datasets,
    to_abbr,
)

SUPPORTED = {
    'train': ('wildguardmix', 'aegis', 'gretel'),
    'test': tuple(safety_test_datasets),
}
EMBEDDING_MODEL_IDS = ('BAAI/bge-m3', 'Qwen/Qwen3-Embedding-0.6B')
HDD_DRAWS = 42
POS_NGRAM_ORDERS = (1, 2, 3, 4)


def embedding_name(model_id):
    """Results folder for an embedding model, e.g. BAAI/bge-m3 -> bge-m3."""
    return model_id.rstrip('/').split('/')[-1].lower()


class PosEmbedder:
    """Embed POS sequences and raw prompts using one cached text encoder."""

    def __init__(self, batch_size=32, device=None, model_id=EMBEDDING_MODEL_IDS[0],
                 dtype='float32'):
        if batch_size < 1:
            raise ValueError('Embedding batch size must be positive')
        self.batch_size = batch_size
        self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.model = None
        self.cache = {}

    def encode(self, pos_seqs):
        texts = [' '.join(seq) for seq in pos_seqs]
        return self.encode_texts(texts)

    def encode_texts(self, texts):
        """Embed unmodified text, preserving input order and duplicate frequency."""
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            # Delay model loading until embeddings are actually needed.
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.model_id, device=self.device,
                    model_kwargs={'torch_dtype': self.dtype},
                )
                # Last-token pooling (Qwen3) reads the final position, so pad on the
                # left; CLS pooling (BGE-M3) reads the first and keeps right padding.
                pooling = [module.get_config_dict() for module in self.model
                           if type(module).__name__ == 'Pooling']
                # sentence-transformers 5 stores one mode string; older versions use flags.
                if pooling and (pooling[0].get('pooling_mode') == 'lasttoken'
                                or pooling[0].get('pooling_mode_lasttoken')):
                    self.model.tokenizer.padding_side = 'left'
            print(f'  Embedding {len(missing):,} unique texts...')
            embeddings = self.model.encode(
                missing, batch_size=self.batch_size, prompt='',
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.cache.update(zip(missing, embeddings))
        # Repeated sequences must retain their frequency in each scored sample.
        return np.asarray([self.cache[text] for text in texts])


def vendi_score(embeddings):
    """exp(entropy(eigenvalues(K / N))), with K the cosine similarity matrix.

    This is the standard q=1 Vendi Score: identical embeddings score 1, while N
    orthogonal embeddings score N. Compute the smaller of the sample and feature
    Gram matrices; their nonzero eigenvalues agree.
    Definition: https://github.com/vertaix/Vendi-Score
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or not all(embeddings.shape):
        raise ValueError('Vendi Score requires a non-empty 2D embedding array')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Vendi Score requires finite, nonzero embeddings')
    embeddings = embeddings / norms
    n, dimension = embeddings.shape
    gram = embeddings @ embeddings.T if n <= dimension else embeddings.T @ embeddings
    eigenvalues = np.linalg.eigvalsh(gram / n)
    # A cosine Gram matrix is PSD; tiny negative eigenvalues are rounding error.
    eigenvalues = eigenvalues[eigenvalues > 0]
    eigenvalues /= eigenvalues.sum()
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def ngram_vendi_score(seqs, ns=POS_NGRAM_ORDERS):
    """n-gram Vendi Score, as in vendi_score.text_utils.ngram_vendi_score.

    The kernel averages, over n, cosine similarities of n-gram count vectors. As
    in the reference, a sequence shorter than n has a zero vector for that n, and
    the eigenvalues of K / N are not renormalized.
    """
    kernel = np.zeros((len(seqs), len(seqs)))
    for n in ns:
        vocabulary, rows, columns = {}, [], []
        for row, seq in enumerate(seqs):
            for gram in nltk.ngrams(seq, n):
                rows.append(row)
                columns.append(vocabulary.setdefault(gram, len(vocabulary)))
        # Repeated (row, column) entries are summed into n-gram counts.
        counts = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                   shape=(len(seqs), max(1, len(vocabulary))))
        counts = preprocessing.normalize(counts)
        kernel += (counts @ counts.T).toarray()
    eigenvalues = np.linalg.eigvalsh(kernel / (len(ns) * len(seqs)))
    eigenvalues = eigenvalues[eigenvalues > 0]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def distsum_cosine(embeddings):
    """NovelSum's DistSum over N (N - 1) ordered pairs i != j: mean 1 - cos(x_i, x_j)."""
    embeddings = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    n = len(embeddings)
    total = embeddings.sum(axis=0)
    # Summing cos over all pairs gives ||sum_i x_i||^2; the n self-pairs add 1 each.
    return float(1 - (total @ total - n) / (n * (n - 1)))


def cluster_inertia(embeddings, n_clusters, seed):
    """k-means sum of squared distances to the nearest centroid (best of 10 inits)."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return float(kmeans.fit(np.asarray(embeddings, dtype=np.float64)).inertia_)


def radius(embeddings):
    """Geometric mean of per-dimension standard deviations (Lai et al., 2020)."""
    std = np.asarray(embeddings, dtype=np.float64).std(axis=0)
    return 0.0 if (std == 0).any() else float(np.exp(np.log(std).mean()))


def knn_cosine_distance(embeddings, k=1):
    """Mean cosine distance from each example to its k-th nearest other example."""
    embeddings = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    if not 1 <= k < len(embeddings):
        raise ValueError('KNN distance requires 1 <= k < number of examples')
    distances = 1 - embeddings @ embeddings.T
    np.fill_diagonal(distances, np.inf)
    # Identical examples are each other's neighbors at distance zero.
    return float(np.partition(distances, k - 1, axis=1)[:, k - 1].clip(min=0).mean())


def tag_corpus(texts, num_proc=4, batch_size=1000):
    """Return (POS tags, lowercased words) per text; words drop punctuation and spaces."""
    import spacy

    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])
    return [([token.tag_ for token in doc],
             [token.lower_ for token in doc if not (token.is_punct or token.is_space)])
            for doc in nlp.pipe(texts, n_process=num_proc, batch_size=batch_size)]


def hdd(seqs, draws=HDD_DRAWS):
    """HD-D of the pooled tokens, as in lexicalrichness.LexicalRichness.hdd.

    Each type contributes its probability of appearing in `draws` tokens drawn
    without replacement, divided by `draws`.
    """
    counts = Counter(token for seq in seqs for token in seq)
    total = sum(counts.values())
    if total < draws:
        return math.nan
    frequencies = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    return float(np.sum(1 - hypergeom.pmf(0, total, frequencies, draws)) / draws)


def distinct_n(seqs, n):
    """Distinct-n across examples: unique / total n-grams in the pooled sample."""
    total = sum(max(0, len(seq) - n + 1) for seq in seqs)
    if not total:
        return math.nan
    return len({gram for seq in seqs for gram in nltk.ngrams(seq, n)}) / total


def self_bleu(seqs, max_n=5):
    """Exact NLTK Self-BLEU-1..max_n with method1 smoothing, caching reference counts.

    Entry k - 1 is cumulative BLEU-k, i.e. sentence_bleu with weights (1 / k,) * k
    against all other sequences. For each n-gram, the two largest per-sentence
    counts let us exclude the hypothesis without rebuilding counts for every
    other sentence. Tied maxima are retained, so identical sentences still count
    as each other's references.
    """
    seqs = [seq for seq in seqs if seq]
    if len(seqs) < 2:
        return [math.nan] * max_n
    counts = [[Counter(nltk.ngrams(seq, n)) for n in range(1, max_n + 1)]
              for seq in seqs]
    maxima = [{} for _ in range(max_n)]
    for sentence_counts in counts:
        for n, grams in enumerate(sentence_counts):
            for gram, count in grams.items():
                first, second = maxima[n].get(gram, (0, 0))
                maxima[n][gram] = (count, first) if count >= first else (first, max(second, count))

    lengths = Counter(map(len, seqs))
    scores = []
    for seq, sentence_counts in zip(seqs, counts):
        log_precisions = []
        for n, grams in enumerate(sentence_counts):
            matches = 0
            for gram, count in grams.items():
                first, second = maxima[n][gram]
                reference_count = second if count == first else first
                matches += min(count, reference_count)
            if n == 0 and matches == 0:
                break  # NLTK returns zero when no unigrams match.
            log_precisions.append(math.log((matches or 0.1) / max(1, sum(grams.values()))))
        if not log_precisions:
            scores.append([0.0] * max_n)
            continue
        reference_length = min(
            (length for length, count in lengths.items() if count > (length == len(seq))),
            key=lambda length: (abs(length - len(seq)), length),
        )
        penalty = brevity_penalty(reference_length, len(seq))
        scores.append([penalty * math.exp(math.fsum(log_precisions[:k]) / k)
                       for k in range(1, max_n + 1)])
    return [sum(column) / len(scores) for column in zip(*scores)]


def split_subcategories(value):
    """Normalize scalar and multi-label subcategories, including HF list columns."""
    if isinstance(value, str):
        categories = value.split(',')
    elif isinstance(value, Iterable):
        categories = value
    elif value is None or pd.isna(value):
        return ['Benign']
    else:
        categories = [value]
    return [str(cat).strip() for cat in categories if str(cat).strip()] or ['Benign']


def category_names(df, dataset_name):
    """Map abbreviations used by align.py to display names."""
    names = {}
    categories = dict.fromkeys(cat for value in df['subcategory']
                               for cat in split_subcategories(value))
    for category in categories:
        abbreviations = (to_abbr(dataset_name, category)
                         if dataset_name in SUPPORTED['train']
                         else [category])
        for abbr in abbreviations:
            names.setdefault(abbr, category)
    return names


def compute_dataset_diversity(df, dataset_name, *, tokenizer, num_train=800,
                              harmful_rate=1.0, benign_data_type='vanilla_benign',
                              max_length=4096, max_n=5, n_rounds=10, seed=42,
                              num_proc=4, split='train', abbrs=None, benign_pool=None,
                              embedders=None, n_clusters=12, knn_k=1):
    """Score fixed-size alignment mixtures; skip groups with insufficient data.

    The overall row samples the union of eligible dataset rows, preserving their
    original order and counting multi-label rows once. Each subcategory row
    reproduces a separate align.py run. `count` is its eligible source-pool size;
    `scored_on` includes both source and WildJailbreak benign examples.
    Returns one DataFrame per entry of `embedders` ({name: PosEmbedder}), under
    the same name; metrics that use no embedding are computed once and shared.
    """
    if num_train < 2 or n_rounds < 1 or max_n < 1 or num_proc < 1:
        raise ValueError('num_train >= 2; n_rounds, max_n and num_proc >= 1 required')
    if not 0 <= harmful_rate <= 1 or max_length < 1:
        raise ValueError('harmful_rate must be in [0, 1] and max_length positive')
    if not 1 <= n_clusters <= num_train or not 1 <= knn_k < num_train:
        raise ValueError('n_clusters must be in [1, num_train] and knn_k in [1, num_train)')

    names = category_names(df, dataset_name)
    if abbrs:
        missing = set(abbrs).difference(names)
        if missing:
            raise ValueError(f'Unknown subcategories in {dataset_name}: {sorted(missing)}')
        names = {abbr: names[abbr] for abbr in abbrs}

    pools = {}
    if split == 'train':
        for abbr in names:
            pools[abbr] = select_harmful_subcategory(df, dataset_name, abbr)
    else:
        # Test datasets need not provide responses or response safety labels.
        df = df.dropna(subset=['prompt']).copy()
        df['prompt'] = df['prompt'].astype(str).str.strip()
        df = df[df['prompt'] != '']
        df['response'] = ''
        labels = df['subcategory'].apply(split_subcategories)
        for abbr, category in names.items():
            pools[abbr] = df[labels.apply(lambda cats: category in cats)]

    eligible_indices = set().union(*(set(pool.index) for pool in pools.values()))
    overall = df.loc[df.index.isin(eligible_indices)].copy()
    overall['prompt'] = overall['prompt'].astype(str).str.strip()
    groups = [('overall', 'overall', overall)] + [
        (abbr, names[abbr], pools[abbr]) for abbr in sorted(names)
    ]
    harmful_count = int(num_train * harmful_rate + 0.5)
    benign_count = num_train - harmful_count
    if benign_count and benign_pool is None:
        benign_pool = load_wildjailbreak_benign(benign_data_type)
    if benign_count and len(benign_pool) < benign_count:
        raise ValueError(f'WildJailbreak has fewer than {benign_count} benign pairs')

    # Tag the complete source pools before any sampling. Deduplicate tagging
    # work, but preserve repeated prompts when reconstructing scored samples.
    source_prompts = overall['prompt'].tolist()
    if benign_count:
        source_prompts.extend(benign_pool['prompt'].tolist())
    source_prompts = list(dict.fromkeys(source_prompts))
    print(f'  POS-tagging {len(source_prompts):,} unique source-pool prompts...')
    tags, words = {}, {}
    for prompt, (prompt_tags, prompt_words) in zip(
            source_prompts, tag_corpus(source_prompts, num_proc=num_proc)):
        tags[prompt], words[prompt] = prompt_tags, prompt_words

    samples = {}
    for abbr, category, pool in groups:
        args = argparse.Namespace(
            alignment_dataset=dataset_name, abbr=abbr, num_train=num_train,
            harmful_rate=harmful_rate, benign_data_type=benign_data_type,
            max_length=max_length, seed=seed,
        )
        rounds = []
        for round_index in range(n_rounds):
            args.seed = seed + round_index
            try:
                sampled, _, _, _ = build_training_data(
                    args, tokenizer, harmful_pool=pool, benign_pool=benign_pool,
                )
            except ValueError as exc:
                if not str(exc).startswith('Requested '):
                    raise
                print(f'  Skipped {category}: {exc}')
                break
            rounds.append(sampled['prompt'].tolist())
        else:
            samples[abbr] = (category, len(pool), rounds)
    if not samples:
        raise ValueError(f'No group in {dataset_name} can supply {num_train} examples')

    prompts = list(dict.fromkeys(prompt for _, _, rounds in samples.values()
                                for sample in rounds for prompt in sample))
    if any(not tags[prompt] for prompt in prompts):
        raise ValueError('A sampled prompt produced no POS tags; cannot score the full sample')
    if embedders is None:
        embedders = {embedding_name(EMBEDDING_MODEL_IDS[0]): PosEmbedder()}
    prompt_indices = {prompt: i for i, prompt in enumerate(prompts)}

    # Metrics without embeddings are computed once and shared by every embedder.
    shared = {}
    for abbr, (category, count, rounds) in samples.items():
        print(f'  Scoring {category}: {num_train} examples x {n_rounds} round(s)')
        shared[abbr] = []
        for sample in tqdm(rounds, desc=f'{dataset_name} / {category}', unit='round'):
            seqs = [tags[prompt] for prompt in sample]
            sample_words = [words[prompt] for prompt in sample]
            shared[abbr].append({
                'lexical_hdd': hdd(sample_words),
                'lexical_distinct_2': distinct_n(sample_words, 2),
                **{f'structural_self_bleu_{n}': score
                   for n, score in enumerate(self_bleu(seqs, max_n), start=1)},
                'structural_distinct_1_4': float(np.mean(
                    [distinct_n(seqs, n) for n in POS_NGRAM_ORDERS])),
                'structural_ngram_vendi': ngram_vendi_score(seqs),
            })

    results = {}
    for name, embedder in embedders.items():
        print(f'  Embedding metrics with {embedder.model_id}')
        pos_embeddings = embedder.encode([tags[prompt] for prompt in prompts])
        semantic_embeddings = embedder.encode_texts(prompts)
        rows = []
        for abbr, (category, count, rounds) in samples.items():
            scores = []
            for round_index, sample in enumerate(rounds):
                # One index list selects both representations of the same examples,
                # including repeats and any WildJailbreak benign examples.
                sample_indices = [prompt_indices[prompt] for prompt in sample]
                semantic = semantic_embeddings[sample_indices]
                structural = pos_embeddings[sample_indices]
                common = shared[abbr][round_index]
                scores.append({
                    **{key: value for key, value in common.items() if key.startswith('lexical_')},
                    'semantic_vendi': vendi_score(semantic),
                    'semantic_distsum_cosine': distsum_cosine(semantic),
                    'semantic_cluster_inertia': cluster_inertia(
                        semantic, n_clusters, seed + round_index),
                    'semantic_radius': radius(semantic),
                    **{key: value for key, value in common.items() if key.startswith('structural_')},
                    'structural_pos_vendi': vendi_score(structural),
                    'structural_knn_cosine': knn_cosine_distance(structural, knn_k),
                })
            rows.append({
                'subcategory': category, 'abbr': abbr, 'count': count,
                'scored_on': num_train, 'subcategory_count': harmful_count,
                'benign_count': benign_count, 'sample_rounds': len(rounds),
                # Arrows mark whether higher (↑) or lower (↓) values mean more diversity.
                **{metric + ('↓' if 'self_bleu' in metric else '↑'):
                   round(float(np.mean([score[metric] for score in scores])), 6)
                   for metric in scores[0]},
            })
        results[name] = pd.DataFrame(rows)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', type=str.lower, default=['wildguardmix'])
    parser.add_argument('--split', choices=SUPPORTED, default='train')
    parser.add_argument('--abbr', nargs='+', help='Only score these subcategory abbreviations')
    parser.add_argument('--num_train', '--group_size', dest='num_train', type=int, default=600,
                        help='Total examples per group, including benign data (default: 600)')
    parser.add_argument('--harmful_rate', type=float, default=1.0,
                        help='Subcategory fraction; 1 excludes WildJailbreak, 0.5 mixes equally')
    parser.add_argument('--benign_data_type', choices=BENIGN_DATA_TYPES, default='vanilla_benign')
    parser.add_argument('--model', default='qwen2.5-ins', help='Tokenizer model, as in align.py')
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_n', type=int, default=5,
                        help='Report structural Self-BLEU-1 through Self-BLEU-max_n (default: 5)')
    parser.add_argument('--n_clusters', type=int, default=12,
                        help='k-means clusters for semantic cluster inertia; 12 keeps '
                             "NovelSum's 50 examples per cluster at 600 examples (default: 12)")
    parser.add_argument('--knn_k', type=int, default=1,
                        help='Neighbor rank for structural KNN cosine distance (default: 1)')
    parser.add_argument('--n_rounds', type=int, default=10,
                        help='Sampling repetitions per metric; report their mean (default: 10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_proc', type=int, default=4)
    parser.add_argument('--embedding_models', '--embedding_model', dest='embedding_models',
                        nargs='+', default=list(EMBEDDING_MODEL_IDS),
                        help='Sentence-transformers embedding models, one results folder each '
                             f'(default: {" ".join(EMBEDDING_MODEL_IDS)})')
    parser.add_argument('--embed_dtype', choices=('bfloat16', 'float16', 'float32'),
                        default='float32',
                        help='Embedding weights dtype; use bfloat16 for large models such as '
                             'Qwen3-Embedding-8B (default: float32)')
    parser.add_argument('--embed_batch_size', type=int, default=32,
                        help='Batch size for POS and prompt embeddings; lower it for large '
                             'models (default: 32)')
    parser.add_argument('--embed_device', default=None,
                        help='Embedding device, e.g. cuda:0 or cpu (default: auto)')
    parser.add_argument('--output_dir', type=Path, default=Path('../results/diversity'))
    args = parser.parse_args()
    if any(dataset not in SUPPORTED[args.split] for dataset in args.datasets):
        parser.error(f'--split {args.split} supports {SUPPORTED[args.split]}')
    if args.num_train < 2 or min(args.max_length, args.max_n, args.n_rounds, args.num_proc) < 1:
        parser.error('--num_train must be >= 2; length, order, rounds and processes must be positive')
    if not 0 <= args.harmful_rate <= 1:
        parser.error('--harmful_rate must be between 0 and 1')
    if args.embed_batch_size < 1:
        parser.error('--embed_batch_size must be positive')
    if not 1 <= args.n_clusters <= args.num_train or not 1 <= args.knn_k < args.num_train:
        parser.error('--n_clusters must be in [1, num_train] and --knn_k in [1, num_train)')
    if len({embedding_name(model) for model in args.embedding_models}) != len(args.embedding_models):
        parser.error('--embedding_models must have distinct model names')
    return args


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from helper import load_safety_dataset, model_mapping

    tokenizer = AutoTokenizer.from_pretrained(
        model_mapping.get(args.model, args.model), trust_remote_code=True, padding_side='right',
    )
    if tokenizer.chat_template is None:
        raise ValueError('Use an instruct model with a chat template, as required by align.py')
    benign_count = args.num_train - int(args.num_train * args.harmful_rate + 0.5)
    benign_pool = load_wildjailbreak_benign(args.benign_data_type) if benign_count else None
    embedders = {
        embedding_name(model_id): PosEmbedder(
            batch_size=args.embed_batch_size, device=args.embed_device,
            model_id=model_id, dtype=args.embed_dtype,
        )
        for model_id in args.embedding_models
    }
    for dataset_name in args.datasets:
        print(f'Loading {dataset_name} ({args.split})')
        results = compute_dataset_diversity(
            load_safety_dataset(dataset_name, args.split), dataset_name,
            tokenizer=tokenizer, num_train=args.num_train, harmful_rate=args.harmful_rate,
            benign_data_type=args.benign_data_type, max_length=args.max_length,
            max_n=args.max_n, n_rounds=args.n_rounds, seed=args.seed, num_proc=args.num_proc,
            split=args.split, abbrs=args.abbr, benign_pool=benign_pool,
            embedders=embedders, n_clusters=args.n_clusters, knn_k=args.knn_k,
        )
        for name, result in results.items():
            print(f'{name}: ↑ higher = more diverse | ↓ lower = more diverse')
            print(result.to_string(index=False))
            output = (args.output_dir / name / 'csv' /
                      f'{dataset_name}_{args.split}_diversity_{args.num_train}_{args.harmful_rate}.csv')
            output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
            print(f'Saved: {output}')


if __name__ == '__main__':
    main()
