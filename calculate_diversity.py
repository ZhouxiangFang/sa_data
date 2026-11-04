"""Lexical, semantic and structural diversity of alignment samples, per subcategory.

Each sample is drawn by helper.sample_examples, the code align.py draws its
training examples with, from the same pools and with the same seeds: run n of a
training setting uses seed n * 42 (--runs, default 3: seeds 42, 84, 126, as in
align_run.sh), so the sample scored for seed s holds exactly the harmful prompts
the run with seed s was trained on. CLI samples total 600 examples by default,
the harmful part of the default 1,800-example (1:1:1) mixture; use the run's
harmful example count as --num_train (e.g. 300 for the 900-example tactic runs).
Set --harmful_rate below 1 to also score WildJailbreak benign examples (drawn
as in training, with seed + 1). A sample can depend on the tokenizer only when
a prompt reaches --max_length tokens; that is reported as a warning (no
training prompt is near 4,096 tokens with any of the four models' tokenizers).

All source-pool prompts, including benign data when used, are tagged with spaCy
before sampling; tags are reused across groups and seeds. Every metric of a
seed is computed on that seed's one sample. Per embedding model, each seed's
scores go to diversity/seed<seed>/<model name>/csv/<name>.csv (one row per
group; the values plot_bars_diversity.py pairs with the runs trained on that
seed), and their mean over seeds to diversity/<model name>/csv/<name>.csv, whose
seeds column lists the seeds averaged.

Training splits also get rows for the random and evenly baselines (after the
overall row) unless --no_baselines is given. As in align.py, both draw from the
non-benign subcategories with enough safe responses to train on their own
(helper.baseline_abbrs), or with --group_by tactic from the tactics with at
least --num_train eligible rows (helper.select_tactic): random samples their
pooled prompts uniformly, evenly gives each group the same share. Their
`count` is the pooled pool size.
--merge updates each seed's existing CSV in place: the scored rows replace rows
with the same abbr and every other row is kept (the mean CSV is then recomputed
from the merged seed CSVs), so baselines can be added with --abbr random evenly
--merge without rescoring every subcategory. With --abbr, the overall row (the
union of only the selected pools) is not merged.
Each embedding model in --embedding_models (default: only BAAI/bge-m3 and
Qwen/Qwen3-Embedding-0.6B, in float32) gets its own CSV under
<output_dir>/<model name>/csv/ (default output_dir: diversity/ beside sa_data),
e.g. diversity/bge-m3/csv/ (per seed: diversity/seed42/bge-m3/csv/), beside the
figs/ folder that plot_bars_diversity.py writes. Metrics that use no
embedding are computed once and repeated in every model's CSV. Each model is
loaded once and its embeddings are reused across groups and datasets.
Qwen3-Embedding-4B and -8B are not run by default; pass them with
--embedding_models and --embed_dtype bfloat16, and their results follow the
same layout.

Every metric column is prefixed with the aspect of diversity it measures:
  lexical_     lowercased words (spaCy tokens without punctuation or whitespace)
    hdd               HD-D of the pooled sample (McCarthy & Jarvis, 2010; 42 draws)
    distinct_2..4     unique / total word n-grams (n = 2, 3, 4), pooled across examples
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
    knn_cosine_1..10  mean cosine distance to the k nearest POS embeddings, one column
                      per k in --knn_ks (default: 1, 3, 5, 10)
Each metric column ends in ↑ if higher values mean more diversity, or ↓ if lower
values do (Self-BLEU only).
DistSum, cluster inertia, radius and KNN distance follow NovelSum (Yang et al.,
2025), with DistSum averaged rather than summed over pairs and KNN distance averaged
over the k nearest neighbors (NovelSum's 1-NN distance at k = 1). The embedding models
are trained on unit-normalized embeddings and their raw norms carry no trained
meaning, so all metrics use unit vectors. Squared-L2 DistSum would then equal 2 * distsum_cosine,
so it is not reported.
Test splits remain available for prompt-only analysis, without training filters.

--group_by tactic (gretel train only) scores one row per jailbreak tactic instead
of per subcategory. Rows pass the same training filters as the non-benign
subcategories (harmful prompt, safe response, non-empty pair; multi-label rows
once) and are sampled the same way; tactics that cannot supply --num_train
examples are skipped. Results go to <dataset>_train_tactic_diversity_<N>_<rate>.csv;
its subcategory and abbr columns hold the tactic name and its snake_case form.

Examples:
    python calculate_diversity.py   # wildguardmix, aegis and gretel
    python calculate_diversity.py --abbr random evenly --merge
    python calculate_diversity.py --datasets aegis --harmful_rate 0.5
    python calculate_diversity.py --datasets wildguardmix --abbr cyberattack
    python calculate_diversity.py --datasets gretel --abbr information_hazards
    python calculate_diversity.py --datasets gretel --group_by tactic --num_train 300 --n_clusters 6
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
    BASELINE_ABBRS,
    BENIGN_DATA_TYPES,
    DEFAULT_RUNS,
    DEFAULT_SEED,
    load_wildjailbreak_benign,
    mixture_counts,
    run_seeds,
    sample_examples,
    seed_diversity_csv,
    select_harmful_subcategory,
    select_tactic,
    tactic_pools,
    safety_test_datasets,
    to_abbr,
)

SUPPORTED = {
    'train': ('wildguardmix', 'aegis', 'gretel'),
    'test': tuple(safety_test_datasets),
}
EMBEDDING_MODEL_IDS = ('BAAI/bge-m3', 'Qwen/Qwen3-Embedding-0.6B')
# Training datasets scored by default, as in align_run.sh.
DEFAULT_DATASETS = ('wildguardmix', 'aegis', 'gretel')
# Display names of the training-data baselines (helper.BASELINE_ABBRS).
BASELINE_NAMES = {'random': 'Random baseline', 'evenly': 'Evenly baseline'}
HDD_DRAWS = 42
LEXICAL_DISTINCT_ORDERS = (2, 3, 4)
POS_NGRAM_ORDERS = (1, 2, 3, 4)
KNN_KS = (1, 3, 5, 10)


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
    """Mean cosine distance from each example to its k nearest other examples."""
    embeddings = preprocessing.normalize(np.asarray(embeddings, dtype=np.float64))
    if not 1 <= k < len(embeddings):
        raise ValueError('KNN distance requires 1 <= k < number of examples')
    distances = 1 - embeddings @ embeddings.T
    np.fill_diagonal(distances, np.inf)
    # Identical examples are each other's neighbors at distance zero. After the
    # partition, the first k entries of each row are its k smallest distances.
    return float(np.partition(distances, k - 1, axis=1)[:, :k].clip(min=0).mean())


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


def compute_dataset_diversity(df, dataset_name, *, tokenizer, num_train=600,
                              harmful_rate=1.0, benign_data_type='vanilla_benign',
                              max_length=4096, max_n=5, seeds=None,
                              num_proc=4, split='train', abbrs=None, benign_pool=None,
                              embedders=None, n_clusters=12, knn_ks=KNN_KS,
                              group_by='subcategory', baselines=True):
    """Score the training samples of each group and seed; skip groups with insufficient data.

    `seeds` defaults to helper.run_seeds() (42, 84, 126). For each seed, a group's
    sample is drawn by helper.sample_examples exactly as the align.py run with
    that seed draws its harmful (and benign) examples. The overall row samples
    the union of eligible dataset rows, preserving their original order and
    counting multi-label rows once. `count` is a group's eligible source-pool
    size; `scored_on` includes both source and WildJailbreak benign examples.
    Returns one DataFrame per entry of `embedders` ({name: PosEmbedder}), under
    the same name, with one row per group and seed (seed_means averages them);
    metrics that use no embedding are computed once and shared.
    With group_by='tactic', rows are jailbreak tactics (see tactic_pools) and
    abbrs selects tactic abbreviations. With baselines, training subcategory
    scores also get random and evenly rows that reproduce align.py's baselines.
    """
    seeds = list(run_seeds() if seeds is None else seeds)
    if num_train < 2 or not seeds or max_n < 1 or num_proc < 1:
        raise ValueError('num_train >= 2, at least one seed, and max_n, num_proc >= 1 required')
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Seeds must be distinct: {seeds}')
    if not 0 <= harmful_rate <= 1 or max_length < 1:
        raise ValueError('harmful_rate must be in [0, 1] and max_length positive')
    if not 1 <= n_clusters <= num_train or not all(1 <= k < num_train for k in knn_ks):
        raise ValueError('n_clusters must be in [1, num_train] and each knn_ks in [1, num_train)')

    if group_by not in ('subcategory', 'tactic'):
        raise ValueError(f'Unknown group_by: {group_by!r}')
    if group_by == 'tactic' and split != 'train':
        raise ValueError('group_by="tactic" applies the training filters; use split="train"')
    if group_by == 'tactic':
        names, pools = tactic_pools(df, dataset_name)
    else:
        names, pools = category_names(df, dataset_name), {}
    # Same rounding as align.py; diversity samples have no general instructions.
    harmful_count, benign_count, _ = mixture_counts(num_train, harmful_rate)
    baseline_names = (
        {abbr: BASELINE_NAMES[abbr] for abbr in BASELINE_ABBRS}
        if baselines and split == 'train' and harmful_count
        else {}
    )
    if abbrs:
        missing = set(abbrs).difference(names, baseline_names)
        if missing:
            raise ValueError(f'Unknown {group_by} abbreviations in {dataset_name}: '
                             f'{sorted(missing)}')
        names = {abbr: names[abbr] for abbr in abbrs if abbr in names}
        baseline_names = {abbr: name for abbr, name in baseline_names.items() if abbr in abbrs}

    if group_by == 'tactic':
        pools = {abbr: pools[abbr] for abbr in names}
    elif split == 'train':
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
    # The baseline pools pool the eligible subcategories or tactics (align.py's selection).
    select_baseline = select_tactic if group_by == 'tactic' else select_harmful_subcategory
    baseline_pools = {abbr: select_baseline(df, dataset_name, abbr, harmful_count)
                      for abbr in baseline_names}
    groups = ([('overall', 'overall', overall)] if names else []) + [
        (abbr, name, baseline_pools[abbr]) for abbr, name in baseline_names.items()
    ] + [(abbr, names[abbr], pools[abbr]) for abbr in sorted(names)]
    if benign_count and benign_pool is None:
        benign_pool = load_wildjailbreak_benign(benign_data_type)
    if benign_count and len(benign_pool) < benign_count:
        raise ValueError(f'WildJailbreak has fewer than {benign_count} benign pairs')

    # Tag the complete source pools before any sampling. Deduplicate tagging
    # work, but preserve repeated prompts when reconstructing scored samples.
    source_prompts = overall['prompt'].tolist()
    for pool in baseline_pools.values():
        source_prompts.extend(pool['prompt'].astype(str).str.strip())
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
        runs = []
        for seed in seeds:
            try:
                # The same call align.py makes (through build_training_data).
                parts, info = sample_examples(
                    seed, tokenizer, max_length, harmful_pool=pool,
                    harmful_count=harmful_count, abbr=abbr, dataset_name=dataset_name,
                    group_by=group_by, benign_pool=benign_pool, benign_count=benign_count,
                    benign_data_type=benign_data_type,
                )
            except ValueError as exc:
                if not str(exc).startswith('Requested '):
                    raise
                print(f'  Skipped {category}: {exc}')
                break
            if info['overlong']:
                print(f'  Warning: {category}, seed {seed}: skipped {info["overlong"]:,} '
                      f'prompt(s) of at least {max_length:,} tokens; a model whose '
                      'tokenizer skips different prompts was trained on a different sample')
            runs.append(pd.concat([parts['harmful'], parts['benign']])['prompt'].tolist())
        else:
            samples[abbr] = (category, len(pool), runs)
    if not samples:
        raise ValueError(f'No group in {dataset_name} can supply {num_train} examples')

    prompts = list(dict.fromkeys(prompt for _, _, runs in samples.values()
                                for sample in runs for prompt in sample))
    if any(not tags[prompt] for prompt in prompts):
        raise ValueError('A sampled prompt produced no POS tags; cannot score the full sample')
    if embedders is None:
        embedders = {embedding_name(EMBEDDING_MODEL_IDS[0]): PosEmbedder()}
    prompt_indices = {prompt: i for i, prompt in enumerate(prompts)}

    # Metrics without embeddings are computed once and shared by every embedder.
    shared = {}
    for abbr, (category, count, runs) in samples.items():
        print(f'  Scoring {category}: {num_train} examples x {len(seeds)} seed(s)')
        shared[abbr] = []
        for sample in tqdm(runs, desc=f'{dataset_name} / {category}', unit='seed'):
            seqs = [tags[prompt] for prompt in sample]
            sample_words = [words[prompt] for prompt in sample]
            shared[abbr].append({
                'lexical_hdd': hdd(sample_words),
                **{f'lexical_distinct_{n}': distinct_n(sample_words, n)
                   for n in LEXICAL_DISTINCT_ORDERS},
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
        for abbr, (category, count, runs) in samples.items():
            for seed, sample, common in zip(seeds, runs, shared[abbr]):
                # One index list selects both representations of the same examples,
                # including repeats and any WildJailbreak benign examples.
                sample_indices = [prompt_indices[prompt] for prompt in sample]
                semantic = semantic_embeddings[sample_indices]
                structural = pos_embeddings[sample_indices]
                score = {
                    **{key: value for key, value in common.items() if key.startswith('lexical_')},
                    'semantic_vendi': vendi_score(semantic),
                    'semantic_distsum_cosine': distsum_cosine(semantic),
                    'semantic_cluster_inertia': cluster_inertia(semantic, n_clusters, seed),
                    'semantic_radius': radius(semantic),
                    **{key: value for key, value in common.items() if key.startswith('structural_')},
                    'structural_pos_vendi': vendi_score(structural),
                    **{f'structural_knn_cosine_{k}': knn_cosine_distance(structural, k)
                       for k in knn_ks},
                }
                rows.append({
                    'subcategory': category, 'abbr': abbr, 'seed': seed, 'count': count,
                    'scored_on': num_train, 'subcategory_count': harmful_count,
                    'benign_count': benign_count,
                    # Arrows mark whether higher (↑) or lower (↓) values mean more diversity.
                    **{metric + ('↓' if 'self_bleu' in metric else '↑'): round(float(value), 6)
                       for metric, value in score.items()},
                })
        results[name] = pd.DataFrame(rows)
    return results


def is_metric(column):
    """Metric columns end in the ↑/↓ direction of more diversity."""
    return column.endswith(('↑', '↓'))


def seed_means(by_seed):
    """One row per group: each metric's mean over the group's seeds, listed in `seeds`."""
    metrics = [column for column in by_seed if is_metric(column)]
    rows = []
    for _, group in by_seed.groupby('abbr', sort=False):
        row = group.iloc[0].drop(['seed', *metrics]).to_dict()
        row['seeds'] = ' '.join(str(int(seed)) for seed in sorted(group['seed']))
        row.update({metric: round(float(group[metric].mean()), 6) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def merge_rows(existing, scored, keep_overall=False):
    """Replace existing rows by abbr with scored rows and keep all other rows.

    Rows are ordered overall, then the baselines, then the remaining rows in
    their existing order, with new subcategories appended alphabetically.
    """
    if list(existing.columns) != list(scored.columns):
        raise ValueError('Cannot merge: the existing CSV has different columns; '
                         'rerun without --merge')
    if keep_overall:
        scored = scored.loc[scored['abbr'] != 'overall']
    rows = {row['abbr']: row for row in existing.to_dict('records')}
    rows.update((row['abbr'], row) for row in scored.to_dict('records'))
    first = ['overall', *BASELINE_ABBRS]
    order = [abbr for abbr in first if abbr in rows]
    order += [abbr for abbr in existing['abbr'] if abbr not in first]
    order += sorted(abbr for abbr in rows if abbr not in order)
    return pd.DataFrame([rows[abbr] for abbr in order], columns=existing.columns)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', type=str.lower, default=list(DEFAULT_DATASETS),
                        help=f'Datasets to score (default: {" ".join(DEFAULT_DATASETS)})')
    parser.add_argument('--split', choices=SUPPORTED, default='train')
    parser.add_argument('--abbr', nargs='+',
                        help='Only score these subcategory (or, with --group_by tactic, '
                             'tactic) abbreviations; random and evenly select the baselines')
    parser.add_argument('--no_baselines', dest='baselines', action='store_false',
                        help='Leave out the random and evenly baseline rows (default: '
                             'include them for training subcategories)')
    parser.add_argument('--merge', action='store_true',
                        help="Replace only the scored rows of each seed's existing CSV and keep "
                             'the rest (the overall row is kept when --abbr is given)')
    parser.add_argument('--group_by', choices=('subcategory', 'tactic'), default='subcategory',
                        help='Score one row per subcategory, or per jailbreak tactic '
                             '(gretel train only) (default: subcategory)')
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
    parser.add_argument('--knn_ks', '--knn_k', dest='knn_ks', type=int, nargs='+',
                        default=list(KNN_KS),
                        help='Neighbor counts for structural KNN cosine distance, one column '
                             f'each (default: {" ".join(map(str, KNN_KS))})')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help='Training runs to score per group; run n uses seed n * --seed, '
                             f'as in align_run.sh (default: {DEFAULT_RUNS})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Seed of run 1 (default: {DEFAULT_SEED})')
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
    parser.add_argument('--output_dir', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'diversity',
                        help='Diversity root, independent of the working directory '
                             '(default: diversity/ beside sa_data)')
    args = parser.parse_args()
    if any(dataset not in SUPPORTED[args.split] for dataset in args.datasets):
        parser.error(f'--split {args.split} supports {SUPPORTED[args.split]}')
    if args.group_by == 'tactic' and (args.split != 'train' or args.datasets != ['gretel']):
        parser.error('--group_by tactic supports only --datasets gretel --split train')
    if args.num_train < 2 or min(args.max_length, args.max_n, args.runs, args.num_proc) < 1:
        parser.error('--num_train must be >= 2; length, order, runs and processes must be positive')
    try:
        args.seeds = run_seeds(args.runs, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.harmful_rate <= 1:
        parser.error('--harmful_rate must be between 0 and 1')
    if args.embed_batch_size < 1:
        parser.error('--embed_batch_size must be positive')
    if (not 1 <= args.n_clusters <= args.num_train
            or not all(1 <= k < args.num_train for k in args.knn_ks)):
        parser.error('--n_clusters must be in [1, num_train] and each --knn_ks in [1, num_train)')
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
    _, benign_count, _ = mixture_counts(args.num_train, args.harmful_rate)
    benign_pool = load_wildjailbreak_benign(args.benign_data_type) if benign_count else None
    embedders = {
        embedding_name(model_id): PosEmbedder(
            batch_size=args.embed_batch_size, device=args.embed_device,
            model_id=model_id, dtype=args.embed_dtype,
        )
        for model_id in args.embedding_models
    }
    grouping = 'tactic_' if args.group_by == 'tactic' else ''
    for dataset_name in args.datasets:
        outputs = {
            name: args.output_dir / name / 'csv' / (
                f'{dataset_name}_{args.split}_{grouping}diversity_'
                f'{args.num_train}_{args.harmful_rate}.csv')
            for name in embedders
        }
        for output in outputs.values():
            # Check before scoring: older CSVs hold means over seeds 42-51 only.
            missing = [seed for seed in args.seeds
                       if not seed_diversity_csv(output, seed).exists()]
            if args.merge and output.exists() and missing:
                raise SystemExit(f'{output} has no per-seed CSV for seed(s) {missing} '
                                 f'(e.g. {seed_diversity_csv(output, missing[0])}), so it '
                                 'cannot be merged; rerun without --merge')
        print(f'Loading {dataset_name} ({args.split}); seeds {" ".join(map(str, args.seeds))}')
        results = compute_dataset_diversity(
            load_safety_dataset(dataset_name, args.split), dataset_name,
            tokenizer=tokenizer, num_train=args.num_train, harmful_rate=args.harmful_rate,
            benign_data_type=args.benign_data_type, max_length=args.max_length,
            max_n=args.max_n, seeds=args.seeds, num_proc=args.num_proc,
            split=args.split, abbrs=args.abbr, benign_pool=benign_pool,
            embedders=embedders, n_clusters=args.n_clusters, knn_ks=args.knn_ks,
            group_by=args.group_by, baselines=args.baselines,
        )
        for name, result in results.items():
            output = outputs[name]
            per_seed = []
            for seed, rows in result.groupby('seed', sort=False):
                seed_output = seed_diversity_csv(output, seed)
                seed_output.parent.mkdir(parents=True, exist_ok=True)
                if args.merge and seed_output.exists():
                    rows = merge_rows(pd.read_csv(seed_output), rows.reset_index(drop=True),
                                      keep_overall=bool(args.abbr))
                rows.to_csv(seed_output, index=False)
                print(f'Saved: {seed_output}')
                per_seed.append(rows)
            means = seed_means(pd.concat(per_seed, ignore_index=True))
            print(f'{name} (mean over seeds): ↑ higher = more diverse | ↓ lower = more diverse')
            print(means.to_string(index=False))
            output.parent.mkdir(parents=True, exist_ok=True)
            means.to_csv(output, index=False)
            print(f'Saved: {output}')


if __name__ == '__main__':
    main()
