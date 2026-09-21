"""Syntactic diversity of alignment samples, overall and per subcategory.

Training samples use exactly the filtering, tokenizer length limit, rounding,
shuffle and seeds used by align.py. CLI samples total 600 examples by default.
Set --harmful_rate below 1 to include WildJailbreak benign examples. To reproduce
an alignment run, use its model, num_train, harmful_rate, benign_data_type,
max_length and seed. Additional rounds use seed + round (default: 10 rounds).

All source-pool prompts, including benign data when used, are POS-tagged before
sampling; tags are reused across groups and rounds. Each metric reports the
mean across rounds, using the same sample for all metrics within a round.
POS sequences (e.g. "DT NN VBZ") are embedded with Qwen3-Embedding-0.6B for
the cosine-kernel Vendi Score. Embeddings are reused across groups and datasets.
Self-BLEU is lower for more diverse syntax; POS n-gram diversity and Vendi Score
are higher. Vendi Score is the effective number of distinct embedded structures.
Test splits remain available for prompt-only analysis, without training filters.

Examples:
    python calculate_diversity.py --datasets wildguardmix aegis
    python calculate_diversity.py --datasets aegis --harmful_rate 0.5
    python calculate_diversity.py --datasets wildguardmix --abbr cyberattack
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
from tqdm.auto import tqdm

from helper import (
    BENIGN_DATA_TYPES,
    build_training_data,
    load_wildjailbreak_benign,
    select_harmful_subcategory,
)

SUPPORTED = {
    'train': ('wildguardmix', 'aegis'),
    'test': ('wildguardmix', 'aegis', 'wildjailbreak', 'ailuminate'),
}
EMBEDDING_MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'


class PosEmbedder:
    """Embed space-separated POS tags once per distinct sequence."""

    def __init__(self, batch_size=32, device=None):
        if batch_size < 1:
            raise ValueError('Embedding batch size must be positive')
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.cache = {}

    def encode(self, pos_seqs):
        texts = [' '.join(seq) for seq in pos_seqs]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            # Delay model loading until embeddings are actually needed.
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    EMBEDDING_MODEL_ID, device=self.device,
                    tokenizer_kwargs={'padding_side': 'left'},
                )
            print(f'  Embedding {len(missing):,} unique POS sequences...')
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


def pos_tag_corpus(texts, num_proc=4, batch_size=1000):
    import spacy

    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])
    return [[token.tag_ for token in doc]
            for doc in nlp.pipe(texts, n_process=num_proc, batch_size=batch_size)]


def pos_ngram_diversity(pos_seqs, max_n=4):
    """Sum of unique/total POS n-gram ratios, without storing duplicate n-grams."""
    score = 0.0
    for n in range(1, max_n + 1):
        total = sum(max(0, len(seq) - n + 1) for seq in pos_seqs)
        if total:
            unique = {gram for seq in pos_seqs for gram in nltk.ngrams(seq, n)}
            score += len(unique) / total
    return score


def syntactic_self_bleu(pos_seqs, max_n=4):
    """Exact NLTK Self-BLEU with method1 smoothing, caching reference counts.

    For each n-gram, the two largest per-sentence counts let us exclude the
    hypothesis without rebuilding counts for every other sentence. Tied maxima
    are retained, so identical sentences still count as each other's references.
    """
    seqs = [seq for seq in pos_seqs if seq]
    if len(seqs) < 2:
        return math.nan
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
        precisions = []
        for n, grams in enumerate(sentence_counts):
            matches = 0
            for gram, count in grams.items():
                first, second = maxima[n][gram]
                reference_count = second if count == first else first
                matches += min(count, reference_count)
            if n == 0 and matches == 0:
                break  # NLTK returns zero when no unigrams match.
            precisions.append((matches or 0.1) / max(1, sum(grams.values())))
        if len(precisions) != max_n:
            scores.append(0.0)
            continue
        reference_length = min(
            (length for length, count in lengths.items() if count > (length == len(seq))),
            key=lambda length: (abs(length - len(seq)), length),
        )
        scores.append(brevity_penalty(reference_length, len(seq)) * math.exp(
            math.fsum(math.log(precision) / max_n for precision in precisions)
        ))
    return sum(scores) / len(scores)


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
    if dataset_name in ('wildguardmix', 'aegis'):
        from helper import to_abbr

    names = {}
    categories = dict.fromkeys(cat for value in df['subcategory']
                               for cat in split_subcategories(value))
    for category in categories:
        abbreviations = (to_abbr(dataset_name, category)
                         if dataset_name in ('wildguardmix', 'aegis')
                         else [category])
        for abbr in abbreviations:
            names.setdefault(abbr, category)
    return names


def compute_dataset_diversity(df, dataset_name, *, tokenizer, num_train=800,
                              harmful_rate=1.0, benign_data_type='vanilla_benign',
                              max_length=4096, max_n=4, n_rounds=10, seed=42,
                              num_proc=4, split='train', abbrs=None, benign_pool=None,
                              pos_embedder=None):
    """Score fixed-size alignment mixtures; skip groups with insufficient data.

    The overall row samples the union of eligible dataset rows, preserving their
    original order and counting multi-label rows once. Each subcategory row
    reproduces a separate align.py run. `count` is its eligible source-pool size;
    `scored_on` includes both source and WildJailbreak benign examples.
    """
    if num_train < 2 or n_rounds < 1 or max_n < 1 or num_proc < 1:
        raise ValueError('num_train >= 2; n_rounds, max_n and num_proc >= 1 required')
    if not 0 <= harmful_rate <= 1 or max_length < 1:
        raise ValueError('harmful_rate must be in [0, 1] and max_length positive')

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
    tags = dict(zip(source_prompts, pos_tag_corpus(source_prompts, num_proc=num_proc)))

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
    if pos_embedder is None:
        pos_embedder = PosEmbedder()
    embeddings = pos_embedder.encode([tags[prompt] for prompt in prompts])
    prompt_indices = {prompt: i for i, prompt in enumerate(prompts)}

    rows = []
    for abbr, (category, count, rounds) in samples.items():
        print(f'  Scoring {category}: {num_train} examples x {n_rounds} round(s)')
        scores = []
        for sample in tqdm(rounds, desc=f'{dataset_name} / {category}', unit='round'):
            seqs = [tags[prompt] for prompt in sample]
            sample_embeddings = embeddings[[prompt_indices[prompt] for prompt in sample]]
            scores.append((syntactic_self_bleu(seqs, max_n),
                           pos_ngram_diversity(seqs, max_n),
                           vendi_score(sample_embeddings)))
        rows.append({
            'subcategory': category, 'abbr': abbr, 'count': count,
            'scored_on': num_train, 'subcategory_count': harmful_count,
            'benign_count': benign_count,
            'self_bleu': round(sum(score[0] for score in scores) / n_rounds, 4),
            'pos_ngram_diversity': round(sum(score[1] for score in scores) / n_rounds, 4),
            'vendi_score': round(sum(score[2] for score in scores) / n_rounds, 4),
        })
    return pd.DataFrame(rows)


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
    parser.add_argument('--max_n', type=int, default=4)
    parser.add_argument('--n_rounds', type=int, default=10,
                        help='Sampling repetitions per metric; report their mean (default: 10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_proc', type=int, default=4)
    parser.add_argument('--embed_batch_size', type=int, default=32,
                        help='Batch size for Qwen POS embeddings (default: 32)')
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
    pos_embedder = PosEmbedder(batch_size=args.embed_batch_size, device=args.embed_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        print(f'Loading {dataset_name} ({args.split})')
        result = compute_dataset_diversity(
            load_safety_dataset(dataset_name, args.split), dataset_name,
            tokenizer=tokenizer, num_train=args.num_train, harmful_rate=args.harmful_rate,
            benign_data_type=args.benign_data_type, max_length=args.max_length,
            max_n=args.max_n, n_rounds=args.n_rounds, seed=args.seed, num_proc=args.num_proc,
            split=args.split, abbrs=args.abbr, benign_pool=benign_pool,
            pos_embedder=pos_embedder,
        )
        print('self_bleu: LOWER = more diverse | '
              'pos_ngram_diversity, vendi_score: HIGHER = more diverse')
        print(result.to_string(index=False))
        output = args.output_dir / f'{dataset_name}_{args.split}_diversity_{args.num_train}_{args.harmful_rate}.csv'
        result.to_csv(output, index=False)
        print(f'Saved: {output}')


if __name__ == '__main__':
    main()
