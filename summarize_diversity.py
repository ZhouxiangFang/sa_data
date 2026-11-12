"""Tabulate diversity-safety correlations across embedding models, one CSV per training dataset.

Each value is the mean of the Pearson and Spearman coefficients between a
diversity metric and the model-average harmful score, read from
diversity/<model>/csv/<dataset>_diversity_safety_correlations.csv
(written by plot_bars_diversity.py), for the --embedders folders (default: every
embedding-model folder of --diversity_dir that has a csv/ folder). Rows are grouped structural, lexical, then
semantic, and sorted alphabetically within each group. In each group, the
lowest (most negative: more diversity, less harm) value of every model column is
marked **bold** (Markdown). Every metric is listed except POS Self-BLEU-1, which
saturates near 0.999, including those the figures leave out.

Example:
    python sa_data/summarize_diversity.py --datasets wildguardmix aegis gretel
"""

import argparse
from pathlib import Path

import pandas as pd

from helper import TRAIN_DATASETS
from plot_bars_diversity import METRICS, PLOT_METHODS
from plot_bars_overall import DIVERSITY_DIR

GROUP_ORDER = ('structural', 'lexical', 'semantic')
SUMMARY_METRICS = [metric for metric in METRICS if metric != 'structural_self_bleu_1↓']


def embedder_folders(diversity_dir):
    """Embedding-model folders of a diversity root: those with csv/, but not the per-seed
    (seed<seed>/) or summary folders."""
    return sorted(path.parent.name for path in Path(diversity_dir).glob('*/csv')
                  if path.parent.name != 'summary' and not path.parent.name.startswith('seed'))


def correlation_table(diversity_dir, dataset, embedders=None):
    """Metrics x embedding models of mean Pearson/Spearman coefficients, in embedders order."""
    columns = {}
    embedders = embedders or embedder_folders(diversity_dir)
    for embedder in embedders:
        model_dir = diversity_dir / embedder
        path = model_dir / 'csv' / f'{dataset}_diversity_safety_correlations.csv'
        if not path.exists():
            print(f'Warning: missing {path}')
        else:
            table = pd.read_csv(path)
            table = table[(table['model'] == 'Average') & table['method'].isin(PLOT_METHODS)]
            columns[model_dir.name] = table.groupby('metric')['coefficient'].mean()
    if not columns:
        raise FileNotFoundError(f'No {dataset} correlation CSVs under {diversity_dir}')
    return pd.DataFrame(columns).reindex(SUMMARY_METRICS)


def ranked(table):
    """Order the groups, sort each alphabetically, and bold each column's best."""
    blocks = []
    for group in GROUP_ORDER:
        # Natural order, so knn_cosine_10 follows knn_cosine_5.
        block = table[table.index.str.startswith(f'{group}_')].sort_index(
            key=lambda index: index.str.replace(r'\d+', lambda m: m[0].zfill(3), regex=True))
        formatted = block.map(lambda value: f'{value:.3f}' if pd.notna(value) else '')
        for model in block:
            if block[model].notna().any():
                best = block[model].idxmin()
                formatted.loc[best, model] = f'**{formatted.loc[best, model]}**'
        blocks.append(formatted)
    return pd.concat(blocks).rename_axis('metric').reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=list(TRAIN_DATASETS))
    parser.add_argument('--diversity_dir', type=Path, default=DIVERSITY_DIR)
    parser.add_argument('--embedders', nargs='+',
                        help='Embedding-model folders, one column each (default: every one '
                             'in --diversity_dir)')
    parser.add_argument('--output_dir', type=Path, help='Default: <diversity_dir>/summary')
    args = parser.parse_args()
    output_dir = args.output_dir or args.diversity_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        result = ranked(correlation_table(args.diversity_dir, dataset, args.embedders))
        path = output_dir / f'{dataset}_correlation_by_embedding_model.csv'
        result.to_csv(path, index=False)
        print(f'{dataset}:\n{result.to_string(index=False)}\nSaved: {path}\n')


if __name__ == '__main__':
    main()
