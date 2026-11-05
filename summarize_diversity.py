"""Tabulate diversity-safety correlations across embedding models, one CSV per training dataset.

Each value is the mean of the Pearson and Spearman coefficients between a
diversity metric and the model-average harmful score, read from
diversity/<model>/csv/<dataset>_diversity_safety_correlations.csv
(written by plot_bars_diversity.py), for the --embedders folders (default:
bge-m3 and qwen3-embedding-0.6b, the calculate_diversity.py defaults). Rows are grouped structural, lexical, then
semantic, and sorted alphabetically within each group. In each group, the
lowest (most negative: more diversity, less harm) value of every model column is
marked **bold** (Markdown). POS Self-BLEU-1 is left out, as in the figures.

Example:
    python sa_data/summarize_diversity.py --datasets wildguardmix aegis gretel
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_bars_diversity import FIGURE_METRICS, PLOT_METHODS
from plot_bars_overall import DIVERSITY_DIR

GROUP_ORDER = ('structural', 'lexical', 'semantic')
# Folders of calculate_diversity.EMBEDDING_MODEL_IDS, the default embedding models.
DEFAULT_EMBEDDERS = ('bge-m3', 'qwen3-embedding-0.6b')


def correlation_table(diversity_dir, dataset, embedders=DEFAULT_EMBEDDERS):
    """Metrics x embedding models of mean Pearson/Spearman coefficients, in embedders order."""
    columns = {}
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
    return pd.DataFrame(columns).reindex(FIGURE_METRICS)


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
    parser.add_argument('--datasets', nargs='+', default=['aegis', 'gretel', 'wildguardmix'])
    parser.add_argument('--diversity_dir', type=Path, default=DIVERSITY_DIR)
    parser.add_argument('--embedders', nargs='+', default=list(DEFAULT_EMBEDDERS),
                        help='Embedding-model folders, one column each (default: %(default)s)')
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
