"""Tabulate diversity-safety correlations across embedding models, one CSV per training dataset.

Each value is the mean of the Pearson and Spearman coefficients between a
diversity metric and the model-average harmful score, read from
results/diversity/<model>/csv/<dataset>_diversity_safety_correlations.csv
(written by plot_bars_diversity.py). Rows are grouped structural, lexical, then
semantic, and sorted within each group by the mean over embedding models, most
negative (more diversity, less harm) first. In each group, the lowest value of
every model column is marked **bold** (Markdown). POS Self-BLEU-1 is left out,
as in the figures.

Example:
    python sa_data/summarize_diversity.py --datasets wildguardmix aegis gretel
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_bars_diversity import FIGURE_METRICS, PLOT_METHODS, ROOT

GROUP_ORDER = ('structural', 'lexical', 'semantic')


def correlation_table(diversity_dir, dataset):
    """Metrics x embedding models of mean Pearson/Spearman coefficients (bge-m3 first)."""
    columns = {}
    for model_dir in sorted(diversity_dir.iterdir(), key=lambda path: (path.name != 'bge-m3', path.name)):
        path = model_dir / 'csv' / f'{dataset}_diversity_safety_correlations.csv'
        if path.exists():
            table = pd.read_csv(path)
            table = table[(table['model'] == 'Average') & table['method'].isin(PLOT_METHODS)]
            columns[model_dir.name] = table.groupby('metric')['coefficient'].mean()
    if not columns:
        raise FileNotFoundError(f'No {dataset} correlation CSVs under {diversity_dir}')
    return pd.DataFrame(columns).reindex(FIGURE_METRICS)


def ranked(table):
    """Order the groups, sort each by the mean over models, and bold each column's best."""
    blocks = []
    for group in GROUP_ORDER:
        block = table[table.index.str.startswith(f'{group}_')]
        block = block.loc[block.mean(axis=1).sort_values(na_position='last').index]
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
    parser.add_argument('--diversity_dir', type=Path, default=ROOT / 'results' / 'diversity')
    parser.add_argument('--output_dir', type=Path, help='Default: <diversity_dir>/summary')
    args = parser.parse_args()
    output_dir = args.output_dir or args.diversity_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        result = ranked(correlation_table(args.diversity_dir, dataset))
        path = output_dir / f'{dataset}_correlation_by_embedding_model.csv'
        result.to_csv(path, index=False)
        print(f'{dataset}:\n{result.to_string(index=False)}\nSaved: {path}\n')


if __name__ == '__main__':
    main()
