#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


def get_metrics_df(data):
    res = []
    for framework_name, framework_data in data.items():
        for metric_name, items in framework_data['metrics'].items():
            for subset_name, values in items.items():
                for x, y in values:
                    res.append({
                        'framework': framework_name,
                        'metric': metric_name,
                        'subset': subset_name,
                        'epoch': x,
                        'value': y,
                    })
    return pd.DataFrame(res)


def get_total_time_df(data):
    res = []
    for framework_name, framework_data in data.items():
        res.append({
            'framework': framework_name,
            'value': framework_data['time_ms_total'],
        })
    return pd.DataFrame(res)


def main():
    data = {}

    results_path = '/tmp/results'
    images_path = '/tmp/images'

    for name in ['keras', 'nn_framework']:
        with open(f'{results_path}/{name}.json') as f:
            data[name] = json.load(f)

    with sns.axes_style("whitegrid"):
        df = get_metrics_df(data)
        metrics = sorted(set(df.metric))
        subsets = sorted(set(df.subset))
        fig, axs = plt.subplots(len(metrics), len(subsets), figsize=(20, 10), squeeze=False)
        for axs_row, metric in zip(axs, metrics):
            for ax, subset in zip(axs_row, subsets):
                sns.lineplot('epoch', 'value', hue='framework', data=df[(df.subset == subset) & (df.metric == metric)],
                             marker='o', ax=ax)
                ax.set(ylabel=metric)
                ax.set_title(f'{metric} ({subset})')
        fig.savefig(f'{images_path}/metrics.png')

    with sns.axes_style("whitegrid"):
        df = get_total_time_df(data).sort_values(['framework']).reset_index(drop=True)
        frameworks = sorted(df['framework'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.barplot('framework', 'value', data=df, ax=ax)
        for index, row in df.iterrows():
            ax.text(frameworks.index(row.framework), row.value, f'{row.value} ms', color='black', ha="center")
        ax.set(ylabel='time ms')
        ax.set_title(f'Total fitting time')
        fig.savefig(f'{images_path}/fitting-time.png')


if __name__ == '__main__':
    main()
