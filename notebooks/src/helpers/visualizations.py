from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metric_score_variation(
    data: pd.DataFrame, param: str, colors: list,
    xytext_locs: List[Tuple[int, int]], scale_y_axis: bool = True
) -> None:

    pretty_param = param.split('__')[1]
    titles = [
        'F-Score', 'G-Mean', 'Precision',
        'Recall', 'ROC AUC', 'Specificity']

    # get list of metrics to plot
    metrics_to_plot = [
        col.replace('mean_test_', '')
        for col in data.columns
        if col.startswith('mean_test_')]

    # plot metric score variation in relation with a param change by experiment
    fig = plt.figure(figsize=[10, 13])
    plt.suptitle(f'Variación de "{pretty_param}"', fontsize=14)
    plot_params = {'data': data, 'x': param}
    for index, metric in enumerate(metrics_to_plot):
        test_metric = f'mean_test_{metric}'
        train_metric = f'mean_train_{metric}'

        plt.subplot(3, 2, index+1)

        sns.lineplot(
            **plot_params,
            y=test_metric,
            label=f'test',
            color=colors[0])
        ax = sns.lineplot(
            **plot_params,
            y=train_metric,
            label=f'train',
            color=colors[1])

        set_lineplot_annotation(ax, colors, xytext_locs)

        plt.legend().remove()
        plt.title(titles[index], fontsize=14)
        plt.xlabel(pretty_param)
        plt.ylabel('puntuación')
        plt.ylim([0, 1]) if scale_y_axis else None

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(  # title='Legenda'
        handles, labels, loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        ncol=2, fancybox=True, shadow=False,
        facecolor='white', edgecolor='grey')
    plt.tight_layout()


def set_lineplot_annotation(ax, colors: list, xytext_locs: List[Tuple[int, int]] = None) -> None:
    tex_locs = ['top', 'bottom']
    xytext_locs = xytext_locs or [(0, 0), (0, 0)]
    annotate_params = {
        'xytext': (0, 0),
        'textcoords': "offset points",
        'ha': 'center',
        'weight': 'bold',
        'bbox': {
            'boxstyle': 'round,pad=0.3',
            'fc': 'white',
            'alpha': 0.5}}

    for i, line in enumerate(ax.lines):
        annotate_params.update(
            {'xytext': xytext_locs[i], 'va': tex_locs[i], 'color': colors[i]})

        y_max = np.max(line.get_ydata())
        max_index = np.where(line.get_ydata() == y_max)[0][0]
        x_max = line.get_xdata()[max_index]
        ax.annotate(
            '{:.2f}%'.format(y_max*100),
            (x_max, y_max),
            **annotate_params)

        y_min = np.min(line.get_ydata())
        min_index = np.where(line.get_ydata() == y_min)[0][0]
        x_min = line.get_xdata()[min_index]
        ax.annotate(
            '{:.2f}%'.format(y_min*100),
            (x_min, y_min),
            **annotate_params)
