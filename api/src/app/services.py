import json
from typing import Any, Dict, List, Union

import pandas as pd

MetricData = List[Dict[str, Union[str, Dict]]]


def get_all_metric_data() -> MetricData:
    with open('/app/data/ml_info.log') as metric_file:
        metric_lines = metric_file.readlines()
    return [json.loads(line) for line in metric_lines]


def filter_metric_data_by_experiment_name(data: MetricData, experiment_name: str) -> MetricData:
    return [record for record in data if record['name'] == experiment_name]


def filter_metric_data_by_metric_id(data: MetricData, metric_id: str) -> MetricData:
    return [record for record in data if record['message']['metric_id'] == metric_id]


def get_grid_data_per_sampling_strategy() -> Dict[str, Any]:
    raw_grid_data = filter_metric_data_by_metric_id(
        data=get_all_metric_data(), metric_id='grid_cv_results')
    grid_data = [record['message']['data'] for record in raw_grid_data]

    df = pd.DataFrame()
    for grid_record in grid_data:
        df_grid = pd.DataFrame(grid_record)
        df = pd.concat([df, df_grid], ignore_index=True)

    metrics = [col for col in df.columns if col.startswith('mean_test_')]
    mask = df['param_resampling__sampling_strategy'] != ''
    cols = ['exp_id', 'param_resampling__sampling_strategy'] + metrics

    return df.loc[mask, cols] \
        .groupby(['exp_id', 'param_resampling__sampling_strategy']) \
        .max() \
        .reset_index() \
        .to_dict('split')
