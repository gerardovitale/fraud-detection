import json
from typing import Any, Dict, List, Union

MetricData = List[Dict[str, Union[str, Dict]]]


def get_all_metric_data() -> MetricData:
    with open('/app/data/ml_info.log') as metric_file:
        metric_lines = metric_file.readlines()
    return [json.loads(line) for line in metric_lines]


def filter_metric_data_by_experiment_name(data: MetricData, experiment_name: str) -> MetricData:
    return [record for record in data if record['name'] == experiment_name]


def filter_metric_data_by_metric_id(data: MetricData, metric_id: str) -> MetricData:
    return [record for record in data if record['message']['metric_id'] == metric_id]
