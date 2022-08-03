import json
from logging import Logger
from typing import Any, Dict, List

from imblearn.metrics import geometric_mean_score
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def get_data_metrics(logger: Logger, y: Series, subset: str, res_method: str = None) -> List[Dict[str, Any]]:
    metrics = [{
        'class': int(y.index[i]),
        'count': int(y.value_counts()[i]),
        'percent': float(y.value_counts()[i] / y.__len__()),
        'subset': subset,
        'is_resampled': True if res_method else False,
        'res_method': res_method if res_method else '',
    } for i in y.unique()]
    logger.info(json.dumps(metrics))
    return metrics


def get_model_metrics(logger: Logger, y_test: Series, y_pred: Series) -> Dict[str, float]:
    metrics = {
        'f_score': float(f1_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'g_mean': float(geometric_mean_score(y_test, y_pred)),
        'roc_auc_score': float(roc_auc_score(y_test, y_pred)),
    }
    logger.info(json.dumps(metrics))
    return metrics


def publish_model_metric_comparison(logger: Logger, model_metrics: Dict[str, float],
                                    model_metrics_res: Dict[str, float]) -> Dict[str, dict]:
    metrics_comparison = {}
    metric_names = set(list(model_metrics.keys()) + list(model_metrics_res.keys()))
    for metric in metric_names:
        metrics_comparison[metric] = {
            'basic_points': model_metrics_res[metric] - model_metrics[metric],
            'percentage': (model_metrics_res[metric] - model_metrics[metric]) / model_metrics[metric]
        }
    logger.info(json.dumps(metrics_comparison))
    return metrics_comparison
