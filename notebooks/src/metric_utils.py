from typing import Any, Dict, List

from imblearn.metrics import geometric_mean_score
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def get_data_metrics(y: Series, subset: str, experiment_id: str,
                     is_resampled: bool = False) -> List[Dict[str, Any]]:
    metrics = [{
        'class': int(y.value_counts().index[i]),
        'count': int(y.value_counts()[i]),
        'percent': float(y.value_counts()[i] / y.__len__()),
        'subset': subset,
        'is_resampled': is_resampled,
        'exp_id': experiment_id,
    } for i in y.unique()]
    return metrics


def get_model_metrics(y_test: Series, y_pred: Series, experiment_id: str,
                      is_resampled: bool = False) -> Dict[str, float]:
    metrics = {
        'is_resampled': is_resampled,
        'exp_id': experiment_id,
        'f_score': float(f1_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'g_mean': float(geometric_mean_score(y_test, y_pred)),
        'roc_auc_score': float(roc_auc_score(y_test, y_pred)),
    }
    return metrics


def get_model_metric_comparison(model_metrics: Dict[str, float],
                                model_metrics_res: Dict[str, float]) -> Dict[str, dict]:
    metrics_comparison = {}
    metric_names = set(list(model_metrics.keys()) + list(model_metrics_res.keys()))
    for metric in metric_names:
        metrics_comparison[metric] = {
            'basic_points': model_metrics_res[metric] - model_metrics[metric],
            'percentage': (model_metrics_res[metric] - model_metrics[metric]) / model_metrics[metric]
        }
    return metrics_comparison


def sort_model_metrics(metrics: Dict[str, float], descending=True):
    return sorted(list(metrics.items()), key=lambda tup: tup[1], reverse=descending)
