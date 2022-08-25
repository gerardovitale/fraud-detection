from logging import Logger
from typing import Any, Dict, List, Union

from imblearn.metrics import geometric_mean_score
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def create_message(metric_id: str, metrics: Union[dict, list]):
    return {'message': {'metric_id': metric_id, 'data': metrics}}


def get_data_metrics(logger: Logger, y: Series, subset: str, experiment_id: str,
                     is_resampled: bool = False) -> List[Dict[str, Any]]:
    metrics = [{
        'class': int(y.value_counts().index[i]),
        'count': int(y.value_counts()[i]),
        'percent': float(y.value_counts()[i] / y.__len__()),
        'subset': subset,
        'is_resampled': is_resampled,
        'exp_id': experiment_id,
    } for i in y.unique()]
    logger.info(create_message('dataframe', metrics))
    return metrics


def get_model_metrics(logger: Logger, y_test: Series, y_pred: Series, experiment_id: str,
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
    logger.info(create_message('model', metrics))
    return metrics
