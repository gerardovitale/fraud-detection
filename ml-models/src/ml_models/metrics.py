from logging import Logger
from numbers import Number
from typing import Any, Dict, List, Union

from imblearn.metrics import geometric_mean_score, specificity_score
from pandas import Series
from sklearn.metrics import (f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)


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
        'specificity': float(specificity_score(y_test, y_pred)),
        'g_mean': float(geometric_mean_score(y_test, y_pred)),
        'roc_auc_score': float(roc_auc_score(y_test, y_pred)),
    }
    logger.info(create_message('model', metrics))
    return metrics


def get_cross_scores():
    return {
        'f1': make_scorer(f1_score, greater_is_better=True),
        'recall': make_scorer(recall_score, greater_is_better=True),
        'precision': make_scorer(precision_score, greater_is_better=True),
        'specificity': make_scorer(specificity_score, greater_is_better=True),
        'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
        'geometric_mean_score': make_scorer(geometric_mean_score, greater_is_better=True)
    }


def cast_scores(scores: Dict[str, Any]) -> Dict[str, List[float]]:
    def cast_score_list(arr: list): 
        return [float(value) if isinstance(value, Number) else value for value in arr]
    return {key: cast_score_list(score_list) for key, score_list in scores.items()}
