from logging import Logger
from typing import Dict

from numpy import sqrt
from pandas import Series
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def get_metrics(y: Series) -> dict:
    return {
        'magnitude': {
            'rows': y.__len__(),
            'non_fraud': y.value_counts()[0],
            'fraud': y.value_counts()[1],
        },
        'percentage': {
            'non_fraud': y.value_counts()[0] / y.__len__() * 100,
            'fraud': y.value_counts()[1] / y.__len__() * 100,
        }
    }


def publish_preprocessing_metrics(logger: Logger, y: Series, y_val: Series = None) -> None:
    logger.info(get_metrics(y).__str__())
    if getattr(y_val, 'all', False) and isinstance(y_val, Series):
        logger.info(get_metrics(y_val).__str__())


def get_model_metrics(logger: Logger, y_test: Series, y_pred: Series) -> dict:
    metrics = {}
    metrics['f_score'] = f1_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['g_mean'] = sqrt(metrics['recall'] * metrics['precision'])
    metrics['roc_auc_score'] = roc_auc_score(y_test, y_pred)

    logger.info(metrics.__str__())

    return metrics


def publish_model_metric_comparison(logger: Logger, model_metrics: Dict[str, float],
                                    model_metrics_res: Dict[str, float]) -> None:
    result = {}
    metric_names = set(list(model_metrics.keys()) + list(model_metrics_res.keys()))
    for metric in metric_names:
        result[metric] = {}
        result[metric]['basic_points'] = model_metrics_res[metric] - model_metrics[metric]
        result[metric]['percentage'] = (model_metrics_res[metric] - model_metrics[metric]) / model_metrics[metric]

    logger.info(result.__str__())
