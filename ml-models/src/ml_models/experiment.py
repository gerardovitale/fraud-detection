from imblearn.base import BaseSampler
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ml_models.config.constants import RANDOM_STATE, TEST_SIZE
from ml_models.config.logger import get_logger
from ml_models.metrics import publish_preprocessing_metrics, publish_model_metric_comparison, get_model_metrics


def execute_experiment(
        experiment_identifier: str,
        X: DataFrame,
        y: Series,
        resampling_strategy: BaseSampler,
        estimator: BaseEstimator,
) -> None:
    logger = get_logger(experiment_identifier)
    logger.debug('Beginning of the experiment')

    # preprocessing metrics before resampling
    publish_preprocessing_metrics(logger, y)

    # resampling data
    logger.debug('Resampling data')
    X_res, y_res = resampling_strategy.fit_resample(X, y)

    # preprocessing metrics after resampling
    publish_preprocessing_metrics(logger, y_res)

    # split dataset into training and testing samples
    # try to add a splitting strategy as a function
    logger.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # preprocessing metrics after splitting
    publish_preprocessing_metrics(logger, y_train, y_test)
    publish_preprocessing_metrics(logger, y_train_res, y_test_res)

    # fit ml model and get predictions based on test samples
    logger.debug('Training ML models')
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    y_pred_res = estimator.fit(X_train_res, y_train_res).predict(X_test_res)

    # get metrics of the models and compare model metrics
    logger.debug('Comparing metrics')
    publish_model_metric_comparison(logger, get_model_metrics(logger, y_test, y_pred),
                                    get_model_metrics(logger, y_test_res, y_pred_res))