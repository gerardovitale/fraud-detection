from typing import Dict, List

from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)

from ml_models.config.constants import N_JOBS, RANDOM_STATE, TEST_SIZE
from ml_models.config.logger import get_logger
from ml_models.metrics import (cast_cross_scores, create_message,
                               get_cross_scores, get_data_metrics,
                               get_model_metrics)


def exec_cross_exp(exp_id: str, X: DataFrame, y: Series, resampling_strategy: BaseSampler,
                   estimator: BaseEstimator, cv_strategy) -> None:
    logger = get_logger(exp_id)
    logger.debug('[{0}] Starting experiment'.format(exp_id))
    is_resampled = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    if resampling_strategy:
        is_resampled = True
        estimator = Pipeline(
            [('resampling', resampling_strategy), ('estimator', estimator)])

    scores: dict = cross_validate(
        estimator, X_train, y_train, scoring=get_cross_scores(), cv=cv_strategy, n_jobs=N_JOBS)
    scores = cast_cross_scores(scores)
    scores.update({'exp_id': exp_id})
    logger.info(create_message('cross', scores))

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    get_model_metrics(logger, y_test, y_pred, exp_id, is_resampled)

    logger.debug('[{0}] Experiment finished'.format(exp_id))


def exec_exp(exp_id: str, X: DataFrame, y: Series, resampling_strategy: BaseSampler,
             estimator: BaseEstimator) -> None:
    logger = get_logger(exp_id)
    logger.debug('[{0}] Starting experiment'.format(exp_id))

    # preprocessing metrics before resampling
    get_data_metrics(logger, y, 'main', exp_id)

    # split dataset into training and testing samples
    # try to add a splitting strategy as a function
    logger.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # resampling data
    logger.debug('Resampling data')
    X_train_res, y_train_res = resampling_strategy.fit_resample(X_train, y_train)

    # preprocessing metrics after splitting
    get_data_metrics(logger, y_train, 'train', exp_id)
    get_data_metrics(logger, y_test, 'test', exp_id)
    get_data_metrics(logger, y_train_res, 'train', exp_id, True)

    # fit ml model and get predictions based on test samples
    logger.debug('Training ML models')
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    y_pred_res = estimator.fit(X_train_res, y_train_res).predict(X_test)

    # get metrics of the models and compare model metrics
    logger.debug('Calculating model metrics')
    get_model_metrics(logger, y_test, y_pred, exp_id)
    get_model_metrics(logger, y_test, y_pred_res, exp_id, True)

    logger.debug('[{0}] Experiment finished'.format(exp_id))
