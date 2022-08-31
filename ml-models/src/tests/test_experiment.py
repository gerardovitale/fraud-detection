from unittest import TestCase
from unittest.mock import MagicMock, patch

from ml_models.config.constants import TEST_DATA_PATH, N_JOBS
from ml_models.experiment import exec_exp, exec_cross_exp
from ml_models.load import load_data
from tests.mock_estimator import MockEstimator


class TestExperiment(TestCase):
    def setUp(self) -> None:
        logger_patcher = patch('ml_models.experiment.get_logger')
        pipeline_patcher = patch('ml_models.experiment.Pipeline')
        cross_validate_patcher = patch('ml_models.experiment.cross_validate')
        get_data_metrics_patcher = patch('ml_models.experiment.get_data_metrics')
        get_model_metrics_patcher = patch('ml_models.experiment.get_model_metrics')
        get_cross_scores_patcher = patch('ml_models.experiment.get_cross_scores')
        cast_cross_scores_patcher = patch('ml_models.experiment.cast_cross_scores')
        self.addCleanup(logger_patcher.stop)
        self.addCleanup(pipeline_patcher.stop)
        self.addCleanup(cross_validate_patcher.stop)
        self.addCleanup(get_data_metrics_patcher.stop)
        self.addCleanup(get_model_metrics_patcher.stop)
        self.addCleanup(get_cross_scores_patcher.stop)
        self.addCleanup(cast_cross_scores_patcher.stop)
        self.mock_logger = logger_patcher.start()
        self.mock_pipeline = pipeline_patcher.start()
        self.mock_cross_validate = cross_validate_patcher.start()
        self.mock_get_data_metrics = get_data_metrics_patcher.start()
        self.mock_get_model_metrics = get_model_metrics_patcher.start()
        self.mock_get_cross_scores = get_cross_scores_patcher.start()
        self.mock_cast_cross_scores = cast_cross_scores_patcher.start()

    def test_exec_exp(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        resampling_strategy.fit_resample.return_value = X, y
        estimator = MockEstimator()

        exec_exp('test + experiment', X, y, resampling_strategy, estimator)

        resampling_strategy.fit_resample.assert_called_once()
        self.assertEqual(estimator.fit_call_list.__len__(), 2)
        self.assertEqual(estimator.predict_call_list.__len__(), 2)
        self.assertEqual(self.mock_get_data_metrics.call_count, 6)
        self.assertEqual(self.mock_get_model_metrics.call_count, 2)

    def test_exec_cross_exp(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        resampling_strategy.fit_resample.return_value = X, y
        estimator = MockEstimator()
        cv_strategy = MagicMock()

        exec_cross_exp(
            'test + experiment', X, y, 
            resampling_strategy,
            estimator, cv_strategy)

        self.mock_get_cross_scores.assert_called_once()
        self.mock_pipeline.assert_called_once_with(
            [('resampling', resampling_strategy), ('estimator', estimator)])
        self.mock_cross_validate.assert_called_once_with(
            self.mock_pipeline(), X, y, scoring=self.mock_get_cross_scores(), 
            cv=cv_strategy, n_jobs=N_JOBS)
        self.mock_cast_cross_scores.assert_called_once()
