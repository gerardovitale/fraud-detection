from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from ml_models.config.constants import TEST_DATA_PATH, N_JOBS
from ml_models.experiment import exec_exp, exec_cross_exp, exec_grid_exp
from ml_models.load import load_data
from tests.mock_estimator import MockEstimator


class TestExperiment(TestCase):
    def setUp(self) -> None:
        logger_patcher = patch('ml_models.experiment.get_logger')
        pipeline_patcher = patch('ml_models.experiment.Pipeline')
        cross_validate_patcher = patch('ml_models.experiment.cross_validate')
        grid_patcher = patch('ml_models.experiment.GridSearchCV')
        get_data_metrics_patcher = patch('ml_models.experiment.get_data_metrics')
        get_model_metrics_patcher = patch('ml_models.experiment.get_model_metrics')
        get_cross_scores_patcher = patch('ml_models.experiment.get_cross_scores')
        cast_scores_patcher = patch('ml_models.experiment.cast_scores')
        create_message_patcher = patch('ml_models.experiment.create_message')
        self.addCleanup(logger_patcher.stop)
        self.addCleanup(pipeline_patcher.stop)
        self.addCleanup(cross_validate_patcher.stop)
        self.addCleanup(grid_patcher.stop)
        self.addCleanup(get_data_metrics_patcher.stop)
        self.addCleanup(get_model_metrics_patcher.stop)
        self.addCleanup(get_cross_scores_patcher.stop)
        self.addCleanup(cast_scores_patcher.stop)
        self.addCleanup(create_message_patcher.stop)
        self.mock_logger = logger_patcher.start()
        self.mock_pipeline = pipeline_patcher.start()
        self.mock_cross_validate = cross_validate_patcher.start()
        self.mock_grid_search = grid_patcher.start()
        self.mock_get_data_metrics = get_data_metrics_patcher.start()
        self.mock_get_model_metrics = get_model_metrics_patcher.start()
        self.mock_get_cross_scores = get_cross_scores_patcher.start()
        self.mock_cast_scores = cast_scores_patcher.start()
        self.mock_create_message = create_message_patcher.start()

    def test_exec_exp(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        resampling_strategy.fit_resample.return_value = X, y
        estimator = MockEstimator()

        exec_exp('test + experiment', X, y, resampling_strategy, estimator)

        resampling_strategy.fit_resample.assert_called_once()
        self.assertEqual(estimator.fit_call_list.__len__(), 2)
        self.assertEqual(estimator.predict_call_list.__len__(), 2)
        self.assertEqual(self.mock_get_data_metrics.call_count, 4)
        self.assertEqual(self.mock_get_model_metrics.call_count, 2)

    def test_exec_cross_exp(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        estimator = MockEstimator()
        cv_strategy = MagicMock()

        exec_cross_exp(
            'test + experiment', X, y,
            resampling_strategy,
            estimator, cv_strategy)

        self.mock_pipeline.assert_called_once_with(
            [('resampling', resampling_strategy), ('estimator', estimator)])
        self.mock_get_cross_scores.assert_called_once()
        self.mock_cross_validate.assert_called_once()
        self.mock_cast_scores.assert_called_once()
        self.mock_pipeline().fit.assert_called_once()
        self.mock_pipeline().predict.assert_called_once()
        self.mock_get_model_metrics.assert_called_once()

    def test_exec_cross_epx_when_resampling_strategy_is_none(self):
        X, y = load_data(TEST_DATA_PATH)
        estimator = MockEstimator()
        cv_strategy = MagicMock()

        exec_cross_exp(
            'test + experiment', X, y,
            None, estimator, cv_strategy)

        self.mock_pipeline.assert_not_called()
        self.mock_get_cross_scores.assert_called_once()
        self.mock_cross_validate.assert_called_once()
        self.mock_cast_scores.assert_called_once()
        self.assertEqual(estimator.fit_call_list.__len__(), 1)
        self.assertEqual(estimator.predict_call_list.__len__(), 1)
        self.mock_get_model_metrics.assert_called_once()

    def test_exec_grid_esp(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        estimator = MockEstimator()
        cv_strategy = MagicMock()
        grid_params = {
            'test_param': [1, 2, 3]
        }

        exec_grid_exp(
            'test', X, y, resampling_strategy, 
            estimator, cv_strategy, grid_params)

        self.mock_pipeline.assert_called_once_with(
            [('resampling', resampling_strategy), ('estimator', estimator)])
        self.mock_grid_search.assert_called_once()
        self.mock_grid_search().fit.assert_called_once()
        self.mock_pipeline().fit.assert_not_called()
        self.mock_pipeline().predict.assert_not_called()
