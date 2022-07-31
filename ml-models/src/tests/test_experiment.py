from unittest import TestCase
from unittest.mock import patch, MagicMock, Mock

from ml_models.config.constants import TEST_DATA_PATH
from ml_models.experiment import execute_experiment
from ml_models.load import load_data
from tests.mock_estimator import MockEstimator


class TestExperiment(TestCase):
    def setUp(self) -> None:
        logger_patcher = patch('ml_models.config.logger.logging')
        self.addCleanup(logger_patcher.stop)
        self.mock_logger = logger_patcher.start()

    def test_something(self):
        X, y = load_data(TEST_DATA_PATH)
        resampling_strategy = MagicMock()
        resampling_strategy.fit_resample.return_value = X, y
        estimator = MockEstimator()

        execute_experiment('test experiment', X, y, resampling_strategy, estimator)

        resampling_strategy.fit_resample.assert_called_once()
        self.assertEqual(estimator.fit_call_list.__len__(), 2)
        self.assertEqual(estimator.predict_call_list.__len__(), 2)
