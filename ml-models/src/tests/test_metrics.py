import unittest
from unittest.mock import MagicMock, patch

from pandas import Series

from ml_models.metrics import get_data_metrics, get_model_metrics, publish_model_metric_comparison


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        f1_patcher = patch('ml_models.metrics.f1_score')
        recall_patcher = patch('ml_models.metrics.recall_score')
        precision_patcher = patch('ml_models.metrics.precision_score')
        roc_auc_patcher = patch('ml_models.metrics.roc_auc_score')
        self.addCleanup(f1_patcher.stop)
        self.addCleanup(recall_patcher.stop)
        self.addCleanup(precision_patcher.stop)
        self.addCleanup(roc_auc_patcher.stop)
        self.mock_f1 = f1_patcher.start()
        self.mock_recall = recall_patcher.start()
        self.mock_precision = precision_patcher.start()
        self.mock_roc_auc = roc_auc_patcher.start()
        self.mock_logger = MagicMock()

    def test_get_metrics(self):
        test_data1 = Series(data=[0, 1, 0, 0, 1, 1])
        test_data2 = Series(data=[0, 0, 0, 0, 0, 1])
        expected_metrics1 = [
            {'class': 0, 'count': 3, 'percent': 0.5, 'subset': 'original', 'is_resampled': False, 'exp_id': 'ROS + LogReg'},
            {'class': 1, 'count': 3, 'percent': 0.5, 'subset': 'original', 'is_resampled': False, 'exp_id': 'ROS + LogReg'}]
        expected_metrics2 = [
            {'class': 0, 'count': 5, 'percent': 0.8333333333333334, 'subset': 'test', 'is_resampled': True, 'exp_id': 'RUS + LogReg'},
            {'class': 1, 'count': 1, 'percent': 0.16666666666666666, 'subset': 'test', 'is_resampled': True, 'exp_id': 'RUS + LogReg'}]
        expected_list_sum = [
            {'class': 0, 'count': 3, 'percent': 0.5, 'subset': 'original', 'is_resampled': False, 'exp_id': 'ROS + LogReg'},
            {'class': 1, 'count': 3, 'percent': 0.5, 'subset': 'original', 'is_resampled': False, 'exp_id': 'ROS + LogReg'},
            {'class': 0, 'count': 5, 'percent': 0.8333333333333334, 'subset': 'test', 'is_resampled': True, 'exp_id': 'RUS + LogReg'},
            {'class': 1, 'count': 1, 'percent': 0.16666666666666666, 'subset': 'test', 'is_resampled': True, 'exp_id': 'RUS + LogReg'}]

        actual_metrics1 = get_data_metrics(self.mock_logger, test_data1, 'original', 'ROS + LogReg')
        actual_metrics2 = get_data_metrics(self.mock_logger, test_data2, 'test', 'RUS + LogReg', True)

        self.assertEqual(self.mock_logger.info.call_count, 2)
        self.assertEqual(actual_metrics1, expected_metrics1)
        self.assertEqual(actual_metrics2, expected_metrics2)
        self.assertEqual(actual_metrics1 + actual_metrics2, expected_list_sum)

    def test_get_model_metrics(self):
        test_y_test = Series(data=[0, 1, 0, 0, 1, 1])
        test_y_pred = Series(data=[0, 0, 0, 0, 0, 1])
        expected_keys = ['f_score', 'recall', 'precision', 'g_mean', 'roc_auc_score']

        actual_metrics = get_model_metrics(self.mock_logger, test_y_test, test_y_pred)

        self.assertIsInstance(actual_metrics, dict)
        self.mock_f1.assert_called_once()
        self.mock_recall.assert_called_once()
        self.mock_precision.assert_called_once()
        self.mock_roc_auc.assert_called_once()
        self.mock_logger.info.assert_called_once()
        self.assertEqual(expected_keys, list(actual_metrics.keys()))

    def test_publish_model_metric_comparison(self):
        test_metrics = {'f_score': 0.70, 'recall': 0.72, 'precision': 0.65, 'g_mean': 0.67, 'roc_auc_score': 0.80}
        test_metrics_res = {'f_score': 0.95, 'recall': 0.90, 'precision': 0.77, 'g_mean': 0.91, 'roc_auc_score': 0.88}
        expected_comparison = {
            'f_score': {'basic_points': 0.25, 'percentage': 0.35714286},
            'recall': {'basic_points': 0.18, 'percentage': 0.25},
            'precision': {'basic_points': 0.12, 'percentage': 0.18461538},
            'g_mean': {'basic_points': 0.24, 'percentage': 0.35820896},
            'roc_auc_score': {'basic_points': 0.08, 'percentage': 0.1},
        }

        actual_comparison = publish_model_metric_comparison(self.mock_logger, test_metrics, test_metrics_res)

        self.mock_logger.info.assert_called_once()
        for key, value in actual_comparison.items():
            self.assertAlmostEqual(expected_comparison[key]['basic_points'], value['basic_points'])
            self.assertAlmostEqual(expected_comparison[key]['percentage'], value['percentage'])
