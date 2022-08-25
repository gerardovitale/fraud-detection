import unittest

from app.services import (filter_metric_data_by_experiment_name,
                          filter_metric_data_by_metric_id, get_all_metric_data)


class TestServices(unittest.TestCase):

    def test_get_all_metric_data(self):
        actual_data = get_all_metric_data()

        self.assertIsInstance(actual_data, list)
        self.assertIsInstance(actual_data[0], dict)
        self.assertTrue(actual_data[0].get('message').get('metric_id'))
        self.assertTrue(actual_data[0].get('message').get('data'))

    def test_filter_metric_data_by_experiment_name(self):
        test_cases = [
            ('ROS + LogReg', False),
            ('RUS + LogReg', False),
            ('SMOTE + LogReg', False),
            ('', True),
            ('different name', True),
            (None, True),
        ]
        data = get_all_metric_data()

        for exp_name, is_error in test_cases:
            actual_data = filter_metric_data_by_experiment_name(data, exp_name)
            if is_error:
                self.assertFalse(actual_data)
            else:
                self.assertTrue(actual_data)

    def test_filter_metric_data_by_metric_id(self):
        test_cases = [
            ('model', False),
            ('dataframe', False),
            ('comparison', True),
            ('', True),
            ('something different', True),
            (None, True),
        ]
        data = get_all_metric_data()

        for metric_id, is_error in test_cases:
            actual_data = filter_metric_data_by_metric_id(data, metric_id)
            if is_error:
                self.assertFalse(actual_data)
            else:
                self.assertTrue(actual_data)
