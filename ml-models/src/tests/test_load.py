from unittest import TestCase

from ml_models.config.constants import TEST_DATA_PATH
from ml_models.load import load_data
from pandas import DataFrame, Series


class TestExperiment(TestCase):

    def test_load_data(self):
        actual_X, actual_y = load_data(TEST_DATA_PATH)

        self.assertIsInstance(actual_X, DataFrame)
        self.assertEqual(actual_X.shape[0], 10)
        self.assertEqual(actual_X.shape[1], 30)
        self.assertIsInstance(actual_y, Series)
        self.assertEqual(actual_y.shape.__len__(), 1)
        self.assertEqual(actual_y.shape[0], 10)
        self.assertEqual(list(actual_y.unique()), [0, 1])
