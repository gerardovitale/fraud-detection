import unittest

from tests.client import create_test_client
from tests.expected_outputs import EXPECTED_GRID_BEST_PARAMS

class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self.client = create_test_client()

    def test_endpoints(self):
        test_cases = [
            ('/', 200),
            ('/home', 200),
            ('/directory', 200),
            ('/metric_data.json', 200),
            ('/model_data.json', 200),
            ('/dataframe_data.json', 200),
            ('/not_an_endpoint.json', 404),
            ('/cross_model_data.json', 200),
            ('/grid_cv_results_model_data.json', 200),
            ('/grid_data_per_sampling_strategy.json', 200),
        ]

        for endpoint, code in test_cases:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, code)

    def test_get_grid_best_params(self):
        response = self.client.get('/grid_best_params.json')

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.get_json(), dict)
        self.assertEqual(EXPECTED_GRID_BEST_PARAMS[0].keys(), response.get_json().get('result')[0].keys())
        self.assertEqual(EXPECTED_GRID_BEST_PARAMS[1].keys(), response.get_json().get('result')[-1].keys())
