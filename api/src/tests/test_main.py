import unittest

from tests.client import create_test_client


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
        ]

        for endpoint, code in test_cases:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, code)
