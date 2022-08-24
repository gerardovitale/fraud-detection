import unittest

from werkzeug.test import Client
from werkzeug.testapp import test_app


class TestMain(unittest.TestCase):

    def test_expose_metric_data(self):
        c = Client(test_app)
        response = c.get("/metric_data.json")

        self.assertEqual(response.status_code, 200)
