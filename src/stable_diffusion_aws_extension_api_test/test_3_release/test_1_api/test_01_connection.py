from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from tenacity import stop_after_delay, retry

logger = logging.getLogger(__name__)


class TestConnectApi:

    @classmethod
    def setup_class(self):

        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_test_connection_get_without_key(self):
        resp = self.api.test_connection()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_2_test_connection_get_with_bad_key(self):
        headers = {'x-api-key': "bad_key"}

        resp = self.api.test_connection(headers=headers)

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    # failed test if all retries failed after 20 seconds
    @retry(stop=stop_after_delay(20))
    def test_3_test_connection_get_success(self):
        headers = {'x-api-key': config.api_key}
        resp = self.api.test_connection(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["message"] == "Success"
