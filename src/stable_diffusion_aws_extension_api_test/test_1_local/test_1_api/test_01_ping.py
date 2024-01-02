from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestPingApi:

    @classmethod
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_ping_get_without_key(self):
        resp = self.api.ping()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_2_ping_with_bad_key(self):
        headers = {'x-api-key': "bad_key"}

        resp = self.api.ping(headers=headers)

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_3_ping_success(self):
        headers = {'x-api-key': config.api_key}
        resp = self.api.ping(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["message"] == "pong"
