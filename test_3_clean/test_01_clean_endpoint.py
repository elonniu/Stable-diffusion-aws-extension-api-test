from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanEndpoint:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_delete_endpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name_list": [
                f"infer-endpoint-{config.endpoint_name}"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)
        assert resp.status_code == 204
