from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


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
            "delete_endpoint_list": [
                f"infer-endpoint-{config.endpoint_name}"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.text == '"Endpoint deleted"'
