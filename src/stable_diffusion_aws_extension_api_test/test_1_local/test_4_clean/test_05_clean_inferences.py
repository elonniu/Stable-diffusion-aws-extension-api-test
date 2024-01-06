from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanInferences:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_clean_inferences(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        items = self.api.list_inferences(headers=headers).json()['data']['items']
        for item in items:

            data = {
                "inference_id_list": [item['id']],
            }

            resp = self.api.delete_inferences(headers=headers, data=data)
            assert resp.status_code == 204
