from __future__ import print_function

import logging

import config as config
from utils.api import Api

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

        inferences = self.api.list_inferences(headers=headers).json()['data']['inferences']
        for inference in inferences:
            data = {
                "inference_id_list": [inference['InferenceJobId']],
            }

            resp = self.api.delete_inferences(headers=headers, data=data)
            assert resp.status_code == 204, resp.dumps()
