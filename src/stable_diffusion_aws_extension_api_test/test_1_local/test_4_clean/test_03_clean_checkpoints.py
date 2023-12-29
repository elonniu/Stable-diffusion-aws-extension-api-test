from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanCheckpoints:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_clean_checkpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_checkpoints(headers=headers)
        ckpts = resp.json()['data']['checkpoints']

        id_list = []
        for ckpt in ckpts:
            if 'params' not in ckpt:
                continue
            if ckpt['params'] and 'message' not in ckpt['params']:
                continue

            if ckpt['params']['message'] == config.ckpt_message:
                id_list.append(ckpt['id'])

        data = {
            "checkpoint_id_list": id_list
        }

        resp = self.api.delete_checkpoints(headers=headers, data=data)
        assert resp.status_code == 204
