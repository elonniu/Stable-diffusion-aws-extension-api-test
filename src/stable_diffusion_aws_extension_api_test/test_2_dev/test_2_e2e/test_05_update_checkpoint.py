from __future__ import print_function

import logging
from time import sleep

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
checkpoint_id = None
signed_urls = None


def ckpt_url():
    if config.is_gcr:
        return "https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/cartoony.safetensors"
    return "https://raw.githubusercontent.com/elonniu/safetensors/main/cartoony.safetensors"


class TestUpdateCheckPointE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_clean_checkpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_checkpoints(headers=headers).json()
        checkpoints = resp['data']["checkpoints"]

        id_list = []
        for checkpoint in checkpoints:
            id_list.append(checkpoint['id'])

        if id_list:
            data = {
                "checkpoint_id_list": id_list
            }
            resp = self.api.delete_checkpoints(headers=headers, data=data)
            assert resp.status_code == 204

    def test_1_upload_lora_checkpoint_by_url(self):
        headers = {"x-api-key": config.api_key}

        data = {
            "checkpoint_type": "Lora",
            "urls": [
                ckpt_url()
            ],
            "params": {
                "creator": config.username,
                "message": config.ckpt_message
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 202
        assert 'message' in resp.json()

    def test_2_checkpoint_unique_by_url(self):
        sleep(5)
        headers = {"x-api-key": config.api_key}

        data = {
            "checkpoint_type": "Lora",
            "urls": [
                ckpt_url()
            ],
            "params": {
                "creator": config.username,
                "message": config.ckpt_message
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 400
        assert 'already exists' in resp.json()['message']

    def test_3_checkpoint_update_name(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        ckpts = self.api.list_checkpoints(headers=headers).json()['data']['checkpoints']
        for ckpt in ckpts:
            if ckpt['name'][0] == 'cartoony.safetensors':
                checkpoint_id = ckpt['id']
                logger.info(f"checkpoint_id: {checkpoint_id}")
                data = {
                    "name": "cartoony"
                }
                resp = self.api.update_checkpoint_new(headers=headers, checkpoint_id=checkpoint_id, data=data)
                assert resp.status_code == 202

    def test_4_checkpoint_update_name_check(self):
        sleep(5)
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        rename = False
        ckpts = self.api.list_checkpoints(headers=headers).json()['data']['checkpoints']
        for ckpt in ckpts:
            if ckpt['name'][0] == 'cartoony':
                rename = True

        assert rename
