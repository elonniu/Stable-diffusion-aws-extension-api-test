from __future__ import print_function

import logging

import config as config
from utils.api import Api
from utils.helper import upload_multipart_file, wget_file, get_parts_number

logger = logging.getLogger(__name__)
checkpoint_id = None
signed_urls = None

filename = "cartoony.safetensors"
checkpoint_type = "Lora"
local_path = f"data/models/{checkpoint_type}/{filename}"


class TestCheckPointDeleteE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_clean_checkpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_checkpoints(headers=headers)
        assert resp.status_code == 200, resp.dumps()
        ckpts = resp.json()['data']['items']

        id_list = []
        for ckpt in ckpts:
            if 'params' not in ckpt:
                continue
            if not ckpt['params'] or 'message' not in ckpt['params']:
                continue

            if ckpt['params']['message'] == config.ckpt_message:
                id_list.append(ckpt['id'])

        if len(id_list) == 0:
            logger.info("No checkpoints to clean")
            return

        data = {
            "checkpoint_id_list": id_list
        }

        resp = self.api.delete_checkpoints(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()

    def test_1_create_checkpoint(self):
        wget_file(
            local_path,
            "https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/cartoony.safetensors"
        )

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": get_parts_number(local_path)
                }
            ],
            "params": {
                "message": config.ckpt_message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 201, resp.dumps()
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36

        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    def test_2_update_checkpoint(self):
        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, local_path)

        global checkpoint_id

        data = {
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.update_checkpoint(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200

    def test_3_delete_checkpoints_succeed(self):
        global checkpoint_id

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_id_list": [
                checkpoint_id
            ],
        }

        resp = self.api.delete_checkpoints(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()
