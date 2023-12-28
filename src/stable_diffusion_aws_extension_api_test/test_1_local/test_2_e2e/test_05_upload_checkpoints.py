from __future__ import print_function

import logging

import pytest
import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import clear_checkpoint, upload_multipart_file, wget_file

logger = logging.getLogger(__name__)
checkpoint_id = None
signed_urls = None

message = "api-test-message"


class TestCheckPointE2E:

    def setup_class(self):
        self.api = Api(config)
        clear_checkpoint(message)

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

    def test_1_create_checkpoint_v15(self):
        filename = "v1-5-pruned-emaonly.safetensors"
        checkpoint_type = "Stable-diffusion"
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 5
                }
            ],
            "params": {
                "message": message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36

        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    def test_2_update_checkpoint_v15_with_bad_params(self):
        global checkpoint_id

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "status": "Active",
            "name": ""
        }

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 400

    def test_3_update_checkpoint_v15(self):
        filename = "v1-5-pruned-emaonly.safetensors"
        local_path = f"data/models/Stable-diffusion/{filename}"
        wget_file(
            local_path,
            'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/v1-5-pruned-emaonly.safetensors'
        )
        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, local_path)
        checkpoint_type = "Stable-diffusion"

        global checkpoint_id

        data = {
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    def test_4_list_checkpoints_v15_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)
        assert resp.status_code == 200
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_5_create_checkpoint_cute(self):
        checkpoint_type = "Stable-diffusion"
        filename = "LahCuteCartoonSDXL_alpha.safetensors"

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 6
                }
            ],
            "params": {
                "message": message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_6_update_checkpoint_cute(self):
        filename = "LahCuteCartoonSDXL_alpha.safetensors"
        local_path = f"data/models/Stable-diffusion/{filename}"
        wget_file(
            local_path,
            'https://aws-gcr-solutions.s3.amazonaws.com/stable-diffusion-aws-extension-github-mainline/models/LahCuteCartoonSDXL_alpha.safetensors',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/LahCuteCartoonSDXL_alpha.safetensors'
        )
        checkpoint_type = "Stable-diffusion"
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

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_7_list_checkpoints_cute_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)

        assert resp.status_code == 200
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_8_create_checkpoint_lora(self):
        checkpoint_type = "Lora"
        filename = "nendoroid_xl_v7.safetensors"

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 1
                }
            ],
            "params": {
                "message": message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_9_update_checkpoint_lora(self):
        filename = "nendoroid_xl_v7.safetensors"
        local_path = f"data/models/Lora/{filename}"
        wget_file(
            local_path,
            'https://aws-gcr-solutions.s3.amazonaws.com/stable-diffusion-aws-extension-github-mainline/models/nendoroid_xl_v7.safetensors',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/nendoroid_xl_v7.safetensors'
        )
        checkpoint_type = "Lora"
        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, local_path)
        global checkpoint_id

        data = {
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_10_list_checkpoints_lora_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)

        assert resp.status_code == 200
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_11_create_checkpoint_canny(self):
        checkpoint_type = "ControlNet"
        filename = "control_v11p_sd15_canny.pth"

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 2
                }
            ],
            "params": {
                "message": message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_12_update_checkpoint_canny(self):
        filename = "control_v11p_sd15_canny.pth"
        local_path = f"data/models/ControlNet/{filename}"
        wget_file(
            local_path,
            'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/control_v11p_sd15_canny.pth'
        )
        checkpoint_type = "ControlNet"
        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, local_path)
        global checkpoint_id

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_13_list_checkpoints_canny_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)

        assert resp.status_code == 200
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_14_create_checkpoint_openpose(self):
        checkpoint_type = "ControlNet"
        filename = "control_v11p_sd15_openpose.pth"

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 2
                }
            ],
            "params": {
                "message": message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_15_update_checkpoint_openpose(self):
        filename = "control_v11p_sd15_openpose.pth"
        local_path = f"data/models/ControlNet/{filename}"
        wget_file(
            local_path,
            'https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/control_v11p_sd15_openpose.pth'
        )
        checkpoint_type = "ControlNet"
        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, local_path)
        global checkpoint_id

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        resp = self.api.update_checkpoint_new(checkpoint_id=checkpoint_id, headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_16_list_checkpoints_openpose_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)

        assert resp.status_code == 200
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    def test_17_upload_lora_checkpoint_by_url(self):
        headers = {"x-api-key": config.api_key}
        if config.is_gcr:
            url = "https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/cartoony.safetensors"
        else:
            url = "https://raw.githubusercontent.com/elonniu/safetensors/main/cartoony.safetensors"

        data = {
            "checkpoint_type": "Lora",
            "urls": [
                url
            ],
            "params": {
                "creator": config.username,
                "message": "api-test-message"
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 202
        assert 'message' in resp.json()
