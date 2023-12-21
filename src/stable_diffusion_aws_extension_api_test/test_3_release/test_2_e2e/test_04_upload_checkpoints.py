from __future__ import print_function

import logging

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

    def test_1_checkpoint_v15_post(self):
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

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()["checkpoint"]['id']) == 36

        global checkpoint_id
        checkpoint_id = resp.json()["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()["s3PresignUrl"][filename]

    def test_2_checkpoint_v15_put_with_bad_params(self):
        global checkpoint_id

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "bad_params": {}
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["errorMessage"] == "__init__() got an unexpected keyword argument 'bad_params'"

    def test_3_checkpoint_v15_put(self):
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
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type

    def test_4_checkpoint_v15_check_list(self):
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
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()["checkpoints"]]

    def test_5_checkpoint_cute_post(self):
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

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()["s3PresignUrl"][filename]

    def test_6_checkpoint_cute_put(self):
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
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type

    def test_7_checkpoint_cute_check_list(self):
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
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()["checkpoints"]]

    def test_8_checkpoint_lora_post(self):
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

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()["s3PresignUrl"][filename]

    def test_9_checkpoint_lora_put(self):
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
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type

    def test_10_checkpoint_lora_check_list(self):
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
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()["checkpoints"]]

    def test_11_checkpoint_canny_post(self):
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

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()["s3PresignUrl"][filename]

    def test_12_checkpoint_canny_put(self):
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
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type

    def test_13_checkpoint_canny_check_list(self):
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
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()["checkpoints"]]

    def test_14_checkpoint_openpose_post(self):
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

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()["checkpoint"]['id']) == 36
        global checkpoint_id
        checkpoint_id = resp.json()["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()["s3PresignUrl"][filename]

    def test_15_checkpoint_openpose_put(self):
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
            "checkpoint_id": checkpoint_id,
            "status": "Active",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        resp = self.api.update_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["checkpoint"]['type'] == checkpoint_type

    def test_16_checkpoint_openpose_check_list(self):
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
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()["checkpoints"]]

    def test_17_upload_lora_checkpoint_by_url(self):
        headers = {"x-api-key": config.api_key}
        if config.is_gcr:
            url = "https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/cartoony.safetensors"
        else:
            # todo will use global link
            url = "https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/cartoony.safetensors"

        data = {
            "checkpointType": "Lora",
            "modelUrl": [
                url
            ],
            "params": {
                "creator": config.username,
                "message": "api-test-message"
            }
        }

        resp = self.api.upload_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert 'checkpoint' in resp.json()
        assert resp.json()['checkpoint']["type"] == "Lora"
        assert resp.json()['checkpoint']["status"] == "Active"
