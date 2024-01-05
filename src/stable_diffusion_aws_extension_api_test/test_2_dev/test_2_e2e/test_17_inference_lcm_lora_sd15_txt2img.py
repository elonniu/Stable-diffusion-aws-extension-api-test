from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.enums import InferenceStatus, InferenceType
from stable_diffusion_aws_extension_api_test.utils.helper import upload_multipart_file, wget_file
from stable_diffusion_aws_extension_api_test.utils.helper import upload_with_put, get_inference_job_status_new, \
    delete_inference_jobs

logger = logging.getLogger(__name__)
checkpoint_id = None
signed_urls = None

filename = "sd_xl_turbo_1.0_fp16.safetensors"
checkpoint_type = "Stable-diffusion"
inference_data = {}


class TestTurboE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

        global inference_data
        if 'id' in inference_data:
            delete_inference_jobs([inference_data['id']])

    def test_1_create_turbo_checkpoint(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 7
                }
            ],
            "params": {
                "message": config.ckpt_message,
                "creator": config.username
            }
        }

        resp = self.api.create_checkpoint_new(headers=headers, data=data)

        assert resp.status_code == 201, resp.dumps()

        assert resp.json()["statusCode"] == 201
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type
        assert len(resp.json()['data']["checkpoint"]['id']) == 36

        global checkpoint_id
        checkpoint_id = resp.json()['data']["checkpoint"]['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]

    def test_2_update_turbo_checkpoint(self):
        local_path = f"data/models/Stable-diffusion/{filename}"
        wget_file(
            local_path,
            'https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors',
            'https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/stable-diffusion-aws-extension-github-mainline/models/sd_xl_turbo_1.0_fp16.safetensors'
        )
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

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["checkpoint"]['type'] == checkpoint_type

    def test_3_list_turbo_checkpoint(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_checkpoints(headers=headers, params=params)
        assert resp.status_code == 200, resp.dumps()
        global checkpoint_id
        assert checkpoint_id in [checkpoint["id"] for checkpoint in resp.json()['data']["checkpoints"]]

    def test_4_turbo_txt2img_inference_job_create(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "task_type": InferenceType.TXT2IMG.value,
            "models": {
                "Stable-diffusion": [filename],
                "embeddings": []
            },
            "filters": {
                "createAt": datetime.now().timestamp(),
                "creator": "sd-webui"
            }
        }

        resp = self.api.create_inference_new(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()
        global inference_data
        inference_data = resp.json()['data']["inference"]

        assert resp.json()["statusCode"] == 201
        assert inference_data["type"] == InferenceType.TXT2IMG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/turbo_api_param.json")

    def test_5_turbo_txt2img_inference_job_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.TXT2IMG.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 202, resp.dumps()
        assert resp.json()['data']["inference"]["status"] == InferenceStatus.INPROGRESS.value

        timeout = datetime.now() + timedelta(minutes=4)

        while datetime.now() < timeout:
            status = get_inference_job_status_new(
                api_instance=self.api,
                job_id=inference_id
            )
            logger.info(f"inference is {status}")
            if status == InferenceStatus.SUCCEED.value:
                break
            if status == InferenceStatus.FAILED.value:
                raise Exception("Inference job failed.")
            time.sleep(5)
        else:
            raise Exception("Inference execution timed out after 4 minutes.")
