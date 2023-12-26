from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.enums import InferenceStatus, InferenceType
from stable_diffusion_aws_extension_api_test.utils.helper import upload_with_put, get_inference_job_status_new, \
    delete_inference_jobs

logger = logging.getLogger(__name__)

inference_data = {}


class TestImg2ImgInferenceE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

        global inference_data
        if 'id' in inference_data:
            delete_inference_jobs([inference_data['id']])

    def test_1_img2img_inference_job_create(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "task_type": InferenceType.IMG2IMG.value,
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
            "filters": {}
        }

        resp = self.api.create_inference_new(headers=headers, data=data)
        assert resp.status_code == 200
        global inference_data
        inference_data = resp.json()['data']["inference"]

        assert resp.json()["statusCode"] == 200
        assert inference_data["type"] == InferenceType.IMG2IMG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/img2img_api_param.json")

    def test_2_img2img_inference_job_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.IMG2IMG.value

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_inferences(headers=headers, params=params)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        inferences = resp.json()['data']["inferences"]
        assert inference_data["id"] in [inference["InferenceJobId"] for inference in inferences]

    def test_5_img2img_inference_job_run_and_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.IMG2IMG.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 200
        assert resp.json()['data']["inference"]["status"] == InferenceStatus.INPROGRESS.value

        timeout = datetime.now() + timedelta(minutes=5)

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
            raise Exception("Inference execution timed out after 5 minutes.")
