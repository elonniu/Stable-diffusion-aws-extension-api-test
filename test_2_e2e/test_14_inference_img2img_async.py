from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import config as config
from utils.api import Api
from utils.enums import InferenceStatus, InferenceType
from utils.helper import upload_with_put, get_inference_job_status, get_inference_job_image

logger = logging.getLogger(__name__)

inference_data = {}


class TestImg2ImgInferenceAsyncE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_img2img_inference_async_create(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "user_id": config.username,
            "inference_type": "Async",
            "task_type": InferenceType.IMG2IMG.value,
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
            "filters": {}
        }

        resp = self.api.create_inference(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        global inference_data
        inference_data = resp.json()['data']["inference"]

        assert resp.json()["statusCode"] == 201
        assert inference_data["type"] == InferenceType.IMG2IMG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/img2img_api_param.json")

    def test_2_img2img_inference_async_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.IMG2IMG.value

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.get_inference_job(headers=headers, job_id=inference_data["id"])
        assert resp.status_code == 200, resp.dumps()

    def test_5_img2img_inference_async_and_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.IMG2IMG.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 202, resp.dumps()

        assert resp.json()['data']["inference"]["status"] == InferenceStatus.INPROGRESS.value

        timeout = datetime.now() + timedelta(minutes=5)

        while datetime.now() < timeout:
            status = get_inference_job_status(
                api_instance=self.api,
                job_id=inference_id
            )
            logger.info(f"img2img_inference_async is {status}")
            if status == InferenceStatus.SUCCEED.value:
                break
            if status == InferenceStatus.FAILED.value:
                logger.error(resp.dumps())
                logger.error(inference_data)
                raise Exception(f"Inference job {inference_id} failed.")
            time.sleep(5)
        else:
            raise Exception("Inference execution timed out after 5 minutes.")

    def test_6_img2img_inference_async_content(self):
        global inference_data

        inference_id = inference_data["id"]

        get_inference_job_image(
            api_instance=self.api,
            job_id=inference_id,
            target_file="./data/api_params/img2img_api_param.png"
        )
