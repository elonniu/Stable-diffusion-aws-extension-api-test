from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import config as config
from utils.api import Api
from utils.enums import InferenceStatus, InferenceType
from utils.helper import upload_with_put, get_inference_job_status_new

logger = logging.getLogger(__name__)

inference_data = {}


class TestRembgInferenceAsyncE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_rembg_inference_async_create(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "inference_type": "Async",
            "task_type": InferenceType.REMBG.value,
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
        assert inference_data["type"] == InferenceType.REMBG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/rembg-api-params.json")

    def test_2_rembg_inference_async_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.REMBG.value

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_inferences(headers=headers, params=params)
        assert resp.status_code == 200, resp.dumps()

        assert resp.json()["statusCode"] == 200
        inferences = resp.json()['data']["inferences"]
        assert inference_data["id"] in [inference["InferenceJobId"] for inference in inferences]

    def test_3_rembg_inference_async_and_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.REMBG.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 202, resp.dumps()

        assert resp.json()['data']["inference"]["status"] == InferenceStatus.INPROGRESS.value

        timeout = datetime.now() + timedelta(minutes=5)

        while datetime.now() < timeout:
            status = get_inference_job_status_new(
                api_instance=self.api,
                job_id=inference_id
            )
            logger.info(f"rembg_inference_async is {status}")
            if status == InferenceStatus.SUCCEED.value:
                break
            if status == InferenceStatus.FAILED.value:
                logger.error(resp.dumps())
                logger.error(inference_data)
                raise Exception(f"Inference job {inference_id} failed.")
            time.sleep(5)
        else:
            raise Exception("Inference execution timed out after 5 minutes.")
