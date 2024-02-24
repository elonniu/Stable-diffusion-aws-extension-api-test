from __future__ import print_function

import logging

import config as config
from utils.api import Api
from utils.enums import InferenceType
from utils.helper import upload_with_put, get_inference_job_image

logger = logging.getLogger(__name__)

inference_data = {}


class TestRembgRealTimeE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_rembg_inference_real_time_create(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "user_id": config.username,
            "inference_type": "Real-time",
            "task_type": InferenceType.REMBG.value,
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
        }

        resp = self.api.create_inference(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        global inference_data
        inference_data = resp.json()['data']["inference"]

        assert resp.json()["statusCode"] == 201
        assert inference_data["type"] == InferenceType.REMBG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/rembg-api-params.json")

    def test_2_rembg_inference_real_time_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.REMBG.value

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.get_inference_job(headers=headers, job_id=inference_data["id"])
        assert resp.status_code == 200, resp.dumps()

    def test_3_rembg_inference_real_time_start_and_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.REMBG.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert 'img_presigned_urls' in resp.json()['data'], resp.dumps()
        assert len(resp.json()['data']['img_presigned_urls']) > 0, resp.dumps()

        if resp.status_code == 504:
            import time
            time.sleep(5)

    def test_4_rembg_inference_real_time_content(self):
        global inference_data

        inference_id = inference_data["id"]

        get_inference_job_image(
            api_instance=self.api,
            job_id=inference_id,
            target_file="./data/api_params/rembg-api-params.png"
        )
