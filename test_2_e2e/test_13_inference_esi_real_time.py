from __future__ import print_function

import logging
from datetime import datetime

import config as config
from utils.api import Api
from utils.enums import InferenceType
from utils.helper import upload_with_put

logger = logging.getLogger(__name__)

inference_data = {}


class TestEsiRealTimeE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_esi_inference_real_time_create(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "inference_type": "Real-time",
            "task_type": InferenceType.ESI.value,
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
            "filters": {
                "createAt": datetime.now().timestamp(),
                "creator": "sd-webui"
            }
        }

        resp = self.api.create_inference(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        global inference_data
        inference_data = resp.json()['data']["inference"]

        assert resp.json()["statusCode"] == 201
        assert inference_data["type"] == InferenceType.ESI.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"],
                        "./data/api_params/extra-single-image-api-params.json")

    def test_2_esi_inference_real_time_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.ESI.value

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

    def test_3_esi_inference_real_time_start_and_succeed(self):
        global inference_data
        assert inference_data["type"] == InferenceType.ESI.value

        inference_id = inference_data["id"]

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert 'img_presigned_urls' in resp.json()['data'], resp.dumps()
        assert len(resp.json()['data']['img_presigned_urls']) > 0, resp.dumps()
