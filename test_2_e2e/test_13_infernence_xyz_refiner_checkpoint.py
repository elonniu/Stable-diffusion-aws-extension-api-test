from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import pytest
import config as config
from utils.api import Api
from utils.enums import InferenceStatus, InferenceType
from utils.helper import upload_with_put, get_inference_job_status_new, \
    delete_inference_jobs

logger = logging.getLogger(__name__)

filename = "v1-5-pruned-emaonly.safetensors"
api_params_filename = "./data/api_params/xyz_refiner_checkpoint_api_param.json"
inference_data = {}


class TestXyzRefinerCheckpointE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

        global inference_data
        if 'id' in inference_data:
            delete_inference_jobs([inference_data['id']])

    @pytest.mark.skip(reason="not ready")
    def test_1_xyz_refiner_checkpoint_txt2img_inference_job_create(self):

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
            }
        }

        resp = self.api.create_inference_new(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()
        global inference_data
        inference_data = resp.json()["inference"]

        assert resp.json()["statusCode"] == 201
        assert inference_data["type"] == InferenceType.TXT2IMG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], api_params_filename)

    @pytest.mark.skip(reason="not ready")
    def test_2_xyz_refiner_checkpoint_txt2img_inference_job_succeed(self):

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

        timeout = datetime.now() + timedelta(minutes=2)

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
            raise Exception("Inference execution timed out after 2 minutes.")
