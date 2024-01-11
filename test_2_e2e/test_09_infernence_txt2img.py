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


class TestTxt2ImgInferenceE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass
        #
        # global inference_data
        # if 'id' in inference_data:
        #     delete_inference_jobs([inference_data['id']])

    def test_1_txt2img_inference_job_create(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "task_type": InferenceType.TXT2IMG.value,
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
        assert inference_data["type"] == InferenceType.TXT2IMG.value
        assert len(inference_data["api_params_s3_upload_url"]) > 0

        upload_with_put(inference_data["api_params_s3_upload_url"], "./data/api_params/txt2img_api_param.json")

    def test_2_txt2img_inference_job_exists(self):
        global inference_data
        assert inference_data["type"] == InferenceType.TXT2IMG.value

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

    def test_5_txt2img_inference_job_run_and_succeed(self):
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
                logger.error("Inference job failed.")
                logger.error(resp.dumps())
                raise Exception("Inference job failed.")
            time.sleep(5)
        else:
            raise Exception("Inference execution timed out after 5 minutes.")

    def test_7_txt2img_inference_job_delete_succeed(self):
        headers = {
            "x-api-key": config.api_key
        }

        data = {
            # todo will use exists id
            "inference_id_list": ['any_one'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()
