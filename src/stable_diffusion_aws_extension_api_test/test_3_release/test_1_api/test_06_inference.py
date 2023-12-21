from __future__ import print_function

import logging
from datetime import datetime

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from tenacity import stop_after_attempt, retry

logger = logging.getLogger(__name__)


class TestInferenceApi:
    @classmethod
    def setup_class(cls):

        cls.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_inference_post_without_key(self):
        resp = self.api.create_inference()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_2_inference_post_without_auth(self):
        data = {
            "user_id": config.username,
            "task_type": "txt2img",
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
            "filters": {
                "createAt": datetime.now().timestamp(),
                "creator": "sd-webui"
            }
        }

        resp = self.api.create_inference(data=data)

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_3_inference_post(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "bad_param": "bad_param",
        }

        resp = self.api.create_inference(headers=headers, data=data)

        assert resp.status_code == 200

        assert 'unexpected keyword argument' in resp.json()['errorMessage']

    def test_4_inference_txt2img_run_job_param_output_without_key(self):
        resp = self.api.get_inference_job_param_output()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    # failed test if all retries failed after 20 seconds
    @retry(stop=stop_after_attempt(5))
    def test_5_inference_txt2img_run_job_param_output(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "jobId": "eae90ee9-1111-2222-3333-40b943ff6fb9"
        }

        resp = self.api.get_inference_job_param_output(
            headers=headers,
            params=params
        )

        assert resp.status_code == 200
        assert len(resp.json()) == 0

    def test_6_inference_get_controlnet_model_list_without_key(self):
        resp = self.api.get_controlnet_model_list()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_7_inference_get_controlnet_model_list(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.get_controlnet_model_list(headers=headers)

        assert resp.status_code == 200
        assert resp.json() == []

    def test_8_inference_get_lora_list_get_without_key(self):
        resp = self.api.get_lora_list_get()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_9_inference_get_lora_list_get(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.get_lora_list_get(
            headers=headers
        )

        assert resp.status_code == 200
        assert resp.json() == []

    def test_10_inference_get_hypernetwork_list_get_without_key(self):
        resp = self.api.get_hypernetwork_list()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_11_inference_get_hypernetwork_list_get(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.get_hypernetwork_list(headers=headers)

        assert resp.status_code == 200
        assert resp.json() == []

    def test_12_inference_get_texual_inversion_list_get_without_key(self):
        resp = self.api.get_texual_inversion_list_get()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_13_inference_get_texual_inversion_list_get(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.get_texual_inversion_list_get(headers=headers)

        assert resp.status_code == 200
        assert len(resp.json()) >= 0

    def test_14_inference_list_inference_jobs_get_without_key(self):
        resp = self.api.list_inferences()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_15_inference_list_inference_jobs_get_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_inferences(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_16_query_inferences_without_key(self):
        resp = self.api.query_inferences()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_17_query_inferences_with_checkpoint_not_exists(self):
        headers = {"x-api-key": config.api_key}

        data = {
            "checkpoint": "not_exists",
            "limit": 1
        }

        resp = self.api.query_inferences(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json() == []
