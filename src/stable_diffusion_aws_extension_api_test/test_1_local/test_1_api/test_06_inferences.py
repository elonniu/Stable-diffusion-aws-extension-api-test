from __future__ import print_function

import logging
from datetime import datetime

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestInferencesApi:
    @classmethod
    def setup_class(cls):
        cls.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_create_inference_without_key(self):
        resp = self.api.create_inference_new()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_2_create_inference_without_auth(self):
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

        resp = self.api.create_inference_new(data=data)

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_3_create_inference_with_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "bad_param": "bad_param",
        }

        resp = self.api.create_inference_new(headers=headers, data=data)

        assert resp.status_code == 400

        assert 'object has missing required properties' in resp.json()['message']

    def test_12_list_inferences_without_key(self):
        resp = self.api.list_inferences()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_13_list_inferences_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_inferences(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_16_delete_inferences_with_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'inference_id_list' in resp.json()["message"]

    def test_17_delete_inferences_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_17_delete_inferences_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "inference_id_list": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 204

    def test_18_get_inference_job_without_key(self):
        resp = self.api.get_inference_job_new(job_id="job_id")
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_19_get_inference_job_not_found(self):
        headers = {
            "x-api-key": config.api_key,
        }

        job_id = "not_exists"

        resp = self.api.get_inference_job_new(job_id=job_id, headers=headers)
        assert resp.status_code == 404
        assert f'inference with id {job_id} not found' == resp.json()["message"]
