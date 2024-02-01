from __future__ import print_function

import logging
from datetime import datetime

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)


class TestInferencesApi:
    @classmethod
    def setup_class(cls):
        cls.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_create_inference_without_key(self):
        resp = self.api.create_inference()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_2_create_inference_without_auth(self):
        data = {
            "user_id": config.username,
            "task_type": "txt2img",
            "inference_type": "Async",
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
        }

        resp = self.api.create_inference(data=data)

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_3_create_inference_with_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "bad_param": "bad_param",
        }

        resp = self.api.create_inference(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()['message']

    def test_12_list_inferences_without_key(self):
        resp = self.api.list_inferences()

        assert resp.status_code == 401, resp.dumps()
        assert resp.json()["message"] == "Unauthorized"

    def test_13_list_inferences_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_inferences(headers=headers)

        assert resp.status_code == 401, resp.dumps()
        assert resp.json()["message"] == "Unauthorized"

    def test_16_delete_inferences_with_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()["message"]
        assert 'inference_id_list' in resp.json()["message"]

    def test_17_delete_inferences_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 403, resp.dumps()

        assert 'Forbidden' == resp.json()["message"]

    def test_17_delete_inferences_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "inference_id_list": ['bad'],
        }

        resp = self.api.delete_inferences(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()

    def test_18_get_inference_job_without_key(self):
        resp = self.api.get_inference_job_new(job_id="job_id")
        assert resp.status_code == 403, resp.dumps()
        assert 'Forbidden' == resp.json()["message"]

    def test_19_get_inference_job_not_found(self):
        headers = {
            "x-api-key": config.api_key,
        }

        job_id = "not_exists"

        resp = self.api.get_inference_job_new(job_id=job_id, headers=headers)
        assert resp.status_code == 404, resp.dumps()
        assert f'inference with id {job_id} not found' == resp.json()["message"]
