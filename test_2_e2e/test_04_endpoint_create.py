from __future__ import print_function

import logging
from datetime import datetime

import config as config
from utils.api import Api
from utils.helper import delete_sagemaker_endpoint

logger = logging.getLogger(__name__)


class TestEndpointCreateE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_endpoints_delete(self):
        delete_sagemaker_endpoint(self.api)
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_endpoints(headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert len(resp.json()['data']["endpoints"]) == 0

    def test_2_no_available_endpoint(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        list = self.api.list_endpoints(headers=headers, params=params)

        if 'endpoints' in list and len(list['data']["endpoints"]) > 0:
            return

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "task_type": "txt2img",
            "inference_type": "Async",
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
        assert resp.status_code == 400, resp.dumps()
        assert resp.json()["statusCode"] == 400
        assert resp.json()["message"] == 'no available Async endpoints for user "admin"'

    def test_3_create_endpoint_async(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name": config.endpoint_name,
            "endpoint_type": "Async",
            "instance_type": config.async_instance_type,
            "initial_instance_count": 1,
            "autoscaling_enabled": False,
            "assign_to_roles": ["IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 202, resp.dumps()
        assert resp.json()["data"]["endpoint_status"] == "Creating"

    def test_4_create_endpoint_real_time(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name": config.endpoint_name,
            "endpoint_type": "Real-time",
            "instance_type": config.real_time_instance_type,
            "initial_instance_count": 1,
            "autoscaling_enabled": False,
            "assign_to_roles": ["byoc"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 202, resp.dumps()
        assert resp.json()["data"]["endpoint_status"] == "Creating"

    def test_4_create_endpoint_exists(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name": config.endpoint_name,
            "endpoint_type": 'Async',
            "instance_type": config.async_instance_type,
            "initial_instance_count": int(config.initial_instance_count),
            "autoscaling_enabled": False,
            "assign_to_roles": ["Designer"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()
        assert "Cannot create already existing model" in resp.json()["message"]
