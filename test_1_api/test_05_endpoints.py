from __future__ import print_function

import logging
from time import sleep

import config as config
from utils.api import Api
from utils.helper import get_endpoint_status, delete_sagemaker_endpoint

logger = logging.getLogger(__name__)


class TestEndpointsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_endpoints_async_delete_before(self):
        endpoint_name = f"esd-async-{config.endpoint_name}"
        while True:
            status = get_endpoint_status(self.api, endpoint_name)
            if status is None:
                break

            if status in ['Creating', 'Updating']:
                logger.warning(f"Endpoint {endpoint_name} is {status}, waiting to delete...")
                sleep(10)
            else:
                delete_sagemaker_endpoint(self.api)
                break
        pass

    def test_1_endpoints_real_time_delete_before(self):
        endpoint_name = f"esd-real-time-{config.endpoint_name}"
        while True:
            status = get_endpoint_status(self.api, endpoint_name)
            if status is None:
                break

            if status in ['Creating', 'Updating']:
                logger.warning(f"Endpoint {endpoint_name} is {status}, waiting to delete...")
                sleep(10)
            else:
                delete_sagemaker_endpoint(self.api)
                break
        pass

    def test_1_list_endpoints_without_key(self):
        resp = self.api.list_endpoints()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_2_list_endpoints_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 401, resp.dumps()
        assert resp.json()["message"] == "Unauthorized"

    def test_3_list_endpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 200, resp.dumps()

        assert resp.json()["statusCode"] == 200
        assert len(resp.json()['data']["endpoints"]) >= 0

    def test_4_list_endpoints_with_username(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_endpoints(headers=headers, params=params)

        assert resp.status_code == 200, resp.dumps()

        assert resp.json()["statusCode"] == 200
        assert len(resp.json()['data']["endpoints"]) >= 0

    def test_5_list_endpoints_with_bad_username(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        params = {
            "username": "admin_error"
        }

        resp = self.api.list_endpoints(headers=headers, params=params)

        assert resp.status_code == 400, resp.dumps()

        assert "user: \"admin_error\" not exist" in resp.json()["message"]

    def test_6_create_endpoint_without_params(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "none": "none",
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()["message"]

    def test_7_create_endpoint_with_bad_instance_count(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "endpoint_name": "test",
            "endpoint_type": "Async",
            "instance_type": config.async_instance_type,
            "initial_instance_count": 1000,
            "autoscaling_enabled": True,
            "assign_to_roles": ["IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()
        assert 'ResourceLimitExceeded' in resp.text

    def test_8_create_endpoint_with_larger(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        instance_type = "ml.g4dn.16xlarge"

        data = {
            "endpoint_name": "dev-test",
            "instance_type": instance_type,
            "endpoint_type": "Async",
            "initial_instance_count": 9,
            "autoscaling_enabled": True,
            "assign_to_roles": ["IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()
        assert f"{instance_type} for endpoint usage' is 0 Instances" in resp.json()["message"]

    def test_9_delete_endpoints_without_key(self):
        resp = self.api.delete_endpoints()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_10_create_endpoint_without_key(self):
        resp = self.api.create_endpoint()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    # if endpoint is old, it still will be deleted
    def test_11_delete_endpoints_old_data(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "endpoint_name_list": [
                f"test"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 204, resp.dumps()

    def test_12_delete_endpoints_bad_username(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "endpoint_name_list": [
                f"test"
            ],
            "username": "bad_user"
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 204, resp.dumps()

