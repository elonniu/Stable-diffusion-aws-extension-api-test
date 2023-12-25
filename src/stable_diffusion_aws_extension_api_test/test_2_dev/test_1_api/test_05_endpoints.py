from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import delete_sagemaker_endpoint_new

logger = logging.getLogger(__name__)


class TestEndpointsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_endpoints_delete(self):
        delete_sagemaker_endpoint_new(self.api)

    def test_1_list_endpoints_without_key(self):
        resp = self.api.list_endpoints()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_list_endpoints_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_list_endpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()['data']["endpoints"]) >= 0

    def test_4_list_endpoints_with_username(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_endpoints(headers=headers, params=params)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()['data']["endpoints"]) >= 0

    def test_5_list_endpoints_with_bad_username(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": "admin_error"
        }

        resp = self.api.list_endpoints(headers=headers, params=params)

        assert resp.status_code == 400
        assert "user: \"admin_error\" not exist" in resp.json()["message"]

    def test_6_create_endpoint_without_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "none": "none",
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]

    def test_7_create_endpoint_with_bad_instance_count(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "instance_type": config.instance_type,
            "initial_instance_count": 1000,
            "autoscaling_enabled": True,
            "assign_to_roles": ["Designer", "IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert 'ResourceLimitExceeded' in resp.text

    def test_8_create_endpoint_with_larger(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        instance_type = "ml.g4dn.16xlarge"

        data = {
            "endpoint_name": "dev-test",
            "instance_type": instance_type,
            "initial_instance_count": 9,
            "autoscaling_enabled": True,
            "assign_to_roles": ["Designer", "IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)

        assert resp.status_code == 400
        assert f"{instance_type} for endpoint usage' is 0 Instances" in resp.json()["message"]

    def test_9_delete_endpoints_without_key(self):
        resp = self.api.delete_endpoints()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_10_create_endpoint_without_key(self):
        resp = self.api.create_endpoint()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    # if endpoint is old, it still will be deleted
    def test_11_delete_endpoints_old_data(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name_list": [
                f"test"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["message"] == "Endpoints Deleted"

    def test_12_delete_endpoints_bad_username(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name_list": [
                f"test"
            ],
            "username": "bad_user"
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 500
        assert resp.json()["statusCode"] == 500
        assert "error deleting sagemaker endpoint with exception: user: \"bad_user\" not exist" in resp.json()[
            "message"]
