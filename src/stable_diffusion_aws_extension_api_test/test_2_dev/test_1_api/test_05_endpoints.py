from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import delete_sagemaker_endpoint

logger = logging.getLogger(__name__)


class TestEndpointsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_endpoints_delete(self):
        delete_sagemaker_endpoint(self.api)

    def test_1_endpoints_get_without_key(self):
        resp = self.api.list_endpoints()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_endpoints_get_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_endpoints_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }
        resp = self.api.list_endpoints(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()["endpoints"]) >= 0

    def test_4_endpoints_get_with_username(self):
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
        assert len(resp.json()["endpoints"]) >= 0

    def test_5_endpoints_get_with_bad_username(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": "admin_error"
        }

        resp = self.api.list_endpoints(headers=headers, params=params)

        assert resp.status_code == 200
        assert "user: \"admin_error\" not exist" in resp.json()["errorMessage"]

    def test_6_endpoints_post_without_params(self):
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

    def test_7_endpoints_post_with_bad_instance_count(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "instance_type": config.instance_type,
            "initial_instance_count": "1000",
            "autoscaling_enabled": True,
            "assign_to_roles": ["Designer", "IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert 'ResourceLimitExceeded' in resp.text

    def test_8_endpoints_post_with_larger(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        instance_type = "ml.g4dn.16xlarge"

        data = {
            "endpoint_name": "dev-test",
            "instance_type": instance_type,
            "initial_instance_count": "9",
            "autoscaling_enabled": True,
            "assign_to_roles": ["Designer", "IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert f"{instance_type} for endpoint usage' is 0 Instances" in resp.json()["message"]

    def test_9_endpoints_delete_without_key(self):
        resp = self.api.delete_endpoints()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_10_endpoints_post_without_key(self):
        resp = self.api.create_endpoint()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    # if endpoint is old, it still will be deleted
    def test_11_endpoints_delete_old_data(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "delete_endpoint_list": [
                f"test"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.text == '"Endpoint deleted"'
