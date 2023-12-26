from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanEndpoint:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_delete_endpoints(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name_list": [
                f"infer-endpoint-{config.endpoint_name}"
            ],
            "username": config.username
        }

        resp = self.api.delete_endpoints(headers=headers, data=data)

        assert resp.status_code == 204
        assert resp.json()["statusCode"] == 204
        assert resp.json()["message"] == "Endpoints Deleted"

    def test_2_clean_datasets(self):
        headers = {
            "x-api-key": config.api_key
        }

        data = {
            "dataset_name_list": [
                config.dataset_name,
            ],
        }

        resp = self.api.delete_datasets(headers=headers, data=data)
        assert resp.status_code == 204
        assert 'datasets deleted' == resp.json()["message"]

    # def test_3_clean_roles(self):
    #     headers = {
    #         "x-api-key": config.api_key,
    #         "Authorization": config.bearer_token
    #     }
    #
    #     role_name_list = []
    #     roles = self.api.list_roles(headers=headers).json()['data']['roles']
    #     for role in roles:
    #         role_name_list.append(role['role_name'])
    #         logger.info(role['role_name'])
    #
    #     data = {
    #         "role_name_list": role_name_list,
    #     }
    #
    #     resp = self.api.delete_roles(headers=headers, data=data)
    #     assert 'roles deleted' == resp.json()["message"]
