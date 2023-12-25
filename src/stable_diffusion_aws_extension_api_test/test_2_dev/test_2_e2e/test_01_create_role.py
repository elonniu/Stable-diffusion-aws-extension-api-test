from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestRoleE2E:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_create_role(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "role_name": config.role_name,
            "creator": "admin",
            "permissions": ['train:all', 'checkpoint:all'],
        }

        resp = self.api.create_role_new(headers=headers, data=data)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["data"]['role_name'] == config.role_name

    def test_2_list_roles_exists(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_roles(headers=headers)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        roles = resp.json()['data']["roles"]
        assert config.role_name in [user["role_name"] for user in roles]

    def test_3_delete_roles_default(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "role_name_list": ['IT Operator'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert 'cannot delete default role' in resp.json()["message"]

    def test_4_delete_roles_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "role_name_list": [config.role_name],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert 'roles deleted' == resp.json()["message"]
