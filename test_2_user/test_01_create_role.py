from __future__ import print_function

import logging

import config as config
from utils.api import Api

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
            "username": config.username,
        }

        data = {
            "role_name": config.role_name,
            "creator": "api",
            "permissions": ['train:all', 'checkpoint:all'],
        }

        resp = self.api.create_role(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        assert resp.json()["statusCode"] == 201

    def test_2_create_role_byoc(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        data = {
            "role_name": "byoc",
            "creator": "api",
            "permissions": ['sagemaker_endpoint:all'],
        }

        resp = self.api.create_role(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        assert resp.json()["statusCode"] == 201

    def test_2_list_roles_exists(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        resp = self.api.list_roles(headers=headers)
        assert resp.status_code == 200, resp.dumps()

        assert resp.json()["statusCode"] == 200
        roles = resp.json()['data']["roles"]
        assert config.role_name in [user["role_name"] for user in roles]

    def test_3_delete_roles_default(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        data = {
            "role_name_list": ['IT Operator'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()
        assert 'cannot delete default role' in resp.json()["message"]

    def test_4_delete_roles_succeed(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        data = {
            "role_name_list": [config.role_name],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()
