from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestRolesApi:
    def setup_class(cls):
        cls.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_init_user_and_role(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "username": config.username,
            "password": "XXXXXXXXXXXXX",
            "creator": config.username,
            "initial": True,
        }

        resp = self.api.create_user_new(headers=headers, data=data)

        assert resp.status_code == 201
        assert resp.json()["message"] == "OK"

    def test_2_list_roles_without_api_key(self):
        resp = self.api.list_roles()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_list_roles_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_roles(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_4_create_role_without_key(self):
        resp = self.api.create_role_new()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_5_create_role_with_bad_creator(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "role_name": "role_name",
            "creator": "bad_creator",
            "permissions": ['train:all', 'checkpoint:all'],
        }

        resp = self.api.create_role_new(headers=headers, data=data)

        assert resp.status_code == 400
        assert resp.json()["statusCode"] == 400
        assert resp.json()["message"] == 'creator bad_creator not exist'

    def test_6_list_roles(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
            'role_params': 'role_name'
        }

        resp = self.api.list_roles(headers=headers, params=params)

        assert resp.status_code == 200
        roles = resp.json()['data']["roles"]
        assert len(roles) >= 1

    def test_7_list_roles_without_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        params = {
        }

        resp = self.api.list_roles(headers=headers, params=params)

        assert resp.status_code == 200
        roles = resp.json()['data']["roles"]
        assert len(roles) >= 1

    def test_8_delete_roles_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_9_delete_roles_with_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'role_name_list' in resp.json()["message"]
