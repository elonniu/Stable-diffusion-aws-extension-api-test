from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestUsersApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_list_users_without_key(self):
        resp = self.api.list_users()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_list_users_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_list_users(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        users = resp.json()["data"]["users"]
        assert len(users) >= 0
        assert users[0]["username"] == config.username

    def test_4_delete_users_without_key(self):
        data = {
            "user_name_list": ["test"],
        }

        resp = self.api.delete_users(headers={}, data=data)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_5_delete_users_not_found(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_name_list": ["test"],
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["message"] == "Users Deleted"

    def test_6_create_user_bad_creator(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "username": "XXXXXXXXXXXXX",
            "password": "XXXXXXXXXXXXX",
            "creator": "bad_creator",
            "roles": ['IT Operator', 'Designer'],
        }

        resp = self.api.create_user_new(headers=headers, data=data)

        assert resp.status_code == 400
        assert resp.json()["message"] == "creator bad_creator not exist"

    def test_7_create_user_with_bad_role(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "username": "XXXXXXXXXXXXX",
            "password": "XXXXXXXXXXXXX",
            "creator": config.username,
            "roles": ["admin"],
        }

        resp = self.api.create_user_new(headers=headers, data=data)

        assert resp.status_code == 400
        assert resp.json()["message"] == 'user roles "admin" not exist'

    def test_8_delete_users_with_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]

    def test_9_delete_users_with_username_empty(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_name_list": [""],
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 400
        assert 'required minimum: 1' in resp.json()["message"]
