from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
username = "username_exists"


class TestUserE2E:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_create_user(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "username": username,
            "password": "XXXXXXXXXXXXX",
            "creator": "admin",
            "roles": ['IT Operator', 'Designer'],
        }

        resp = self.api.create_user_new(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200

    def test_2_list_users_exists_name(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        users = resp.json()["data"]["users"]
        assert username in [user["username"] for user in users]

    def test_3_delete_users(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "user_name_list": [username],
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["message"] == "Users Deleted"

    def test_4_user_delete_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        users = resp.json()["data"]["users"]
        assert username not in [user["username"] for user in users]
