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

    def test_1_user_post(self):
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

        resp = self.api.create_user(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200

    def test_2_user_post_check_exists_name(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["status"] == 200
        assert username in [user["username"] for user in resp.json()["users"]]

    def test_3_user_delete(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.user_delete(headers=headers, username=username)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert resp.json()["user"]["status"] == "deleted"

    def test_4_user_delete_check(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["status"] == 200
        assert username not in [user["username"] for user in resp.json()["users"]]
