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

    def test_1_users_get_without_key(self):
        resp = self.api.list_users()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_users_get_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_users_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200
        assert len(resp.json()["users"]) >= 0
        assert resp.json()["users"][0]["username"] == config.username

    def test_4_user_delete_without_key(self):
        resp = self.api.user_delete("test")

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_5_user_delete_not_found(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.user_delete("username_not_found", headers=headers)

        assert resp.status_code == 200
        assert 'user' in resp.json()
        assert resp.json()["user"]['status'] == "deleted"

    def test_6_user_post_bad_creator(self):
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

        resp = self.api.create_user(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 400
        assert resp.json()["errMsg"] == "creator bad_creator not exist"

    def test_7_user_post_bad_role(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "username": "XXXXXXXXXXXXX",
            "password": "XXXXXXXXXXXXX",
            "creator": "admin",
            "roles": ["admin"],
        }

        resp = self.api.create_user(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 400
        assert resp.json()["errMsg"] == 'user roles "admin" not exist'
