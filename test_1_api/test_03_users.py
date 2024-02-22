from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)


class TestUsersApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_list_users_without_key(self):
        resp = self.api.list_users()

        assert resp.status_code == 403, resp.dumps()

        assert resp.json()["message"] == "Forbidden"

    def test_2_list_users_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 401, resp.dumps()

        assert resp.json()["message"] == "Unauthorized"

    def test_3_list_users(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_users(headers=headers)

        assert resp.status_code == 200, resp.dumps()

        users = resp.json()["data"]["users"]
        assert len(users) >= 0
        assert users[0]["username"] == config.username

    def test_4_delete_users_without_key(self):
        data = {
            "user_name_list": ["test"],
        }

        resp = self.api.delete_users(headers={}, data=data)

        assert resp.status_code == 401, resp.dumps()

        assert resp.json()["message"] == "Unauthorized"

    def test_5_delete_users_not_found(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "user_name_list": ["test"],
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 204, resp.dumps()

    def test_6_create_user_bad_creator(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "username": "XXXXXXXXXXXXX",
            "password": "XXXXXXXXXXXXX",
            "creator": "bad_creator",
            "roles": ['IT Operator'],
        }

        resp = self.api.create_user(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()

        assert resp.json()["message"] == "creator bad_creator not exist"

    def test_7_create_user_with_bad_role(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "username": "XXXXXXXXXXXXX",
            "password": "XXXXXXXXXXXXX",
            "creator": config.username,
            "roles": ["admin"],
        }

        resp = self.api.create_user(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()

        assert resp.json()["message"] == 'user roles "admin" not exist'

    def test_8_delete_users_with_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()["message"]

    def test_9_delete_users_with_username_empty(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "user_name_list": [""],
        }

        resp = self.api.delete_users(headers=headers, data=data)

        assert resp.status_code == 400, resp.dumps()

        assert 'required minimum: 1' in resp.json()["message"]
