from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import init_user_role
from tenacity import stop_after_delay, retry

logger = logging.getLogger(__name__)


class TestRolesApi:
    def setup_class(cls):
        cls.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_init_user_role(self):
        init_user_role()

    def test_1_roles_get_without_api_key(self):
        resp = self.api.list_roles()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_roles_get_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_roles(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_role_post_without_key(self):
        resp = self.api.create_role()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    # failed test if all retries failed after 20 seconds
    @retry(stop=stop_after_delay(20))
    def test_4_role_post_bad_creator(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "role_name": "role_name",
            "creator": "bad_creator",
            "permissions": ['train:all', 'checkpoint:all'],
        }

        resp = self.api.create_role(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 400
        assert resp.json()["errMsg"] == 'creator bad_creator not exist'

    def test_5_roles_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_roles(headers=headers)

        assert resp.status_code == 200
        assert len(resp.json()["roles"]) >= 2
        assert resp.json()["roles"][0]["role_name"] == "Designer"
        assert resp.json()["roles"][1]["role_name"] == "IT Operator"

    def test_6_roles_delete_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_7_roles_delete_bad_request_body(self):
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
