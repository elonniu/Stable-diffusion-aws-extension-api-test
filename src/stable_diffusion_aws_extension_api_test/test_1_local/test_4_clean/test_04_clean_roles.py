from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanRoles:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_clean_roles(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        role_name_list = []
        roles = self.api.list_roles(headers=headers).json()['data']['roles']
        for role in roles:
            role_name_list.append(role['role_name'])
            logger.info(role['role_name'])

        if len(role_name_list) == 0:
            logger.info("No roles to delete")
            return

        data = {
            "role_name_list": role_name_list,
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 204
