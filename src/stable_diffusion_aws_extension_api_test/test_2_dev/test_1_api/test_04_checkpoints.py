from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestCheckpointsApi:

    @classmethod
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_checkpoints_get_without_key(self):
        resp = self.api.list_checkpoints()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_checkpoints_get_without_auth(self):
        headers = {"x-api-key": config.api_key}

        resp = self.api.list_checkpoints(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_checkpoints_get_without_username(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_checkpoints(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()["checkpoints"]) >= 0

    def test_4_checkpoints_get_with_user_name(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {"username": config.username}

        resp = self.api.list_checkpoints(
            headers=headers,
            params=params
        )

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()["checkpoints"]) >= 0

    def test_5_checkpoint_with_bad_username(self):
        filename = "v1-5-pruned-emaonly.safetensors"
        checkpoint_type = "Stable-diffusion"

        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "checkpoint_type": checkpoint_type,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 5
                }
            ],
            "params": {
                "message": "placeholder for chkpts upload test",
                "creator": "bad_username"
            }
        }

        resp = self.api.create_checkpoint(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 500
        assert "user: \"bad_username\" not exist" in resp.json()["error"]

    def test_6_upload_checkpoint_without_key(self):
        resp = self.api.upload_checkpoint()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_7_checkpoints_delete_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_checkpoints(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'checkpoint_id_list' in resp.json()["message"]

    def test_8_checkpoints_delete_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_roles(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]
