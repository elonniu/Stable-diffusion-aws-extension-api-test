from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestModelsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_models_get_without_key(self):
        resp = self.api.list_models()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_models_get_without_auth(self):
        resp = self.api.list_models()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_model_put_without_key(self):
        resp = self.api.update_model()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_4_models_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_models(headers=headers)

        assert resp.status_code == 200
        assert len(resp.json()["models"]) >= 0

    def test_5_model_post_without_key(self):
        resp = self.api.create_model()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_6_models_delete_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_7_models_delete_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'model_id_list' in resp.json()["message"]

    def test_8_models_delete_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "model_id_list": ['id'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 200
        assert 'models deleted' == resp.json()["message"]
