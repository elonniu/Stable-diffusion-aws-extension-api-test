from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestModelApi:
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
