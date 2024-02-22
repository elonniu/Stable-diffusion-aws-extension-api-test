from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)


class TestModelsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_list_models_without_key(self):
        resp = self.api.list_models()

        assert resp.status_code == 401, resp.dumps()
        assert resp.json()["message"] == "Unauthorized"

    def test_2_list_models_without_auth(self):
        resp = self.api.list_models()

        assert resp.status_code == 401, resp.dumps()
        assert resp.json()["message"] == "Unauthorized"

    def test_3_update_model_without_key(self):
        resp = self.api.update_model(model_id="job_id")

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_3_update_model_with_bad_body(self):
        headers = {
            "x-api-key": config.api_key,
        }
        resp = self.api.update_model(headers=headers, model_id="job_id")

        assert resp.status_code == 400, resp.dumps()
        assert 'Unknown error parsing request body' in resp.json()["message"]

    def test_4_list_models(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_models(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        assert len(resp.json()['data']["models"]) >= 0

    def test_5_create_model_without_key(self):
        resp = self.api.create_model()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_6_delete_models_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 403, resp.dumps()
        assert 'Forbidden' == resp.json()["message"]

    def test_7_delete_models_with_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()["message"]
        assert 'model_id_list' in resp.json()["message"]

    def test_8_delete_models_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "model_id_list": ['id'],
        }

        resp = self.api.delete_models(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()
