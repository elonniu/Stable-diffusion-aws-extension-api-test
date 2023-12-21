from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestDatasetsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_datasets_get_without_key(self):
        resp = self.api.list_datasets()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_2_datasets_get_without_auth(self):
        headers = {"x-api-key": config.api_key}
        resp = self.api.list_datasets(headers=headers)

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_3_datasets_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_datasets(headers=headers)
        assert resp.status_code == 200

        assert len(resp.json()["datasets"]) >= 0

    def test_4_dataset_get_missing_name(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.get_dataset_data(
            headers=headers,
            name="not_exists"
        )

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 500
        assert 'error' in resp.json()

    def test_5_dataset_get_not_exists(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.get_dataset_data(
            name="dataset_name_not_exists",
            headers=headers,
        )

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 500
        assert 'not found' in resp.json()["error"]

    def test_6_dataset_post_without_key(self):
        resp = self.api.create_dataset()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_7_dataset_put_without_key(self):
        resp = self.api.update_dataset()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_8_datasets_delete_without_key(self):
        headers = {}

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_datasets(headers=headers, data=data)
        assert resp.status_code == 403
        assert 'Forbidden' == resp.json()["message"]

    def test_9_datasets_delete_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_datasets(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'dataset_name_list' in resp.json()["message"]
