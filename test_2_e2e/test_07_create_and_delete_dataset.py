from __future__ import print_function

import logging
import os

import requests

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)
dataset = {}
dataset_name = "test_dataset_name"


class TestCreateAndDeleteDatasetE2E:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_dataset_post(self):
        dataset_content = []

        for filename in os.listdir("./data/dataset"):
            dataset_content.append({
                'filename': filename,
                'name': filename,
                'type': 'image',
                'params': {}
            })

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            'dataset_name': dataset_name,
            'content': dataset_content,
            'creator': config.username,
            'params': {'description': 'this is description'}
        }

        resp = self.api.create_dataset(headers=headers, data=data)
        assert resp.status_code == 201, resp.dumps()

        global dataset
        dataset = resp.json()

        assert dataset["statusCode"] == 201
        assert dataset['data']["datasetName"] == dataset_name

    def test_2_dataset_img_upload(self):
        global dataset
        for filename, presign_url in dataset['data']['s3PresignUrl'].items():
            file_path = f"./data/dataset/{filename}"
            with open(file_path, 'rb') as file:
                resp = requests.put(presign_url, file)
                resp.raise_for_status()
                assert resp.status_code == 200

    def test_3_dataset_update_status(self):
        global dataset
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        data = {
            "status": "Enabled"
        }

        resp = self.api.update_dataset(dataset_id=dataset_name, headers=headers, data=data)
        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200

    def test_4_datasets_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_datasets(headers=headers)
        assert resp.status_code == 200, resp.dumps()

        datasets = resp.json()['data']["datasets"]
        assert dataset_name in [user["datasetName"] for user in datasets]

    def test_5_dataset_get(self):

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.get_dataset(name=dataset_name, headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200

    def test_6_datasets_delete_succeed(self):
        headers = {
            "x-api-key": config.api_key
        }

        data = {
            "dataset_name_list": [
                dataset_name,
            ],
        }

        resp = self.api.delete_datasets(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()
