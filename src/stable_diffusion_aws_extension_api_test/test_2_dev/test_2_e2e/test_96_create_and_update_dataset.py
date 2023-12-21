from __future__ import print_function

import logging
import os

import requests
import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import delete_prefix_in_s3, delete_dataset_item, \
    clear_dataset_info

logger = logging.getLogger(__name__)
dataset = {}


class TestDatasetE2E:
    def setup_class(self):
        self.api = Api(config)
        clear_dataset_info("DatasetInfoTable", config.dataset_name)
        delete_dataset_item("DatasetItemTable", config.dataset_name)
        delete_prefix_in_s3(f"dataset/{config.dataset_name}")

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
            'dataset_name': config.dataset_name,
            'content': dataset_content,
            'creator': config.username,
            'params': {'description': 'this is description'}
        }

        resp = self.api.create_dataset(headers=headers, data=data)
        assert resp.status_code == 200
        global dataset
        dataset = resp.json()

        assert resp.json()["statusCode"] == 200
        assert resp.json()["datasetName"] == config.dataset_name

    def test_2_dataset_img_upload(self):
        global dataset
        for filename, presign_url in dataset['s3PresignUrl'].items():
            file_path = f"./data/dataset/{filename}"
            with open(file_path, 'rb') as file:
                logger.info(f"Uploading {file_path}")
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
            "dataset_name": config.dataset_name,
            "status": "Enabled"
        }
        resp = self.api.update_dataset(headers=headers, data=data)
        assert resp.status_code == 200

        assert resp.json()["statusCode"] == 200

    def test_4_datasets_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.list_datasets(headers=headers)

        assert config.dataset_name in [user["datasetName"] for user in resp.json()["datasets"]]

    def test_5_dataset_get(self):
        dataset_name = config.dataset_name

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token,
        }

        resp = self.api.get_dataset_data(name=dataset_name, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
