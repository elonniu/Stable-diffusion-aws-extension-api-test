from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import pytest

import config as config
from utils.api import Api
from utils.helper import upload_db_config

logger = logging.getLogger(__name__)
train_job_id = ""


class TestTrainBaseModelE2E:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_1_train_job_create(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        models = self.api.list_models(headers=headers).json()["data"]['models']

        for model in models:
            assert "id" in model
            model_name = model["model_name"]
            if model_name != config.model_name:
                continue
            model_id = model["id"]
            model_type = 'Stable-diffusion'
            s3_model_path = f"s3://{config.bucket}/Stable-diffusion/model/{model_name}/{model_id}/output"

            headers = {
                "x-api-key": config.api_key,
                "username": config.username
            }

            data = {
                "train_type": model_type,
                "model_id": model_id,
                "creator": config.username,
                "filenames": [
                    "db_config.tar"
                ],
                "params": {
                    "training_params": {
                        "s3_model_path": s3_model_path,
                        "model_name": model_name,
                        "model_type": model_type,
                        "data_tar_list": [
                            f"s3://{config.bucket}/dataset/{config.dataset_name}"
                        ],
                        "class_data_tar_list": [
                            ""
                        ],
                        "s3_data_path_list": [
                            f"s3://{config.bucket}/dataset/{config.dataset_name}"
                        ],
                        "s3_class_data_path_list": [
                            ""
                        ],
                        "training_instance_type": config.async_instance_type
                    }
                }
            }

            resp = self.api.create_training_job(headers=headers, data=data)
            assert resp.status_code == 201, resp.dumps()
            assert resp.json()["statusCode"] == 201
            job = resp.json()['data']["job"]
            assert job["status"] == "Initial"
            global train_job_id
            train_job_id = job["id"]
            s3_presign_url = resp.json()['data']["s3PresignUrl"]["db_config.tar"]
            upload_db_config(s3_presign_url)

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_2_train_put(self):
        global train_job_id
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        data = {
            "status": "Training"
        }

        resp = self.api.start_training_job(training_id=train_job_id, headers=headers, data=data)
        assert resp.status_code == 202, resp.dumps()
        assert resp.json()["statusCode"] == 202

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_3_trains_get(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        global train_job_id
        jobs = resp.json()['data']["trainJobs"]
        assert train_job_id in [train["id"] for train in jobs]

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_4_train_post_wait_for_complete(self):

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)
        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        global train_job_id
        jobs = resp.json()['data']["trainJobs"]
        assert train_job_id in [train["id"] for train in jobs]

        timeout = datetime.now() + timedelta(minutes=50)

        while datetime.now() < timeout:
            result = self.train_wait_for_complete()
            if result:
                break
            time.sleep(50)
        else:
            raise Exception(f"train {train_job_id} timed out after 50 minutes.")

    def train_wait_for_complete(self):
        global train_job_id

        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        jobs = resp.json()['data']["trainJobs"]
        for train in jobs:
            if train["id"] == train_job_id:
                if train["status"] == "Completed":
                    return True
                if train["status"] == "Fail":
                    logger.error("Train job failed.")
                    logger.error(resp.dumps())
                    raise Exception("Train job failed.")
                logger.info(f"Model {train_job_id} is {train['status']}...")
                return False

        return False
