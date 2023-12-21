from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import get_test_model, \
    delete_train_item, upload_db_config

logger = logging.getLogger(__name__)
train_job_id = ""


class TestTrainE2E:
    def setup_class(self):
        self.api = Api(config)
        delete_train_item()

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_train_post(self):
        models = get_test_model()

        assert 'Items' in models
        assert len(models['Items']) == 1

        for model in models["Items"]:
            assert "id" in model
            model_name = model["name"]['S']
            if model_name != config.model_name:
                continue
            model_id = model["id"]['S']
            model_type = model["model_type"]['S']
            s3_model_path = f"s3://{config.bucket}/Stable-diffusion/model/{model_name}/{model_id}/output"

            headers = {
                "x-api-key": config.api_key,
                "Authorization": config.bearer_token
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
                        "training_instance_type": config.instance_type
                    }
                }
            }

            resp = self.api.create_train(headers=headers, data=data)
            assert resp.status_code == 200
            assert resp.json()["statusCode"] == 200
            assert resp.json()["job"]["status"] == "Initial"
            global train_job_id
            train_job_id = resp.json()["job"]["id"]
            s3_presign_url = resp.json()["s3PresignUrl"]["db_config.tar"]
            upload_db_config(s3_presign_url)

    def test_2_train_put(self):
        global train_job_id
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "train_job_id": train_job_id,
            "status": "Training"
        }

        resp = self.api.start_train(headers=headers, data=data)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200

    def test_3_trains_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trains(headers=headers)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        global train_job_id
        assert train_job_id in [train["id"] for train in resp.json()["trainJobs"]]

    def test_4_train_post_wait_for_complete(self):

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trains(headers=headers)
        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        global train_job_id
        assert train_job_id in [train["id"] for train in resp.json()["trainJobs"]]

        timeout = datetime.now() + timedelta(minutes=50)

        while datetime.now() < timeout:
            result = self.train_wait_for_complete()
            if result:
                break
            time.sleep(50)
        else:
            raise Exception("train timed out after 50 minutes.")

    def train_wait_for_complete(self):
        global train_job_id

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trains(headers=headers)

        assert resp.status_code == 200
        for train in resp.json()["trainJobs"]:
            if train["id"] == train_job_id:
                if train["status"] == "Complete":
                    return True
                if train["status"] == "Fail":
                    raise Exception("Train job failed.")
                logger.info(f"Model {train_job_id} is {train['status']}...")
                return False

        return False
