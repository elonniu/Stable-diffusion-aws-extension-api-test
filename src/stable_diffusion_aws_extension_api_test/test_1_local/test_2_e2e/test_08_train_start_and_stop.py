from __future__ import print_function

import logging

import pytest
import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.helper import get_test_model, \
    delete_train_item, upload_db_config

logger = logging.getLogger(__name__)
train_job_id = ""


class TestTrainStartStopE2E:
    def setup_class(self):
        self.api = Api(config)
        delete_train_item()

    @classmethod
    def teardown_class(cls):
        pass

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_1_train_job_create(self):
        models = get_test_model()

        assert 'Items' in models
        if len(models['Items']) == 0:
            pass
            return

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

            resp = self.api.create_training_job(headers=headers, data=data)
            assert resp.status_code == 201, resp.dumps()
            assert resp.json()["statusCode"] == 201
            job = resp.json()['data']["job"]
            assert job["status"] == "Initial"
            global train_job_id
            train_job_id = job["id"]
            s3_presign_url = resp.json()['data']["s3PresignUrl"]["db_config.tar"]
            upload_db_config(s3_presign_url)

    @pytest.mark.skipif(config.fast_test, reason="fast_test")
    def test_2_train_stop(self):
        global train_job_id

        if not train_job_id:
            pass

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.stop_training_job(training_id=train_job_id, headers=headers)
        assert resp.status_code == 200, resp.dumps()
