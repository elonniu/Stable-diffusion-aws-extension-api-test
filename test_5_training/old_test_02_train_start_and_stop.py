from __future__ import print_function

import logging

import pytest

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)
train_job_id = ""


class TestTrainStartStopE2E:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_clear_all_trains(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        assert 'trainJobs' in resp.json()["data"]
        trainJobs = resp.json()["data"]["trainJobs"]
        for trainJob in trainJobs:
            data = {
                "training_id_list": [trainJob["id"]],
            }
            resp = self.api.delete_trainings(
                data=data,
                headers=headers
            )
            assert resp.status_code == 204, resp.dumps()

    def test_1_train_job_create(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        checkpoints = self.api.list_checkpoints(headers=headers).json()["data"]["checkpoints"]

        for checkpoint in checkpoints:
            if checkpoint['type'] == "Stable-diffusion":
                s3_model_path = checkpoint['s3Location'] + "/" + checkpoint['name'][0]
                s3_data_path = "s3://elonniu/dataset/10_technic"

                payload = {
                    "lora_train_type": "kohya",
                    "params": {
                        "training_params": {
                            "training_instance_type": "ml.g5.2xlarge",
                            "s3_model_path": s3_model_path,
                            "s3_data_path": s3_data_path
                        },
                        "config_params": {
                            "saving_arguments": {
                                "output_name": "lego_technic_4",
                                "save_every_n_epochs": 1000
                            },
                            "training_arguments": {
                                "max_train_epochs": 100
                            }
                        }
                    }
                }

                resp = self.api.create_training_job(headers=headers, data=payload)
                assert resp.status_code == 201, resp.dumps()

    def test_3_all_trains(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        assert 'trainJobs' in resp.json()["data"]
        trainJobs = resp.json()["data"]["trainJobs"]
        assert len(trainJobs) > 0
        for trainJob in trainJobs:
            print(trainJob)

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_2_train_stop(self):
        global train_job_id

        if not train_job_id:
            pass

        headers = {
            "x-api-key": config.api_key,
        }

        resp = self.api.stop_training_job(training_id=train_job_id, headers=headers)
        assert resp.status_code == 200, resp.dumps()
