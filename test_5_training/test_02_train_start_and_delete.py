from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)


class TestTrainStartDeleteE2E:
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
        assert 'trainings' in resp.json()["data"]
        trainJobs = resp.json()["data"]["trainings"]
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

        payload = {
            "lora_train_type": "kohya",
            "params": {
                "training_params": {
                    "training_instance_type": "ml.g5.2xlarge",
                    "model": config.default_model_id,
                    "dataset": config.dataset_name,
                    "fm_type": "sd_1_5"
                },
                "config_params": {
                    "saving_arguments": {
                        "output_name": "model_name",
                        "save_every_n_epochs": 1
                    },
                    "training_arguments": {
                        "max_train_epochs": 1
                    }
                }
            }
        }

        resp = self.api.create_training_job(headers=headers, data=payload)
        assert resp.status_code == 201, resp.dumps()

    def test_3_delete_all_trains_jobs(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        assert 'trainings' in resp.json()["data"]
        train_jobs = resp.json()["data"]["trainings"]
        assert len(train_jobs) > 0
        for trainJob in train_jobs:
            data = {
                "training_id_list": [trainJob["id"]],
            }
            resp = self.api.delete_trainings(
                data=data,
                headers=headers
            )
            assert resp.status_code == 204, resp.dumps()
