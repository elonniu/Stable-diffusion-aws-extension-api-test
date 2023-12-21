from __future__ import print_function

import logging

import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api

logger = logging.getLogger(__name__)


class TestTrainingsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_train_post_get_without_key(self):
        resp = self.api.start_train()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_2_train_post_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "train_type": "Stable-diffusion",
            "model_id": "bad-2222-3333-1111-b3f0c1c21cee"
        }

        resp = self.api.start_train(headers=headers, data=data)

        assert resp.status_code == 200

        assert resp.json()["errorMessage"] == "'status'"

    def test_3_train_put_bad_id(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "train_job_id": "train_job_id",
            "status": "Training"
        }

        resp = self.api.start_train(headers=headers, data=data)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 500
        assert resp.json()["error"] == "no such train job with id(train_job_id)"

    def test_4_trains_post_without_key(self):
        resp = self.api.create_train()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_5_trains_get_without_key(self):
        resp = self.api.list_trains()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_6_trains_get(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trains(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()["trainJobs"]) >= 0

    def test_7_trainings_delete_bad_request_body(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "bad": ['bad'],
        }

        resp = self.api.delete_trainings(headers=headers, data=data)
        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'training_job_list' in resp.json()["message"]

    def test_8_trainings_delete_succeed(self):
        headers = {
            "x-api-key": config.api_key,
        }

        data = {
            "training_job_list": ['id'],
        }

        resp = self.api.delete_trainings(headers=headers, data=data)
        assert resp.status_code == 200
        assert 'training jobs deleted' == resp.json()["message"]

    def test_9_get_training_job_without_key(self):
        headers = {
        }

        resp = self.api.get_training_job(job_id="no", headers=headers)
        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_10_get_training_job_not_found(self):
        headers = {
            "x-api-key": config.api_key,
        }

        job_id = "job_uuid"

        resp = self.api.get_training_job(job_id=job_id, headers=headers)
        assert resp.status_code == 404
        assert resp.json()["message"] == f"Job with id {job_id} not found"
