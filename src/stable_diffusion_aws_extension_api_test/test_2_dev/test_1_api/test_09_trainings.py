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

    def test_1_start_training_job_without_key(self):
        resp = self.api.start_training_job(training_id="id")

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_2_start_training_job_with_bad_params(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "train_type": "Stable-diffusion",
            "model_id": "bad-2222-3333-1111-b3f0c1c21cee"
        }

        resp = self.api.start_training_job(training_id="id", headers=headers, data=data)

        assert resp.status_code == 400
        assert 'object has missing required properties' in resp.json()["message"]
        assert 'status' in resp.json()["message"]

    def test_3_start_training_job_with_bad_id(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "status": "Training"
        }

        training_id = "train_job_id"

        resp = self.api.start_training_job(training_id=training_id, headers=headers, data=data)

        assert resp.status_code == 404
        assert resp.json()["statusCode"] == 404
        assert resp.json()["message"] == f"no such train job with id({training_id})"

    def test_4_create_training_job_without_key(self):
        resp = self.api.create_training_job()

        assert resp.status_code == 403
        assert resp.json()["message"] == "Forbidden"

    def test_5_list_trainings_without_key(self):
        resp = self.api.list_trainings()

        assert resp.status_code == 401
        assert resp.json()["message"] == "Unauthorized"

    def test_6_list_trainings(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200
        assert resp.json()["statusCode"] == 200
        assert len(resp.json()['data']["trainJobs"]) >= 0

    def test_7_delete_trainings_with_bad_request_body(self):
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

    def test_8_delete_trainings_succeed(self):
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
