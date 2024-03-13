from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)


class TestTrainingsApi:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_4_create_training_job_without_key(self):
        resp = self.api.create_training_job()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_5_list_trainings_without_key(self):
        resp = self.api.list_trainings()

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_6_list_trainings(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)

        assert resp.status_code == 200, resp.dumps()
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
        assert resp.status_code == 400, resp.dumps()

        assert 'object has missing required properties' in resp.json()["message"]
        assert 'training_id_list' in resp.json()["message"]

    def test_8_delete_trainings_succeed(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        data = {
            "training_id_list": ['id'],
        }

        resp = self.api.delete_trainings(headers=headers, data=data)
        assert resp.status_code == 204, resp.dumps()

    def test_9_get_training_job_without_key(self):
        headers = {
        }

        resp = self.api.get_training_job(job_id="no", headers=headers)
        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_10_get_training_job_not_found(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        job_id = "job_uuid"

        resp = self.api.get_training_job(job_id=job_id, headers=headers)
        assert resp.status_code == 404, resp.dumps()
        assert resp.json()["message"] == f"Job with id {job_id} not found"

    def test_11_stop_training_job_without_key(self):
        resp = self.api.stop_training_job(training_id="id")

        assert resp.status_code == 403, resp.dumps()
        assert resp.json()["message"] == "Forbidden"

    def test_12_stop_training_job_without_bad_id(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username,
        }

        resp = self.api.stop_training_job(training_id="id", headers=headers)

        assert resp.status_code == 404, resp.dumps()
