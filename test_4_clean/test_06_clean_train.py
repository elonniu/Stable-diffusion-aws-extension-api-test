from __future__ import print_function

import logging

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanTrainings:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_0_clean_test_trainings(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_trainings(headers=headers)
        assert resp.json()['statusCode'] == 200, resp.dumps()

        assert 'items' in resp.json()['data'], resp.dumps()

        items = resp.json()['data']['items']

        for item in items:
            data = {
                "training_id_list": [item['id']],
            }

            resp = self.api.delete_trainings(headers=headers, data=data)
            assert resp.status_code == 204
