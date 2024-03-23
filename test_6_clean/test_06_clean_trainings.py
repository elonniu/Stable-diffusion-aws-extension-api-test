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

    def test_1_clean_trainings(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        resp = self.api.list_trainings(headers=headers)
        data = resp.json()['data']

        assert 'trainings' in data, resp.dumps()

        trainings = data['trainings']
        for training in trainings:
            data = {
                "training_id_list": [training['id']],
            }

            resp = self.api.delete_trainings(headers=headers, data=data)
            assert resp.status_code == 204, resp.dumps()
