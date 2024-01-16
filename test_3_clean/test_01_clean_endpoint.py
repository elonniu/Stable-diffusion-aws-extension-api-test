from __future__ import print_function

import logging
from time import sleep

import config as config
from utils.api import Api
from utils.helper import get_endpoint_status, delete_sagemaker_endpoint_new

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCleanEndpoint:
    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_delete_endpoints(self):
        endpoint_name = f"esd-async-{config.endpoint_name}"
        while True:
            status = get_endpoint_status(self.api, endpoint_name)
            if status is None:
                break

            if status in ['Creating', 'Updating']:
                logger.error(f"Endpoint {endpoint_name} is {status}, waiting to delete...")
                sleep(10)
            else:
                delete_sagemaker_endpoint_new(self.api)
                break
        pass
