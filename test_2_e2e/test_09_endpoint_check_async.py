from __future__ import print_function

import logging
import time
from datetime import datetime
from datetime import timedelta

import config as config
from utils.api import Api

logger = logging.getLogger(__name__)

endpoint_name = f"esd-async-{config.endpoint_name}"


class TestEndpointCheckE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_list_endpoints_status(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_endpoints(headers=headers, params=params)
        assert resp.status_code == 200, resp.dumps()

        endpoints = resp.json()['data']["endpoints"]
        assert len(endpoints) >= 0

        assert endpoint_name in [endpoint["endpoint_name"] for endpoint in endpoints]

        timeout = datetime.now() + timedelta(minutes=30)

        while datetime.now() < timeout:
            result = self.endpoints_wait_for_in_service()
            if result:
                break
            time.sleep(50)
        else:
            raise Exception("Function execution timed out after 30 minutes.")

    def endpoints_wait_for_in_service(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        params = {
            "username": config.username
        }

        resp = self.api.list_endpoints(headers=headers, params=params)
        assert resp.status_code == 200, resp.dumps()

        for endpoint in resp.json()['data']["endpoints"]:
            if endpoint["endpoint_name"] == endpoint_name:
                if endpoint["endpoint_status"] != "InService":
                    logger.info(f"{endpoint_name} is {endpoint['endpoint_status']}")
                    return False
                else:
                    return True

        return False

    # not support a same role create more than one endpoint
    def test_2_create_endpoint_role_limit(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "endpoint_name": config.endpoint_name,
            "endpoint_type": "Async",
            "instance_type": config.async_instance_type,
            "initial_instance_count": 1,
            "autoscaling_enabled": False,
            "assign_to_roles": ["IT Operator"],
            "creator": config.username
        }

        resp = self.api.create_endpoint(headers=headers, data=data)
        assert resp.status_code == 400, resp.dumps()

        assert resp.json()["statusCode"] == 400
        assert 'not allow to have another one' in resp.json()["message"]
