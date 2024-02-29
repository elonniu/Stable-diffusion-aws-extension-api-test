from __future__ import print_function

import json
import logging
from time import sleep

import config as config
from utils.api import Api
from utils.enums import InferenceType

logger = logging.getLogger(__name__)

filename = "v1-5-pruned-emaonly.safetensors"


class TestInferenceOneApiAsyncE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_inference_one_api_async(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        payload = {
            "inference_type": "Async",
            "task_type": InferenceType.TXT2IMG.value,
            "models": {
                "Stable-diffusion": [filename],
                "embeddings": []
            },

        }

        with open("./data/api_params/txt2img_api_param.json", 'rb') as data:
            data = json.load(data)
            if 's_tmax' in data:
                data["s_tmax"] = 'Infinity'
            payload["endpoint_payload"] = data
            resp = self.api.create_inference(data=payload, headers=headers)
            assert resp.status_code == 202, resp.dumps()
            inference_id = resp.json()['data']['inference']['inference_id']
            sleep(10)
            resp = self.api.get_inference_job(job_id=inference_id, headers=headers)
            assert resp.status_code == 200, resp.dumps()
            result = resp.json()['data']
            assert 'img_presigned_urls' in result, f'img_presigned_urls not found in {result}'
