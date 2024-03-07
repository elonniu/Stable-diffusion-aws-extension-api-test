import json
import logging

import config as config
from utils.api import Api
from utils.enums import InferenceType

logger = logging.getLogger(__name__)

filename = "v1-5-pruned-emaonly.safetensors"
lora_filename = "lcm_lora_1_5.safetensors"


class TestInferenceOneApiLcm15RealTimeE2E:

    def setup_class(self):
        self.api = Api(config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_inference_one_api_lcm_15_real_time(self):
        headers = {
            "x-api-key": config.api_key,
            "username": config.username
        }

        payload = {
            "inference_type": "Real-time",
            "task_type": InferenceType.TXT2IMG.value,
            "models": {
                "Stable-diffusion": [filename],
                "Lora": [lora_filename],
                "embeddings": []
            },

        }

        with open("./data/api_params/txt2img_lcm_15_api_param.json", 'rb') as data:
            data = json.load(data)
            if 's_tmax' in data:
                data["s_tmax"] = 'Infinity'
            payload["endpoint_payload"] = data
            resp = self.api.create_inference(data=payload, headers=headers)
            assert resp.status_code == 200, resp.dumps()
            result = resp.json()['data']
            assert 'img_presigned_urls' in result, f'img_presigned_urls not found in {result}'
