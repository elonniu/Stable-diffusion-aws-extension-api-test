from __future__ import print_function

import json
import logging
import time
from datetime import datetime
from datetime import timedelta

import requests
import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.enums import InferenceStatus, InferenceType
from stable_diffusion_aws_extension_api_test.utils.helper import get_inference_job_status, delete_inference_job

logger = logging.getLogger(__name__)

inference_data = {}


class TestSLaTxt2Img:

    def setup_class(self):

        self.api = Api(config=config, debug=False)

    @classmethod
    def teardown_class(cls):
        pass

    def test_1_sla_txt2img(self):

        with open("./data/sla/prompts.txt", "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
            prompts = [prompt for prompt in prompts if prompt != ""]
            prompts_count = len(prompts)
            duration_list = []
            result_list = []
            for prompt in prompts:
                result, duration = self.sla_job(prompt)
                result_list.append(result)
                if result:
                    duration_list.append(duration)

            max_duration_seconds = max(duration_list)
            min_duration_seconds = min(duration_list)
            avg_duration_seconds = sum(duration_list) / len(duration_list)
            success_rate = result_list.count(True) / len(result_list)

            json_result = {
                "model_id": config.default_model_id,
                "instance_type": config.instance_type,
                "instance_count": config.initial_instance_count,
                "count": prompts_count,
                "succeed": result_list.count(True),
                "failed": result_list.count(False),
                "success_rate": success_rate,
                "max_duration": max_duration_seconds,
                "min_duration": min_duration_seconds,
                "avg_duration": avg_duration_seconds
            }

            with open("/tmp/txt2img_sla_report.json", "w") as sla_report:
                sla_report.write(json.dumps(json_result))

            logger.info(json_result)

    def sla_job(self, prompt: str):
        # get start time
        start_time = datetime.now()
        result = False
        try:
            result = self.start_job(prompt)
        except Exception as e:
            logger.info(f"Error: {e}")

        end_time = datetime.now()

        duration = (end_time - start_time).seconds
        logger.info(f"result:{result} duration:{duration} prompt:{prompt}")
        return result, duration

    def start_job(self, prompt: str):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "user_id": config.username,
            "task_type": InferenceType.TXT2IMG.value,
            "models": {
                "Stable-diffusion": [config.default_model_id],
                "embeddings": []
            },
            "filters": {
            }
        }

        resp = self.api.create_inference(headers=headers, data=data)

        if 'inference' not in resp.json():
            logger.error(resp.json())
            return False

        inference = resp.json()['inference']

        inference_id = inference["id"]

        with open("./data/api_params/txt2img_api_param.json", 'rb') as data:
            data = json.load(data)
            data["prompt"] = prompt
            response = requests.put(inference["api_params_s3_upload_url"], data=json.dumps(data))
            response.raise_for_status()

        result = self.sla_txt2img_inference_job_run_and_succeed(inference_id)

        delete_inference_job(inference_id)

        return result

    def sla_txt2img_inference_job_run_and_succeed(self, inference_id: str):

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.inference_run(job_id=inference_id, headers=headers)
        if 'statusCode' not in resp.json():
            logger.error(resp.json())
            return False

        if resp.json()['statusCode'] != 200:
            logger.error(resp.json())
            return False

        timeout = datetime.now() + timedelta(minutes=2)

        while datetime.now() < timeout:
            status = get_inference_job_status(
                api_instance=self.api,
                job_id=inference_id
            )
            if status == InferenceStatus.SUCCEED.value:
                return self.sla_txt2img_inference_job_image(inference_id)
            if status == InferenceStatus.FAILED.value:
                return False
            time.sleep(1)

        return False

    def sla_txt2img_inference_job_image(self, inference_id: str):
        global inference_data

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.get_inference_image_output(job_id=inference_id, headers=headers)
        return len(resp.json()) > 0
