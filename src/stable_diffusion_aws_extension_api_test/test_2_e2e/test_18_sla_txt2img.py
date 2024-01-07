from __future__ import print_function

import json
import logging
import os
import time
from datetime import datetime
from datetime import timedelta

import pytest
import requests
import stable_diffusion_aws_extension_api_test.config as config
from stable_diffusion_aws_extension_api_test.utils.api import Api
from stable_diffusion_aws_extension_api_test.utils.enums import InferenceStatus, InferenceType
from stable_diffusion_aws_extension_api_test.utils.helper import get_inference_job_status_new

logger = logging.getLogger(__name__)

inference_data = {}
total_inference_count = 0
inference_index = 1

sla_batch_size = os.getenv("SLA_BATCH_SIZE", 5)

class TestSLaTxt2Img:

    def setup_class(self):
        self.api = Api(config=config)

    @classmethod
    def teardown_class(cls):
        pass

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_1_sla_txt2img(self):

        with open("./data/sla/prompts.txt", "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
            prompts = [prompt for prompt in prompts if prompt != ""]
            prompts = prompts[:sla_batch_size]
            prompts_count = len(prompts)
            global total_inference_count, inference_index
            total_inference_count = prompts_count
            total_duration_list = []
            result_list = []
            failed_list = []
            create_infer_duration_list = []
            upload_duration_list = []
            wait_duration_list = []

            for prompt in prompts:
                result, total_duration, inference_id, create_infer_duration, upload_duration, wait_duration = self.sla_job(
                    prompt)
                logger.info(f"SLA progress: {inference_index}/{total_inference_count}")
                inference_index += 1

                result_list.append(result)
                if result:
                    total_duration_list.append(total_duration)
                else:
                    failed_list.append(inference_id)
                if create_infer_duration:
                    create_infer_duration_list.append(create_infer_duration)
                if upload_duration:
                    upload_duration_list.append(upload_duration)
                if wait_duration:
                    wait_duration_list.append(wait_duration)

            if len(total_duration_list) > 0:
                max_duration_seconds = max(total_duration_list)
                min_duration_seconds = min(total_duration_list)
                avg_duration_seconds = sum(total_duration_list) / len(total_duration_list)
            else:
                max_duration_seconds = 0
                min_duration_seconds = 0
                avg_duration_seconds = 0

            if len(create_infer_duration_list) > 0:
                create_infer_duration_avg = sum(create_infer_duration_list) / len(create_infer_duration_list)
            else:
                create_infer_duration_avg = 0

            if len(upload_duration_list) > 0:
                upload_duration_avg = sum(upload_duration_list) / len(upload_duration_list)
            else:
                upload_duration_avg = 0

            if len(wait_duration_list) > 0:
                wait_duration_avg = sum(wait_duration_list) / len(wait_duration_list)
            else:
                wait_duration_avg = 0

            if len(result_list) > 0:
                success_rate = result_list.count(True) / len(result_list)
                succeed = result_list.count(True)
                failed = result_list.count(False)
            else:
                success_rate = 0
                succeed = 0
                failed = 0

            if len(failed_list) > 0:
                failed_list_string = "failed_list:" + "\\n" + str(failed_list)
            else:
                failed_list_string = ""

            json_result = {
                "model_id": config.default_model_id,
                "instance_type": config.instance_type,
                "instance_count": int(config.initial_instance_count),
                "count": prompts_count,
                "succeed": succeed,
                "failed": failed,
                "success_rate": success_rate,
                "duration_total_max": max_duration_seconds,
                "duration_total_min": min_duration_seconds,
                "duration_total_avg": avg_duration_seconds,
                "failed_list": failed_list_string,
                "duration_avg_create": create_infer_duration_avg,
                "duration_avg_upload": upload_duration_avg,
                "duration_avg_wait_result": wait_duration_avg,
            }

            with open("/tmp/txt2img_sla_report.json", "w") as sla_report:
                sla_report.write(json.dumps(json_result))

            logger.warning(json.dumps(json_result, indent=4, sort_keys=True))

    def sla_job(self, prompt: str):
        start_time = datetime.now()

        result, inference_id, create_infer_duration, upload_duration, wait_duration = self.start_job(prompt)

        end_time = datetime.now()

        total_duration = (end_time - start_time).seconds

        return result, total_duration, inference_id, create_infer_duration, upload_duration, wait_duration

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

        create_infer_start_time = datetime.now()
        resp = self.api.create_inference(headers=headers, data=data)
        create_infer_end_time = datetime.now()
        create_infer_duration = (create_infer_end_time - create_infer_start_time).seconds

        assert 'data' in resp.json(), resp.dumps()
        assert 'inference' in resp.json()['data'], resp.dumps()

        inference = resp.json()['data']['inference']

        inference_id = inference["id"]

        upload_start_time = datetime.now()
        with open("./data/api_params/txt2img_api_param.json", 'rb') as data:
            data = json.load(data)
            data["prompt"] = prompt
            response = requests.put(inference["api_params_s3_upload_url"], data=json.dumps(data))
            response.raise_for_status()
        upload_end_time = datetime.now()
        upload_duration = (upload_end_time - upload_start_time).seconds

        wait_start_time = datetime.now()
        result = self.sla_txt2img_inference_job_run_and_succeed(inference_id)
        wait_end_time = datetime.now()
        wait_duration = (wait_end_time - wait_start_time).seconds

        return result, inference_id, create_infer_duration, upload_duration, wait_duration

    def sla_txt2img_inference_job_run_and_succeed(self, inference_id: str):

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.start_inference_job(job_id=inference_id, headers=headers)
        assert 'statusCode' in resp.json(), resp.dumps()
        assert resp.json()['statusCode'] == 202, resp.dumps()

        timeout = datetime.now() + timedelta(minutes=2)

        while datetime.now() < timeout:
            status = get_inference_job_status_new(
                api_instance=self.api,
                job_id=inference_id
            )
            if status == InferenceStatus.SUCCEED.value:
                return self.sla_txt2img_inference_job_image(inference_id)
            if status == InferenceStatus.FAILED.value:
                return False
            time.sleep(0.5)

        return False

    def sla_txt2img_inference_job_image(self, inference_id: str):
        global inference_data

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.get_inference_job(job_id=inference_id, headers=headers)
        return resp.json()['data']['status'] == 'succeed'
