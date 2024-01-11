from __future__ import print_function

import logging
import tarfile
import time
from datetime import datetime
from datetime import timedelta

import pytest
import config as config
from utils.api import Api
from utils.helper import clear_model_item, \
    upload_multipart_file

logger = logging.getLogger(__name__)

job_id = None
signed_urls = None


class TestModelE2E:

    def setup_class(self):
        self.api = Api(config)
        clear_model_item()
        pass

    @classmethod
    def teardown_class(cls):
        pass

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_1_model_v15_post(self):
        filename = "v1-5-pruned-emaonly.safetensors"
        model_type = "Stable-diffusion"

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "model_type": model_type,
            "name": config.model_name,
            "creator": config.username,
            "filenames": [
                {
                    "filename": filename,
                    "parts_number": 5
                }
            ],
            "params": {
                "create_model_params": {
                    "new_model_name": config.model_name,
                    "ckpt_path": filename,
                    "shared_src": "",
                    "from_hub": False,
                    "new_model_url": "",
                    "new_model_token": "",
                    "extract_ema": False,
                    "train_unfrozen": False,
                    "is_512": True
                }
            }
        }

        resp = self.api.create_model(headers=headers, data=data)

        assert resp.status_code == 201, resp.dumps()
        assert resp.json()["statusCode"] == 201
        job = resp.json()['data']["job"]
        assert job['model_type'] == model_type
        assert job['status'] == "Initial"
        assert len(job['id']) == 36
        global job_id
        job_id = job['id']
        global signed_urls
        signed_urls = resp.json()['data']["s3PresignUrl"][filename]
        s3_base = job["s3_base"]
        print(f"Upload to S3 {s3_base}")
        print(f"Model ID: {job_id}")

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_2_model_v15_put(self):
        filename = "v1-5-pruned-emaonly.safetensors"
        tar_filename = f"data/models/Stable-diffusion/{filename}.tar"
        print(f"Adding data/models/Stable-diffusion/{filename} to {tar_filename}")
        with tarfile.open(tar_filename, mode='w') as tar:
            tar.add(f"data/models/Stable-diffusion/{filename}")

        global signed_urls
        multiparts_tags = upload_multipart_file(signed_urls, tar_filename)

        global job_id

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        data = {
            "status": "Creating",
            "multi_parts_tags": {filename: multiparts_tags}
        }

        resp = self.api.update_model(model_id=job_id, headers=headers, data=data)
        assert resp.status_code == 200, resp.dumps()
        assert resp.json()["statusCode"] == 200
        assert resp.json()['data']["job"]["endpointName"] == "aigc-utils-endpoint"

    @pytest.mark.skipif(config.test_fast, reason="test_fast")
    def test_3_models_v15_check_list(self):
        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_models(headers=headers)

        global job_id
        models = resp.json()['data']["models"]
        assert job_id in [model["id"] for model in models]

        timeout = datetime.now() + timedelta(minutes=15)

        while datetime.now() < timeout:
            result = self.model_wait_for_complete()
            if result:
                break
            time.sleep(30)
        else:
            raise Exception("Function execution timed out after 15 minutes.")

    def model_wait_for_complete(self):
        global job_id

        headers = {
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        }

        resp = self.api.list_models(headers=headers)
        assert resp.status_code == 200, resp.dumps()
        models = resp.json()['data']["models"]
        for model in models:
            if model["id"] == job_id:
                print(f"Model {job_id} is {model['status']}...")
                if model["status"] == "Deleting":
                    return False
                if model["status"] == "Complete":
                    return True
                if model["status"] == "Fail":
                    logger.error("Model creation failed.")
                    logger.error(resp.dumps())
                    raise Exception("Model creation failed.")
                return False

        return False
