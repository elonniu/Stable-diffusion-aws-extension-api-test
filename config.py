import base64
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

host_url = os.environ.get("API_GATEWAY_URL")
if not host_url:
    raise Exception("API_GATEWAY_URL is empty")

region_name = host_url.split('.')[2]
if not region_name:
    raise Exception("API_GATEWAY_URL is invalid")

# Remove "/prod" or "/prod/" from the end of the host_url
host_url = host_url.replace("/prod/", "")
host_url = host_url.replace("/prod", "")
if host_url.endswith("/"):
    host_url = host_url[:-1]
logger.info(f"config.host_url: {host_url}")

api_key = os.environ.get("API_GATEWAY_URL_TOKEN")
if not api_key:
    raise Exception("API_GATEWAY_URL_TOKEN is empty")
logger.info(f"config.api_key: {api_key}")

username = "admin"
logger.info(f"config.username: {username}")

bearer_token = f'Bearer {base64.b16encode(username.encode("utf-8")).decode("utf-8")}'

bucket = os.environ.get("API_BUCKET")
if not bucket:
    raise Exception("API_BUCKET is empty")
logger.info(f"config.bucket: {bucket}")

test_fast = os.environ.get("TEST_FAST") == "true"
logger.info(f"config.test_fast: {test_fast}")

is_gcr = region_name.startswith("cn-")
logger.info(f"config.is_gcr: {is_gcr}")

role_name = "role_name"
logger.info(f"config.role_name: {role_name}")

endpoint_name = "api-test"
logger.info(f"config.endpoint_name: {endpoint_name}")

dataset_name = "huahua"
logger.info(f"config.dataset_name: {dataset_name}")

model_name = "test-model"
logger.info(f"config.model_name: {model_name}")

async_instance_type = os.environ.get("ASYNC_INSTANCE_TYPE", "ml.g4dn.xlarge")
logger.info(f"config.async_instance_type: {async_instance_type}")

real_time_instance_type = os.environ.get("REAL_TIME_INSTANCE_TYPE", "ml.g5.2xlarge")
logger.info(f"config.real_time_instance_type: {real_time_instance_type}")

initial_instance_count = "2"
if is_gcr:
    initial_instance_count = "1"
logger.info(f"config.initial_instance_count: {initial_instance_count}")

default_model_id = "v1-5-pruned-emaonly.safetensors"
logger.info(f"config.default_model_id: {default_model_id}")

ckpt_message = "placeholder for chkpts upload test"
logger.info(f"config.ckpt_message: {ckpt_message}")
