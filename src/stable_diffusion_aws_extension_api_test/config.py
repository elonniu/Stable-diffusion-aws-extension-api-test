import base64
import logging
import os

import shortuuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

uuid = shortuuid.ShortUUID().random(length=8).lower()

host_url = os.environ.get("API_GATEWAY_URL")
if not host_url:
    raise Exception("API_GATEWAY_URL is empty")

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

username = os.environ.get("API_USERNAME")
if not username:
    raise Exception("API_USERNAME is empty")
logger.info(f"config.username: {username}")

bearer_token = f'Bearer {base64.b16encode(username.encode("utf-8")).decode("utf-8")}'

bucket = os.environ.get("API_BUCKET")
if not bucket:
    raise Exception("API_BUCKET is empty")
logger.info(f"config.bucket: {bucket}")

region_name = os.environ.get("API_REGION")
if not region_name:
    raise Exception("API_REGION is empty")
logger.info(f"config.region_name: {region_name}")

fast_test = os.environ.get("FAST_TEST") == "true"
logger.info(f"config.fast_test: {fast_test}")

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

instance_type = "ml.g4dn.2xlarge"
logger.info(f"config.instance_type: {instance_type}")

initial_instance_count = "2"
if is_gcr:
    initial_instance_count = "1"
logger.info(f"config.initial_instance_count: {initial_instance_count}")

default_model_id = "v1-5-pruned-emaonly.safetensors"
logger.info(f"config.default_model_id: {default_model_id}")

ckpt_message = "placeholder for chkpts upload test"
logger.info(f"config.ckpt_message: {ckpt_message}")
