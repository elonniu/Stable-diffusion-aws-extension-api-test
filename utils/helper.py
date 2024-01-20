import io
import json
import logging
import math
import os
import subprocess
import tarfile

import boto3
import requests

import config as config
from utils.api import Api
from utils.enums import InferenceStatus

logger = logging.getLogger(__name__)
ddb_client = boto3.client('dynamodb')
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')


def get_parts_number(local_path: str):
    file_size = os.stat(local_path).st_size
    part_size = 1000 * 1024 * 1024
    return math.ceil(file_size / part_size)


def wget_file(local_file: str, url: str, gcr_url: str = None):
    if gcr_url is not None and config.is_gcr:
        url = gcr_url
    if not os.path.exists(local_file):
        local_path = os.path.dirname(local_file)
        logger.info(f"Downloading {url}")
        wget_process = subprocess.run(['wget', '-qP', local_path, url], capture_output=True)
        if wget_process.returncode != 0:
            raise subprocess.CalledProcessError(wget_process.returncode, 'wget failed')


def get_test_model():
    models = ddb_client.scan(
        TableName="ModelTable",
    )
    return models


def clear_checkpoint(message: str):
    table = "CheckpointTable"
    checkpoints = ddb_client.scan(
        TableName=table
    )
    for checkpoint in checkpoints['Items']:
        if 'params' not in checkpoint:
            logger.info(f"checkpoint no params: {checkpoint}")
            continue
        if 'message' not in checkpoint["params"]['M']:
            logger.info(f"checkpoint no message: {checkpoint}")
            continue
        if checkpoint["params"]['M']['message']['S'] != message:
            continue
        logger.info(f"Deleting {checkpoint}")
        delete_s3_location_object(checkpoint["s3_location"]["S"])
        ddb_client.delete_item(
            TableName=table,
            Key={
                'id': checkpoint["id"],
            }
        )


def clear_dataset_info(table: str, dataset_name: str):
    response = ddb_client.scan(
        TableName=table,
        FilterExpression="dataset_name = :dataset_name",
        ExpressionAttributeValues={
            ":dataset_name": {"S": dataset_name}
        }
    )
    for item in response["Items"]:
        logger.info(f"Deleting {item}")
        ddb_client.delete_item(
            TableName=table,
            Key={
                'dataset_name': item["dataset_name"],
            }
        )


def upload_db_config(s3_presign_url: str):
    config_json = open("./data/train/db_config_cloud.json", "rb")
    config_json = json.loads(config_json.read())

    config_json['concepts_list'][0]["instance_data_dir"] = f"{config.bucket}-dataset-{config.dataset_name}"
    config_json["model_dir"] = f"models/dreambooth/{config.model_name}"
    config_json["pretrained_model_name_or_path"] = f"models/dreambooth/{config.model_name}/working"
    config_json["model_name"] = config.model_name

    logger.info(config_json)

    file = create_tar(
        json.dumps(config_json),
        f"models/sagemaker_dreambooth/{config.model_name}/db_config_cloud.json"
    )

    response = requests.put(
        s3_presign_url,
        file
    )
    response.raise_for_status()


def create_tar(json_string: str, path: str):
    with io.BytesIO() as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            json_bytes = json_string.encode("utf-8")
            json_buffer = io.BytesIO(json_bytes)
            tarinfo = tarfile.TarInfo(name=path)
            tarinfo.size = len(json_bytes)
            tar.addfile(tarinfo, json_buffer)
            return tar_buffer.getvalue()


def delete_dataset_item(table: str, dataset_name: str):
    response = ddb_client.scan(
        TableName=table,
        FilterExpression="dataset_name = :dataset_name",
        ExpressionAttributeValues={
            ":dataset_name": {"S": dataset_name}
        }
    )
    for item in response["Items"]:
        logger.info(f"Deleting {item}")
        ddb_client.delete_item(
            TableName=table,
            Key={
                'dataset_name': item["dataset_name"],
                'sort_key': item["sort_key"]
            }
        )


def list_endpoints(api_instance):
    headers = {
        "x-api-key": config.api_key,
        "Authorization": config.bearer_token
    }
    resp = api_instance.list_endpoints(headers=headers)
    endpoints = resp.json()['data']["endpoints"]
    return endpoints


def get_endpoint_status(api_instance, endpoint_name: str):
    endpoints = list_endpoints(api_instance)
    for endpoint in endpoints:
        if endpoint['endpoint_name'] == endpoint_name:
            return endpoint['endpoint_status']
    return None


def get_inference_job_status_new(api_instance, job_id):
    resp = api_instance.get_inference_job_new(
        job_id=job_id,
        headers={
            "x-api-key": config.api_key,
            "Authorization": config.bearer_token
        },
    )

    if InferenceStatus.FAILED.value == resp.json()['data']['status']:
        logger.error(f"Failed inference: {resp.json()['data']}")

    return resp.json()['data']['status']


def delete_sagemaker_endpoint(api_instance):
    headers = {
        "x-api-key": config.api_key,
        "Authorization": config.bearer_token
    }

    data = {
        "endpoint_name_list": [
            f"esd-async-{config.endpoint_name}",
            f"esd-real-time-{config.endpoint_name}",
        ],
        "username": config.username
    }

    resp = api_instance.delete_endpoints(headers=headers, data=data)
    assert resp.status_code == 204, resp.dumps()


def delete_train_item():
    trains = ddb_client.scan(
        TableName="TrainingTable",
    )
    for train in trains["Items"]:
        model_name = train["params"]['M']['training_params']['M']['model_name']['S']
        if model_name != config.model_name:
            continue

        delete_s3_location_object(train["input_s3_location"]['S'])
        ddb_client.delete_item(TableName="CheckpointTable", Key={'id': train["checkpoint_id"]})
        ddb_client.delete_item(TableName="TrainingTable", Key={'id': train["id"]})


def delete_prefix_in_s3(prefix: str):
    if prefix.startswith("s3://"):
        prefix = prefix.replace(f"s3://{config.bucket}/", "")
    bucket = s3.Bucket(config.bucket)
    bucket.objects.filter(Prefix=prefix).delete()


def delete_inference_jobs(inference_id_list: [str]):
    api = Api(config)

    data = {
        "inference_id_list": inference_id_list,
    }

    api.delete_inferences(
        data=data,
        headers={
            "x-api-key": config.api_key,
        },
    )


def delete_s3_location_object(location: str):
    logger.info(f"Deleting {config.bucket} {location}")
    if location.startswith("s3://"):
        location = location.replace(f"s3://{config.bucket}/", "")
        logger.info(f"Deleting {config.bucket} {location}")
    s3_client.delete_object(Bucket=config.bucket, Key=location)


def upload_with_put(s3_url, local_file):
    with open(local_file, 'rb') as data:
        response = requests.put(s3_url, data=data)
        response.raise_for_status()


def upload_multipart_file(signed_urls, local_path):
    with open(local_path, "rb") as f:
        parts = []

        for i, signed_url in enumerate(signed_urls):
            part_size = 1000 * 1024 * 1024
            file_data = f.read(part_size)
            response = requests.put(signed_url, data=file_data)
            response.raise_for_status()
            etag = response.headers['ETag']
            parts.append({
                'ETag': etag,
                'PartNumber': i + 1
            })
            print(f'model upload part {i + 1}: {response}')

        return parts
