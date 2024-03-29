import json
import logging

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_json(data):
    try:
        # if data is string
        if isinstance(data, str):
            return json.dumps(json.loads(data), indent=4)
        # if data is object
        if isinstance(data, dict):
            json.dumps(dict(data), indent=4)
        return str(data)
    except TypeError:
        return str(data)


class Api:

    def __init__(self, config):
        self.config = config

    def req(self, method: str, path: str, headers=None, data=None, params=None):
        if data is not None:
            data = json.dumps(data)

        url = f"{self.config.host_url}/prod/{path}"

        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            params=params,
            timeout=(30, 40)
        )

        dump_string = ""
        if headers:
            dump_string += f"\nRequest headers: {get_json(headers)}"
        if data:
            dump_string += f"\nRequest data: {get_json(data)}"
        if params:
            dump_string += f"\nRequest params: {get_json(params)}"
        if resp.status_code:
            dump_string += f"\nResponse status_code: {resp.status_code}"
        # if resp.headers:
        #     dump_string += f"\nResponse headers: {get_json(resp.headers)}"
        if resp.text:
            dump_string += f"\nResponse body: {get_json(resp.text)}"

        resp.dumps = lambda: logger.info(
            f"\n----------------------------"
            f"\n{method} {url}"
            f"{dump_string}"
            f"\n----------------------------"
        )

        return resp

    def test_connection(self, headers=None):
        return self.req(
            "GET",
            "inference/test-connection",
            headers=headers
        )

    def ping(self, headers=None):
        return self.req(
            "GET",
            "ping",
            headers=headers
        )

    def list_roles(self, headers=None, params=None):
        return self.req(
            "GET",
            "roles",
            headers=headers,
            params=params
        )

    def delete_roles(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "roles",
            headers=headers,
            data=data
        )

    def delete_datasets(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "datasets",
            headers=headers,
            data=data
        )

    def delete_models(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "models",
            headers=headers,
            data=data
        )

    def delete_trainings(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "trainings",
            headers=headers,
            data=data
        )

    def delete_inferences(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "inferences",
            headers=headers,
            data=data
        )

    def delete_checkpoints(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "checkpoints",
            headers=headers,
            data=data
        )

    def create_role(self, headers=None, data=None):
        return self.req(
            "POST",
            "roles",
            headers=headers,
            data=data
        )

    def list_users(self, headers=None, params=None):
        return self.req(
            "GET",
            "users",
            headers=headers,
            params=params
        )

    def delete_users(self, headers=None, data=None):
        return self.req(
            "DELETE",
            f"users",
            headers=headers,
            data=data
        )

    def create_user(self, headers=None, data=None):
        return self.req(
            "POST",
            "users",
            headers=headers,
            data=data
        )

    def list_checkpoints(self, headers=None, params=None):
        return self.req(
            "GET",
            "checkpoints",
            headers=headers,
            params=params
        )

    def create_checkpoint(self, headers=None, data=None):
        return self.req(
            "POST",
            "checkpoints",
            headers=headers,
            data=data
        )

    def upload_checkpoint(self, headers=None, data=None):
        return self.req(
            "POST",
            "upload_checkpoint",
            headers=headers,
            data=data
        )

    def update_checkpoint(self, checkpoint_id: str, headers=None, data=None):
        return self.req(
            "PUT",
            f"checkpoints/{checkpoint_id}",
            headers=headers,
            data=data
        )

    def delete_endpoints(self, headers=None, data=None):
        return self.req(
            "DELETE",
            "endpoints",
            headers=headers,
            data=data
        )

    def list_endpoints(self, headers=None, params=None):
        return self.req(
            "GET",
            "endpoints",
            headers=headers,
            params=params
        )

    def create_endpoint(self, headers=None, data=None):
        return self.req(
            "POST",
            "endpoints",
            headers=headers,
            data=data
        )

    def create_inference(self, headers=None, data=None):
        return self.req(
            "POST",
            "inferences",
            headers=headers,
            data=data
        )

    def start_inference_job(self, job_id: str, headers=None):
        return self.req(
            "PUT",
            f"inferences/{job_id}/start",
            headers=headers,
        )

    def get_training_job(self, job_id: str, headers=None):
        return self.req(
            "GET",
            f"trainings/{job_id}",
            headers=headers,
        )

    def get_inference_job(self, job_id: str, headers=None):
        return self.req(
            "GET",
            f"inferences/{job_id}",
            headers=headers
        )

    def get_texual_inversion_list_get(self, headers=None, params=None):
        return self.req(
            "GET",
            "inference/get-texual-inversion-list",
            headers=headers,
            params=params
        )

    def get_controlnet_model_list(self, headers=None, params=None):
        return self.req(
            "GET",
            "inference/get-controlnet-model-list",
            headers=headers,
            params=params,
        )

    def get_lora_list_get(self, headers=None, params=None):
        return self.req(
            "GET",
            "inference/get-lora-list",
            headers=headers,
            params=params
        )

    def get_hypernetwork_list(self, headers=None, params=None):
        return self.req(
            "GET",
            "inference/get-hypernetwork-list",
            headers=headers,
            params=params
        )

    def list_datasets(self, headers=None, params=None):
        return self.req(
            "GET",
            "datasets",
            headers=headers,
            params=params
        )

    def get_dataset(self, name: str, headers=None):
        return self.req(
            "GET",
            f"datasets/{name}",
            headers=headers
        )

    def create_dataset(self, headers=None, data=None):
        return self.req(
            "POST",
            "datasets",
            headers=headers,
            data=data
        )

    def update_dataset(self, dataset_id: str, headers=None, data=None):
        return self.req(
            "PUT",
            f"datasets/{dataset_id}",
            headers=headers,
            data=data
        )

    def create_model(self, headers=None, data=None):
        return self.req(
            "POST",
            "models",
            headers=headers,
            data=data
        )

    def update_model(self, model_id: str, headers=None, data=None):
        return self.req(
            "PUT",
            f"models/{model_id}",
            headers=headers,
            data=data
        )

    def list_models(self, headers=None, params=None):
        return self.req(
            "GET",
            "models",
            headers=headers,
            params=params
        )

    def create_training_job(self, headers=None, data=None):
        return self.req(
            "POST",
            "trainings",
            headers=headers,
            data=data
        )

    def list_trainings(self, headers=None, params=None):
        return self.req(
            "GET",
            "trainings",
            headers=headers,
            params=params
        )

    def list_inferences(self, headers=None, params=None):
        return self.req(
            "GET",
            "inferences",
            headers=headers,
            params=params
        )
