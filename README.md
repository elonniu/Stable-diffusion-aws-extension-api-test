# Stable-diffusion-aws-extension-api-test

## Description

This project helps you quickly build a test environment with the **same structure** as **test pipeline** in your local
environment.

# References

You may need to refer to the following documents:

- [API-Automation-Test-Framework-Design](https://code.amazon.com/packages/Stable-diffusion-aws-extension-api-test/trees/mainline)

- Test locally
- **Note**: You must test and successfully complete on the `dev` branch of `stable-diffusion-aws-extension` before you
  can commit the code.

## Clone Test Repo

Before you clone code, you need to run `mwinit` to verify that you have the correct permissions, You must use `ecdsa` to
generate the key pair, and then use the public key to run the following command:

```bash
mwint -k ~/.ssh/id_ecdsa.pub
```

```bash
git clone ssh://git.amazon.com/pkg/Stable-diffusion-aws-extension-api-test
cd Stable-diffusion-aws-extension-api-test
```

# Directory Struct

```agsl
.. data
..... src
........ stable_diffusion_aws_extension_api_test
........... test_case
.............. test_1_unit # Unit test
.............. test_2_e2e # E2E/Functional tests
```

## Build Test Environment

If the shell fails, please refer to the `local_build.sh` file and execute the commands step by step.

```bash
make build
```

## Setup Environment Variables

Create `.env` file with the following content:

```bash
API_GATEWAY_URL=https://{apiId}.execute-api.{region}.amazonaws.com/prod/
API_GATEWAY_URL_TOKEN={apiToken}
API_BUCKET=elonniu
```

## Run Test

```bash
make test
```

## Run Specific Directory/File/Class/Case

```bash
# test directory
make test test_1_unit/
# test file
make test test_1_unit/test_1_connection.py
# test class
make test test_1_unit/test_1_connection.py::TestConnectUnit
# test case
make test test_1_unit/test_1_connection.py::TestConnectUnit::test_test_connection_get_without_key
# test -k
make testk <your-keyword>
```
