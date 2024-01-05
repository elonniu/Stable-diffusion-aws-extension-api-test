set -e

if [ -d "venv" ]; then
    echo "You've set up a test environment. "
    echo "To prevent mis-operation, you can only rebuild by running the 'make rebuild'"
    exit 1
fi

python3 -m venv venv

if [ "$(uname)" == "Darwin" ]; then
    source venv/bin/activate
else
    . venv/bin/activate
fi

cd ../

curl -sSO https://aws-gcr-solutions.s3.amazonaws.com/Solution-data-generator/Solution-data-generator.zip
unzip -q Solution-data-generator.zip
cd Solution-data-generator
pip3 --default-timeout=6000 install -r requirements.txt
cd ../

curl -sSO https://aws-gcr-solutions.s3.amazonaws.com/Solution-api-test-framework/Solution-api-test-framework.zip
unzip -q Solution-api-test-framework.zip
cd Solution-api-test-framework
pip3 --default-timeout=6000 install -r requirements.txt
pip3 install -e ../Solution-data-generator
cd ../

cd Stable-diffusion-aws-extension-api-test
pip3 --default-timeout=6000 install -r src/stable_diffusion_aws_extension_api_test/requirements.txt
pip3 install pytest
pip3 install -e ../Solution-api-test-framework

rm -rf ../Solution-data-generator.zip
rm -rf ../Solution-api-test-framework.zip

echo "Done"
