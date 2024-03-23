set -euxo pipefail

export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export API_BUCKET=esd-test-$ACCOUNT_ID-$AWS_DEFAULT_REGION-$CODEBUILD_BUILD_NUMBER

# aws logs describe-log-groups | jq -r '.logGroups[].logGroupName' | grep -v codebuild | xargs -I {} aws logs delete-log-group --log-group-name {}

aws cloudformation delete-stack --stack-name "$STACK_NAME"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"

echo "----------------------------------------------------------------"
echo "phases -> build -> commands"
echo "----------------------------------------------------------------"
python --version
sudo yum install wget -y

if [ "$DEPLOY_STACK" = "cdk" ]; then
   echo "----------------------------------------------------------------"
   echo "cdk deploy start..."
   echo "----------------------------------------------------------------"
   curl -L -o esd.zip "$CODE_REPO/archive/refs/heads/$CODE_BRANCH.zip"
   unzip -q esd.zip

   pushd "stable-diffusion-aws-extension-$CODE_BRANCH/infrastructure"

   npm i -g pnpm
   pnpm i
   STARTED_TIME=$(date +%s)
   npx cdk deploy --parameters Email=example@amazon.com \
                  --parameters Bucket="$API_BUCKET" \
                  --parameters LogLevel=INFO \
                  --parameters SdExtensionApiKey=09876743210987654322 \
                  --require-approval never
   FINISHED_TIME=$(date +%s)
   export DEPLOY_DURATION_TIME=$(( $FINISHED_TIME - $STARTED_TIME ))
   sleep $SLEEP_AFTER_DEPLOY
   popd
fi

if [ "$DEPLOY_STACK" = "template" ]; then
   echo "----------------------------------------------------------------"
   echo "template deploy start..."
   echo "----------------------------------------------------------------"
   STARTED_TIME=$(date +%s)
   aws cloudformation create-stack --stack-name "$STACK_NAME" \
                                   --template-url "$TEMPLATE_FILE" \
                                   --capabilities CAPABILITY_NAMED_IAM \
                                   --parameters ParameterKey=Email,ParameterValue=example@example.com \
                                                ParameterKey=Bucket,ParameterValue="$API_BUCKET" \
                                                ParameterKey=LogLevel,ParameterValue=INFO \
                                                ParameterKey=SdExtensionApiKey,ParameterValue=09876743210987654322

   aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
   FINISHED_TIME=$(date +%s)
   export DEPLOY_DURATION_TIME=$(( $FINISHED_TIME - $STARTED_TIME ))
   sleep $SLEEP_AFTER_DEPLOY
fi

echo "----------------------------------------------------------------"
echo "Get api gateway url & token"
echo "----------------------------------------------------------------"
stack_info=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME")
export API_GATEWAY_URL=$(echo $stack_info | jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="ApiGatewayUrl").OutputValue')
export API_GATEWAY_URL_TOKEN=$(echo $stack_info | jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="ApiGatewayUrlToken").OutputValue')
echo "API_GATEWAY_URL: $API_GATEWAY_URL"
echo "API_GATEWAY_URL_TOKEN: $API_GATEWAY_URL_TOKEN"

echo "----------------------------------------------------------------"
echo "Download & Build SDE test case"
echo "----------------------------------------------------------------"
wget "$TEST_REPO/archive/refs/heads/$TEST_BRANCH.zip"
unzip -q "$TEST_BRANCH.zip"
mv "esd-api-test-$TEST_BRANCH" esd-api-test
cd esd-api-test
make build

echo "----------------------------------------------------------------"
echo "Running pytest..."
echo "----------------------------------------------------------------"
STARTED_TIME=$(date +%s)
source venv/bin/activate
pytest ./ --exitfirst -rA --log-cli-level=$TEST_LOG_LEVEL --json-report --json-report-summary --json-report-file=detailed_report.json --html="report-${CODEBUILD_BUILD_NUMBER}.html" --self-contained-html --continue-on-collection-errors
FINISHED_TIME=$(date +%s)
export TEST_DURATION_TIME=$(( $FINISHED_TIME - $STARTED_TIME ))
