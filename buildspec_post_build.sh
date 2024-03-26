source env.properties

echo "----------------------------------------------------------------"
printenv

# if env ACCOUNT_ID not set, exit 1
if [ -z "$ACCOUNT_ID" ]; then
  echo "ACCOUNT_ID is not set"
  exit 1
fi

cd esd-api-test

ls -la

echo "----------------------------------------------------------------"
properties=("Account: $ACCOUNT_ID")
properties+=("Repo: $CODE_REPO")
properties+=("Version/Branch: $CODE_BRANCH")
properties+=("Region: $AWS_DEFAULT_REGION")
properties+=("Deploy Method: $DEPLOY_STACK")

if [ -n "$DEPLOY_DURATION_TIME" ]; then
  DEPLOY_DURATION_TIME=$(printf "%dm%ds\n" $(($DEPLOY_DURATION_TIME/60)) $(($DEPLOY_DURATION_TIME%60)))
  properties+=("Deploy Duration: $DEPLOY_DURATION_TIME")
fi


if [ "$CODEBUILD_BUILD_SUCCEEDING" -eq 0 ]; then
  result="Failed"
  CASE_PASSED_RESULT="Failed"
else
  result="Passed"
  CASE_PASSED_RESULT="Passed"
fi

properties+=("Result: ${result}")

if [ -f "detailed_report.json" ]; then
  CASE_TOTAL=$(cat detailed_report.json | jq -r '.summary.total')
  CASE_PASSED=$(cat detailed_report.json | jq -r '.summary.passed')
  properties+=("Total Cases: ${CASE_TOTAL}")
  properties+=("Passed Cases: ${CASE_PASSED}")
  CASE_PASSED_RESULT="Cases $CASE_PASSED Passed"
  CASE_SKIPPED=$(cat detailed_report.json | jq -r '.summary.skipped')
  if [ -n "$CASE_SKIPPED" ]; then
    properties+=("Skipped Cases: ${CASE_SKIPPED}")
  fi
fi

if [ -n "$API_TEST_STARTED_TIME" ]; then
  CURRENT_TIME=$(date +%s)
  API_TEST_DURATION_TIME=$(( $CURRENT_TIME - $API_TEST_STARTED_TIME ))
  API_TEST_DURATION_TIME=$(printf "%dm%ds\n" $(($API_TEST_DURATION_TIME/60)) $(($API_TEST_DURATION_TIME%60)))
  properties+=("Test Duration: ${API_TEST_DURATION_TIME}")
fi

if [ "$result" = "Passed" ]; then
  properties+=("G5 Instance: OK")
  properties+=("G4 Instance: OK")
  properties+=("train Task: OK")
  properties+=("Lora Task: OK")
  properties+=("txt2img Task: OK")
  properties+=("img2img Task: OK")
  properties+=("rembg Task: OK")
  properties+=("extra-single-image Task: OK")

  echo "----------------------------------------------------------------"
  echo "Remove the stack"
  echo "----------------------------------------------------------------"
  echo "Waiting for stack to be deleted..."
  STARTED_TIME=$(date +%s)
  aws cloudformation delete-stack --stack-name "$STACK_NAME"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"
  FINISHED_TIME=$(date +%s)
  REMOVE_DURATION_TIME=$(( $FINISHED_TIME - $STARTED_TIME ))
  REMOVE_DURATION_TIME=$(printf "%dm%ds\n" $(($REMOVE_DURATION_TIME/60)) $(($REMOVE_DURATION_TIME%60)))
  properties+=("Remove Stack Duration: ${REMOVE_DURATION_TIME}")

  if [ "$CLEAN_RESOURCES" = "yes" ]; then
     aws s3 rb "s3://$API_BUCKET" --force

     aws dynamodb delete-table --table-name "CheckpointTable" | jq
     aws dynamodb delete-table --table-name "DatasetInfoTable" | jq
     aws dynamodb delete-table --table-name "DatasetItemTable" | jq
     aws dynamodb delete-table --table-name "ModelTable" | jq
     aws dynamodb delete-table --table-name "MultiUserTable" | jq
     aws dynamodb delete-table --table-name "SDEndpointDeploymentJobTable" | jq
     aws dynamodb delete-table --table-name "SDInferenceJobTable" | jq
     aws dynamodb delete-table --table-name "TrainingTable" | jq

     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:failureCreateModel" | jq
     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:ReceiveSageMakerInferenceError" | jq
     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:ReceiveSageMakerInferenceSuccess" | jq
     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:sde-api-test-result" | jq
     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:StableDiffusionSnsUserTopic" | jq
     aws sns delete-topic --topic-arn "arn:aws:sns:$AWS_DEFAULT_REGION:$ACCOUNT_ID:successCreateModel" | jq
  fi

fi

if [ -f "/tmp/txt2img_sla_report.json" ]; then
  txt2img_sla_report=$(cat /tmp/txt2img_sla_report.json)

  sla_model_id=$(echo "$txt2img_sla_report" | jq -r '.model_id')
  sla_instance_type=$(echo "$txt2img_sla_report" | jq -r '.instance_type')
  sla_instance_count=$(echo "$txt2img_sla_report" | jq -r '.instance_count')
  sla_count=$(echo "$txt2img_sla_report" | jq -r '.count')
  sla_succeed=$(echo "$txt2img_sla_report" | jq -r '.succeed')
  sla_failed=$(echo "$txt2img_sla_report" | jq -r '.failed')
  sla_success_rate=$(echo "$txt2img_sla_report" | jq -r '.success_rate')
  sla_max_duration=$(echo "$txt2img_sla_report" | jq -r '.max_duration')
  sla_min_duration=$(echo "$txt2img_sla_report" | jq -r '.min_duration')
  sla_avg_duration=$(echo "$txt2img_sla_report" | jq -r '.avg_duration')

  properties+=("\\n[Inference SLA]")
  properties+=("model_id: ${sla_model_id}")
  properties+=("instance_type: ${sla_instance_type}")
  properties+=("instance_count: ${sla_instance_count}")
  properties+=("count: ${sla_count}")
  properties+=("succeed: ${sla_succeed}")
  properties+=("failed: ${sla_failed}")
  properties+=("success_rate: ${sla_success_rate}")
  properties+=("max_duration_seconds: ${sla_max_duration}")
  properties+=("min_duration_seconds: ${sla_min_duration}")
  properties+=("avg_duration_seconds: ${sla_avg_duration}")

  failed_list=$(echo "$txt2img_sla_report" | jq -r '.failed_list')
  properties+=("${failed_list}")
fi

properties+=("SNS_ARN: ${SNS_ARN}")

if [ -f "report-${CODEBUILD_BUILD_NUMBER}.html" ]; then
  report_file="report-${CODEBUILD_BUILD_NUMBER}.html"
  aws s3 cp "$report_file" "s3://$REPORT_BUCKET/test_report/"
  properties+=("Report: s3://$REPORT_BUCKET/test_report/$report_file")
fi

properties+=("CodeBuildUrl: ${CODEBUILD_BUILD_URL}")

message=""
for property in "${properties[@]}"; do
   message="${message}${property}\\n\\n"
done

echo -e "$message"
aws sns publish \
        --region "$SNS_REGION" \
        --topic-arn "$SNS_ARN" \
        --message-structure json \
        --subject "ESD $CODE_BRANCH $CASE_PASSED_RESULT - Deploy & API Test" \
        --message-attributes '{"key": {"DataType": "String", "StringValue": "value"}}' \
        --message "{\"default\": \"$message\"}"

if [ "$result" = "Passed" ]; then
  echo "----------------------------------------------------------------"
  echo "Delete log groups"
  echo "----------------------------------------------------------------"
  aws logs describe-log-groups | jq -r '.logGroups[].logGroupName' | grep -v codebuild | xargs -I {} aws logs delete-log-group --log-group-name {}
fi
