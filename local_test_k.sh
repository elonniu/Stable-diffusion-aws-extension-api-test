set -e

if [ "$(uname)" == "Darwin" ]; then
    source venv/bin/activate
else
    . venv/bin/activate
fi

if [ -z "$1" ]; then
    echo "Please input k parameter"
    exit 1
fi

pytest ./src/stable_diffusion_aws_extension_api_test/test_1_local -k "$1" --exitfirst -rA --log-cli-level=ERROR --json-report --json-report-summary --json-report-file=detailed_report.json --html=report.html --self-contained-html --continue-on-collection-errors
