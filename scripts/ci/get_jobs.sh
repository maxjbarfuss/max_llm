#!/bin/bash
# GitHub Actions - Get job details for a specific run
# Usage: ./scripts/ci/get_jobs.sh <run_id>
# Example: ./scripts/ci/get_jobs.sh 22248261206

set -e

REPO="maxjbarfuss/max_llm"
RUN_ID="${1}"
TOKEN_FILE=".github/.github_token"

if [ -z "$RUN_ID" ]; then
    echo "❌ Usage: $0 <run_id>"
    echo "   Example: $0 22248261206"
    exit 1
fi

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Error: $TOKEN_FILE not found. Run 'source setup.sh' first."
    exit 1
fi

export GITHUB_TOKEN=$(cat "$TOKEN_FILE")

echo ""
echo "=== Jobs for Run #$RUN_ID ==="
echo ""

curl -s -H "Authorization: token $GITHUB_TOKEN" \
    "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/jobs" > /tmp/jobs_data.json

python3 << 'EOF'
import json

with open('/tmp/jobs_data.json') as f:
    d = json.load(f)

print(f"{'Job Name':<50} {'Status':<12} {'Conclusion':<12}")
print("-" * 80)

for job in d.get('jobs', []):
    name = job['name'][:48]
    status = job['status']
    conclusion = job.get('conclusion') or '-'
    print(f"{name:<50} {status:<12} {conclusion:<12}")

print(f"\nJob URLs:")
for job in d.get('jobs', []):
    url = job['html_url']
    print(f"  {job['name']}: {url}")
EOF

echo ""
