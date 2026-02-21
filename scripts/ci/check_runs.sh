#!/bin/bash
# GitHub Actions - List recent runs with status
# Usage: ./scripts/ci/check_runs.sh [count]
# Example: ./scripts/ci/check_runs.sh 5

set -e

REPO="maxjbarfuss/max_llm"
COUNT="${1:-10}"
TOKEN_FILE=".github/.github_token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Error: $TOKEN_FILE not found. Run 'source setup.sh' first."
    exit 1
fi

export GITHUB_TOKEN=$(cat "$TOKEN_FILE")

echo ""
echo "=== GitHub Workflow Runs (Latest $COUNT) ==="
echo ""

curl -s -H "Authorization: token $GITHUB_TOKEN" \
    "https://api.github.com/repos/$REPO/actions/runs?per_page=$COUNT" | python3 << 'EOF'
import json, sys
from datetime import datetime

d = json.load(sys.stdin)
print(f"{'Run':<5} {'Status':<12} {'Conclusion':<10} {'Branch':<15} {'Created':<16} {'Title':<50}")
print("-" * 130)

for r in d.get('workflow_runs', []):
    run_num = r['run_number']
    status = r['status']
    conclusion = r.get('conclusion') or '-'
    branch = r.get('head_branch', '?')
    created = r.get('created_at', '')[:10]
    title = r['display_title'][:48]
    
    print(f"{run_num:<5} {status:<12} {conclusion:<10} {branch:<15} {created:<16} {title:<50}")

print(f"\nTotal: {d.get('total_count', 0)} runs")
EOF

echo ""
