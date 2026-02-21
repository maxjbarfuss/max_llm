#!/bin/bash
# GitHub Actions - Poll run until completion
# Usage: ./scripts/ci/poll_run.sh [run_number] [interval_seconds]
# Example: ./scripts/ci/poll_run.sh 7 20
# Polls the latest run (or specified run_number) every interval_seconds until done

REPO="maxjbarfuss/max_llm"
RUN_NUMBER="${1}"  # If not specified, polls latest
INTERVAL="${2:-20}"  # Default 20 seconds
TOKEN_FILE=".github/.github_token"
MAX_POLLS=60  # Max 60 * 20s = 20 minutes

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Error: $TOKEN_FILE not found. Run 'source setup.sh' first."
    exit 1
fi

export GITHUB_TOKEN=$(cat "$TOKEN_FILE")

poll_count=0

while [ $poll_count -lt $MAX_POLLS ]; do
    if [ -z "$RUN_NUMBER" ]; then
        # Get latest run
        curl -s -H "Authorization: token $GITHUB_TOKEN" \
            "https://api.github.com/repos/$REPO/actions/runs?per_page=1" > /tmp/latest_run.json
        RUN_DATA=$(cat /tmp/latest_run.json | python3 -c "import json, sys; d=json.load(sys.stdin); print(json.dumps(d['workflow_runs'][0] if d.get('workflow_runs') else {}))")
        RUN_ID=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('id', ''))")
    else
        curl -s -H "Authorization: token $GITHUB_TOKEN" \
            "https://api.github.com/repos/$REPO/actions/runs" > /tmp/all_runs.json
        RUN_DATA=$(python3 << PYEOF
import json
with open('/tmp/all_runs.json') as f:
    d = json.load(f)
runs = [r for r in d.get('workflow_runs', []) if r['run_number'] == $RUN_NUMBER]
print(json.dumps(runs[0] if runs else {}))
PYEOF
)
        RUN_ID=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('id', ''))")
    fi

    if [ -z "$RUN_ID" ]; then
        echo "❌ Could not find run"
        exit 1
    fi

    STATUS=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('status', 'unknown'))")
    CONCLUSION=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('conclusion', '-'))")
    RUN_NUM=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('run_number', '?'))")
    BRANCH=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('head_branch', '?'))")
    TITLE=$(echo "$RUN_DATA" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('display_title', '')[:50])")

    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[$TIMESTAMP] Run #$RUN_NUM ($BRANCH): $STATUS [$CONCLUSION] - $TITLE"

    if [ "$STATUS" = "completed" ]; then
        echo ""
        if [ "$CONCLUSION" = "success" ]; then
            echo "✅ Run #$RUN_NUM completed successfully!"
        else
            echo "❌ Run #$RUN_NUM failed (conclusion: $CONCLUSION)"
            echo ""
            echo "Get job details: ./scripts/ci/get_jobs.sh $RUN_ID"
        fi
        exit 0
    fi

    poll_count=$((poll_count + 1))
    if [ $poll_count -lt $MAX_POLLS ]; then
        sleep "$INTERVAL"
    fi
done

echo ""
echo "⏱️  Timeout: Run did not complete after $((MAX_POLLS * INTERVAL)) seconds"
exit 1
