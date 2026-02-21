# CI Monitoring Scripts

Helper scripts for monitoring GitHub Actions CI runs and debugging failures.

All scripts use `GITHUB_TOKEN` from `.github/.github_token` (set via `source setup.sh`).

## Scripts

### `check_runs.sh` - List Recent Runs
List all workflow runs with status and conclusion.

```bash
./scripts/ci/check_runs.sh [count]          # Default: 10 runs
./scripts/ci/check_runs.sh 5                # Show last 5 runs
```

**Output columns:**
- `Run`: Run number
- `Status`: queued | in_progress | completed
- `Conclusion`: success | failure | cancelled | N/A
- `Branch`: Branch name (main, test-ci, etc)
- `Created`: Date run was created
- `Title`: Commit message/PR title

---

### `get_jobs.sh` - Get Job Details
Show all jobs for a specific run, including individual status and direct URLs.

```bash
./scripts/ci/get_jobs.sh <run_id>
./scripts/ci/get_jobs.sh 22248261206       # Shows all jobs + their URLs
```

**Output:**
- Job name, status, and conclusion
- Direct GitHub URLs for each job (click to view logs)

---

### `poll_run.sh` - Poll Until Complete
Continuously poll a run until it completes, with real-time status updates.

```bash
./scripts/ci/poll_run.sh                   # Poll latest run every 20s (max 20 min)
./scripts/ci/poll_run.sh 7                 # Poll run #7
./scripts/ci/poll_run.sh 7 10              # Poll run #7 every 10s
```

**Output:**
Shows status updates every 20 seconds (or custom interval). On completion:
- ✅ Success → outputs "completed successfully"
- ❌ Failure → outputs failure conclusion and suggests `get_jobs.sh` to debug

---

## Quick Workflows

### Monitor Latest Push
```bash
./scripts/ci/poll_run.sh              # Waits for latest run to complete
```

### Debug a Failed Run
```bash
./scripts/ci/check_runs.sh 3          # Find run number
./scripts/ci/get_jobs.sh <run_id>     # Get job details + URLs
# Click URL to view logs
```

### Watch test-ci Branch
After pushing to test-ci:
```bash
./scripts/ci/poll_run.sh              # Polls automatically
# When done, inspect results with get_jobs.sh
```

---

## Working Commands (Tested Feb 20, 2026)

These commands are what power the scripts above. Keep them here for reference.

### Check All Runs
```bash
export GITHUB_TOKEN=$(cat .github/.github_token)
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/maxjbarfuss/max_llm/actions/runs?per_page=10" | \
  python3 -m json.tool | head -100
```

### Get Latest Run Status
```bash
export GITHUB_TOKEN=$(cat .github/.github_token)
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/maxjbarfuss/max_llm/actions/runs?per_page=1" | \
  python3 -c "import json, sys; d=json.load(sys.stdin); r=d['workflow_runs'][0]; \
  print(f\"Run #{r['run_number']}: {r['status']} ({r.get('conclusion', 'N/A')})\")"
```

### Get Jobs for a Run
```bash
export GITHUB_TOKEN=$(cat .github/.github_token)
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/maxjbarfuss/max_llm/actions/runs/<RUN_ID>/jobs" | \
  python3 -m json.tool
```

### View Job Logs
```bash
# Find job ID from get_jobs.sh, then:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/maxjbarfuss/max_llm/actions/jobs/<JOB_ID>/logs" > logs.txt
```

---

## Setup

**First time only:**
1. Create GitHub personal access token at https://github.com/settings/tokens (minimal scopes: `repo`, `workflow`)
2. Save to `.github/.github_token` (ignored by git)
3. Run `source setup.sh` to load token into `$GITHUB_TOKEN`

**Every session:**
```bash
source setup.sh  # Loads GITHUB_TOKEN from .github/.github_token
```

---

## Notes

- All scripts are idempotent—safe to run multiple times
- Default polling interval is 20 seconds (max wait ~20 minutes)
- Logs are streamed directly; no caching
- `.github/.github_token` is in `.gitignore` and `.git/info/exclude` for safety
