"""Make sure data/shortlist.json is fresh before the deep dive.

If the shortlist is older than MAX_AGE_H (GitHub sometimes skips scheduled workflow runs), trigger the
"Daily analysis" workflow through the GitHub API using the git credential already stored on this machine,
wait for it to finish, then `git pull`. Prints the outcome; never prints the token.

    python scripts/ensure_fresh.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
REPO = "rfmsant/football-model"
WORKFLOW = "weekly.yml"
MAX_AGE_H = 20


def shortlist_age_h() -> float | None:
    try:
        s = json.loads((ROOT / "data" / "shortlist.json").read_text(encoding="utf-8"))
        t = dt.datetime.fromisoformat(s["generated_at"].replace("Z", "+00:00"))
        return (dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return None


def token() -> str | None:
    p = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True,
                       text=True, env={**__import__("os").environ, "GCM_INTERACTIVE": "never"}, timeout=30)
    for line in p.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    return None


def main() -> int:
    subprocess.run(["git", "-C", str(ROOT), "pull", "--rebase", "-q"], check=False)
    age = shortlist_age_h()
    if age is not None and age <= MAX_AGE_H:
        print(f"shortlist is fresh ({age:.1f} h old)")
        return 0
    print(f"shortlist is stale ({'unknown' if age is None else f'{age:.1f} h'} old): triggering the Daily analysis workflow")
    tok = token()
    if not tok:
        print("no GitHub credential available; continuing with the old shortlist")
        return 1
    h = {"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"}
    api = f"https://api.github.com/repos/{REPO}/actions"
    started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
    r = requests.post(f"{api}/workflows/{WORKFLOW}/dispatches", headers=h, json={"ref": "main"}, timeout=30)
    if r.status_code != 204:
        print(f"could not trigger workflow (HTTP {r.status_code}); continuing with the old shortlist")
        return 1
    run_id = None
    for _ in range(90):  # up to ~30 minutes
        time.sleep(20)
        runs = requests.get(f"{api}/workflows/{WORKFLOW}/runs?per_page=5", headers=h, timeout=30).json().get("workflow_runs", [])
        run = next((x for x in runs if dt.datetime.fromisoformat(x["created_at"].replace("Z", "+00:00")) >= started), None)
        if run:
            run_id = run["id"]
            if run["status"] == "completed":
                print(f"workflow run {run_id} finished: {run['conclusion']}")
                break
    else:
        print(f"workflow run {run_id} still running after 30 min; continuing")
    subprocess.run(["git", "-C", str(ROOT), "pull", "--rebase", "-q"], check=False)
    age = shortlist_age_h()
    print(f"shortlist now {age:.1f} h old" if age is not None else "shortlist unreadable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
