#!/usr/bin/env bash
# Commit and push whatever the last fetch step changed under data/. Usage: commit-data.sh <dataset>
set -euo pipefail
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A data
if git diff --cached --quiet; then
  echo "No changes for $1."
  exit 0
fi
git commit -m "data: $1 ($(date -u +%Y-%m-%d))"
for attempt in 1 2 3; do
  if git pull --rebase && git push; then
    echo "::notice title=commit::$1 committed"
    exit 0
  fi
  sleep $((attempt * 5))
done
echo "::error title=commit::could not push $1"
exit 1
