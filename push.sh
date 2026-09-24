#!/usr/bin/env bash
# Create the GitHub repo and push. Needs ONE of:
#   - gh, authenticated:            gh auth login
#   - a token in the environment:   export GH_TOKEN=github_pat_...
#     (fine-grained PAT with "Administration: read and write" to create the
#      repo, plus "Contents: read and write")
#
#   bash push.sh [repo-name] [private|public]
set -euo pipefail

NAME="${1:-bci-cbramod}"
VIS="${2:-private}"
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh repo create "$NAME" "--$VIS" --source=. --remote=origin --push
    gh repo view --web
    exit 0
fi

TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
[ -n "$TOKEN" ] || { echo "no gh auth and no GH_TOKEN — see the header of this file"; exit 1; }

USER_LOGIN=$(curl -sS -H "Authorization: Bearer $TOKEN" https://api.github.com/user \
             | python -c 'import json,sys;print(json.load(sys.stdin)["login"])')
echo "authenticated as $USER_LOGIN"

PRIVATE=true; [ "$VIS" = "public" ] && PRIVATE=false
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/repos \
     -d "{\"name\":\"$NAME\",\"private\":$PRIVATE,\"description\":\"CBraMod + NeuroTTT arms for Neural Interfaces 2026 Track 2 (BCI decoding)\"}" \
     | python -c 'import json,sys; d=json.load(sys.stdin); print("created", d["full_name"]) if "full_name" in d else sys.exit("create failed: "+json.dumps(d))'

git remote remove origin 2>/dev/null || true
git remote add origin "https://${USER_LOGIN}:${TOKEN}@github.com/${USER_LOGIN}/${NAME}.git"
git push -u origin HEAD
# Do not leave the token sitting in .git/config.
git remote set-url origin "https://github.com/${USER_LOGIN}/${NAME}.git"
echo "pushed -> https://github.com/${USER_LOGIN}/${NAME}"
