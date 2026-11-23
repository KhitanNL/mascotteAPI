#!/usr/bin/env bash
# Switch the cloud GPU behind face.mascotte.ai on or off. No restart needed: the
# gateway reads .cloud-workers on every new call.
#
#   scripts/cloud.sh add NAME URL      asks for that worker's token (not echoed), switched on
#   scripts/cloud.sh on|off NAME       new calls go there / stop going there (open calls keep working)
#   scripts/cloud.sh status            what the office gateway sees
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
F="$ROOT/.cloud-workers"
touch "$F"; chmod 600 "$F"

case "${1:-status}" in
  add)
    name="${2:?name}"; url="${3:?url}"
    read -rsp "WORKER_TOKEN of $name: " token; echo
    grep -vE "^#?${name}\|" "$F" > "$F.tmp" || true
    echo "${name}|${url%/}|${token}" >> "$F.tmp"; mv "$F.tmp" "$F"; chmod 600 "$F"
    echo "$name added and switched on" ;;
  on)
    sed -i -E "s/^#(${2:?name}\|)/\1/" "$F"; echo "${2} on" ;;
  off)
    sed -i -E "s/^(${2:?name}\|)/#\1/" "$F"; echo "${2} off (open calls there keep working)" ;;
  status)
    set -a; . "$ROOT/.env"; set +a
    curl -s -H "Authorization: Bearer $WORKER_TOKEN" http://127.0.0.1:7100/v1/status \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print("office:", d["slots"]); [print("cloud:", c) for c in d.get("cloud", [])] or print("cloud: none listed")' ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
