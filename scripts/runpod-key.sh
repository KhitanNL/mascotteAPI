#!/usr/bin/env bash
# Saves the RunPod API key to .runpod-api-key (chmod 600, gitignored) without showing it.
cd "$(dirname "$0")/.."
read -rsp "Paste your RunPod API key ONCE and press Enter: " k; echo
k=$(printf %s "$k" | tr -d '[:space:]')
# A key pasted twice in a row: keep the first copy.
rest=${k#rpa_}; [ "$rest" != "${rest%%rpa_*}" ] && k="rpa_${rest%%rpa_*}"
case "$k" in rpa_*) ;; *) echo "That is not a RunPod key (it should start with rpa_). Copy the key again and rerun."; exit 1;; esac
umask 077; printf %s "$k" > .runpod-api-key
code=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -H "user-agent: mascotte-ops/1" -H "authorization: Bearer $k" https://api.runpod.io/v2/pods)
[ "$code" = 200 ] && echo "Saved and accepted by RunPod. Tell Claude it is done." || echo "Saved, but RunPod answered $code. Tell Claude."
