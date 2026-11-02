#!/usr/bin/env bash
# face.mascotte.ai -> 127.0.0.1:7100 through a Cloudflare Tunnel of its own.
# Leaves the other tunnels on this machine (and their cert.pem) as they are.
set -euo pipefail
CF=$HOME/.cloudflared
CERT=$CF/cert-mascotte.ai.pem
NAME=mascotte-face
HOST=face.mascotte.ai
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$CERT" ]; then
  [ -f "$CF/cert.pem" ] && cp -p "$CF/cert.pem" "$CF/cert.pem.before-mascotte"
  rm -f "$CF/cert.pem"
  echo "==> Open the link below and choose mascotte.ai"
  cloudflared tunnel login
  mv "$CF/cert.pem" "$CERT"
  [ -f "$CF/cert.pem.before-mascotte" ] && mv "$CF/cert.pem.before-mascotte" "$CF/cert.pem"
fi

C="cloudflared --origincert $CERT"
if ! $C tunnel list 2>/dev/null | grep -q " $NAME "; then
  $C tunnel create "$NAME"
fi
ID=$($C tunnel list 2>/dev/null | awk -v n="$NAME" '$2==n {print $1}')
[ -n "$ID" ] || { echo "tunnel $NAME not found"; exit 1; }
$C tunnel route dns "$ID" "$HOST" || true

sed "s/TUNNEL_ID/$ID/g" "$ROOT/deploy/face-tunnel.yml" | sudo tee /etc/cloudflared/mascotte.yml >/dev/null
sudo tee /etc/systemd/system/cloudflared-mascotte.service >/dev/null <<UNIT
[Unit]
Description=Cloudflare Tunnel for face.mascotte.ai (mascotteAPI worker)
After=network-online.target mascotte-worker.service
Wants=network-online.target

[Service]
User=khitan
ExecStart=/usr/bin/cloudflared --no-autoupdate --config /etc/cloudflared/mascotte.yml tunnel run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-mascotte
echo "==> tunnel $NAME ($ID) for $HOST is up"
