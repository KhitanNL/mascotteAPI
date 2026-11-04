#!/usr/bin/env bash
# Side by side: left.mp4 labelled "Lite", right.mp4 labelled "Pro", sound from the left.
#   tools/compare_film.sh amara_en_lite.mp4 amara_en_pro.mp4 amara_en_compare.mp4
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 lite.mp4 pro.mp4 out.mp4" >&2; exit 1; }
label='drawtext=text=%s:x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8'
ffmpeg -y -loglevel error -i "$1" -i "$2" -filter_complex \
  "[0:v]$(printf "$label" Lite)[a];[1:v]$(printf "$label" Pro)[b];[a][b]hstack=inputs=2:shortest=1[v]" \
  -map "[v]" -map 0:a -c:v libx264 -preset medium -crf 18 -c:a copy -movflags +faststart "$3"
echo "wrote $3"
