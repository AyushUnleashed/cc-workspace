#!/usr/bin/env bash
# Usage: ./clip.sh <youtube-url-or-local-video> <clips.json> [out_dir]
# Downloads (if given a URL), then renders every clip in clips.json as a 9:16 speaker-following mp4.
set -euo pipefail
src="$1"; clips="$2"; out="${3:-work/out}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/models" "$here/work"
[ -f "$here/models/face_detection_yunet_2023mar.onnx" ] || curl -fsSL -o "$here/models/face_detection_yunet_2023mar.onnx" \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
if [[ "$src" == http* ]]; then
  yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 -o "$here/work/source.mp4" "$src"
  src="$here/work/source.mp4"
fi
python3 "$here/reframe.py" "$src" "$clips" "$out"
