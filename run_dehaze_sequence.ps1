$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

python dehaze_sequence.py --input ".\image seq" --output ".\results\video_dehaze" --model dehazeformer-t --device auto --fps 24 --max-side 1280 --comparison --preprocess video --postprocess video
