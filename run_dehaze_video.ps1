$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

python dehaze_video.py --video "Cycling in Foggy weather. Riding To Jaam Gate. #cycling #nature #greenery #fog #mountains #clouds.mp4" --output-root ".\results\video_inputs_enhanced" --model dehazeformer-t --device auto --max-side 1280 --comparison --preprocess video --postprocess video
