@echo off
setlocal
cd /d "%~dp0"
python dehaze_sequence.py --input ".\image seq" --output ".\results\video_dehaze" --model dehazeformer-t --device auto --fps 24 --max-side 1280 --comparison --preprocess video --postprocess video
