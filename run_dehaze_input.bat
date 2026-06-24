@echo off
setlocal
cd /d "%~dp0"
if not exist input mkdir input
if not exist results mkdir results
python infer.py --input .\input --output .\results --model dehazeformer-s --device auto
