$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

New-Item -ItemType Directory -Force -Path .\input | Out-Null
New-Item -ItemType Directory -Force -Path .\results | Out-Null

python infer.py --input .\input --output .\results --model dehazeformer-s --device auto
