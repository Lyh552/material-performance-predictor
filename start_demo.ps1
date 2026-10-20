# 只启动本目录合成演示，不使用正式库。
param([int]$Port=8515)
$ErrorActionPreference='Stop'
$env:MATERIAL_STORE=Join-Path $PSScriptRoot 'local_store'
& "$PSScriptRoot/run_python.ps1" -m streamlit run app/public_demo.py --global.developmentMode false --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
exit $LASTEXITCODE
