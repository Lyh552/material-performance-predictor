# 项目依赖使用 Codex 的 Python 3.12 安装，必须配套解释器运行。
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = "$PSScriptRoot\.deps;$PSScriptRoot;$env:PYTHONPATH"
$python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $python = $command.Source }
}
if (-not (Test-Path -LiteralPath $python)) {
    throw '未找到 Python，请安装项目依赖后使用 python 命令运行。'
}
Push-Location $PSScriptRoot
try {
    & $python @args
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
