#Requires -Version 5.1
# Compatibility entry point; uv and pyproject.toml own installation now.
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
}
& uv sync --project $PSScriptRoot --managed-python
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Ready. Launch with: uv run --project `"$PSScriptRoot`" handmouse"
