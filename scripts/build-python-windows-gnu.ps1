# Build the Python extension with rustup's GNU toolchain (no Visual Studio,
# no admin rights) and place it next to python/rustsolve/__init__.py, so that
# `PYTHONPATH=python` imports it without installing anything.
#
#   .\scripts\build-python-windows-gnu.ps1 -Python C:\path\to\env\python.exe
#
# Release wheels are for CI (maturin build); this script is for local work.
param(
    [Parameter(Mandatory = $true)][string]$Python
)
$ErrorActionPreference = "Continue"
$envRoot = Split-Path -Parent $Python
$repo = Split-Path -Parent $PSScriptRoot
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
# keep build output outside the source tree (which may sit in a synced folder)
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\rustsolve-target" }
# PyO3 links Python through raw-dylib by default, which on windows-gnu needs a
# full MinGW binutils; link the environment's python3.lib instead
$env:PYO3_USE_RAW_DYLIB = "0"
$env:PYO3_PYTHON = $Python
$env:RUSTFLAGS = "-L native=$envRoot\libs"
Push-Location $repo
try {
    cargo build --release -p rustsolve-py
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
Copy-Item "$env:CARGO_TARGET_DIR\release\_rustsolve.dll" "$repo\python\rustsolve\_rustsolve.pyd" -Force
Write-Output "built $repo\python\rustsolve\_rustsolve.pyd"
