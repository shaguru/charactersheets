# MiniMax Video Maker - self-contained installer.
# Everything (uv, Python, venv, pip/uv caches, HF cache) lives inside this folder,
# except model weights, which go to the models_dir set in settings.json (default .\models).
param([switch]$Force)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$RT = Join-Path $Root "runtime"
New-Item -ItemType Directory -Force "$RT\bin", "$RT\cache", "$RT\python", "$Root\projects", "$Root\logs" | Out-Null

# Keep every cache off C:
$env:UV_CACHE_DIR = "$RT\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$RT\python"
$env:PIP_CACHE_DIR = "$RT\cache\pip"
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = "$RT\cache\tmp"; $env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

# 1. uv (single exe) into runtime\bin
$uv = "$RT\bin\uv.exe"
if (-not (Test-Path $uv)) {
  Write-Host "Downloading uv..."
  $zip = "$RT\cache\uv.zip"
  Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
  Expand-Archive $zip -DestinationPath "$RT\cache\uv_x" -Force
  Copy-Item "$RT\cache\uv_x\uv.exe" $uv
}

# 2. Python 3.12 + venv inside runtime\
if ($Force -and (Test-Path "$RT\.venv")) {
  $VenvTarget = [IO.Path]::GetFullPath((Join-Path $RT '.venv'))
  $RuntimeBase = [IO.Path]::GetFullPath($RT).TrimEnd('\') + '\'
  if (-not $VenvTarget.StartsWith($RuntimeBase, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid venv removal path' }
  Remove-Item -LiteralPath $VenvTarget -Recurse -Force
}
if (-not (Test-Path "$RT\.venv\Scripts\python.exe")) {
  & $uv python install 3.12
  if ($LASTEXITCODE) { throw 'Local Python download failed; rerun setup to resume.' }
  & $uv venv "$RT\.venv" --python 3.12 --managed-python
  if ($LASTEXITCODE) { throw 'Could not create the local MiniMax environment.' }
}
$py = "$RT\.venv\Scripts\python.exe"

# 3. Packages. Driver 572.x supports CUDA 12.8, so use cu128 PyTorch.
& $uv pip install --python $py -r "$Root\requirements-torch.txt" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw "torch install failed" }
& $uv pip install --python $py -r "$Root\requirements.txt" --constraint "$Root\requirements-torch.txt"
if ($LASTEXITCODE) { throw "package install failed" }
& $uv pip install --python $py -e "$Root"
if ($LASTEXITCODE) { throw "app install failed" }

# 4. Real GPU check
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.ones((256,256),device='cuda',dtype=torch.bfloat16); assert bool((x@x).isfinite().all()), 'CUDA BF16 calculation failed'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw 'CUDA verification failed. Update the NVIDIA driver and retry.' }
& $py -m mvm doctor
if ($LASTEXITCODE) { throw 'MiniMax GPU/runtime verification failed.' }
Write-Host "`nInstalled. Run mvm.bat --help (CLI) or UI.bat (local web page)."
