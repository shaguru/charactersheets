param([string]$RuntimeDirectory = '', [switch]$ImageOnly)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root 'runtime\minimax_app'
$Settings = Join-Path $Root 'data\user\forge_settings.json'
$Config = @{}
if (Test-Path -LiteralPath $Settings) {
    $ReadConfig = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json
    foreach ($Property in $ReadConfig.PSObject.Properties) { $Config[$Property.Name] = $Property.Value }
}
if (-not $RuntimeDirectory) { $RuntimeDirectory = $env:MINIMAX_DIR }
if (-not $RuntimeDirectory -and $Config.minimax_dir) { $RuntimeDirectory = $Config.minimax_dir }
if (-not $RuntimeDirectory) {
    $Sibling = Join-Path (Split-Path -Parent $Root) 'MINIMAX_VIDEO_Maker'
    if ((Test-Path -LiteralPath (Join-Path $Sibling 'runtime\.venv\Scripts\python.exe')) -and
        (Test-Path -LiteralPath (Join-Path $Sibling 'mvm\forge_bridge.py'))) { $RuntimeDirectory = $Sibling }
    else { $RuntimeDirectory = Join-Path $Root 'runtime\minimax_app' }
}
$RuntimeDirectory = [IO.Path]::GetFullPath($RuntimeDirectory)
$Py = Join-Path $RuntimeDirectory 'runtime\.venv\Scripts\python.exe'
try {
    $Gpu = @(& nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
    if ($LASTEXITCODE -or -not $Gpu.Count) { throw 'MiniMax requires a working NVIDIA driver.' }
    $Parts = $Gpu[0].Split(',') | ForEach-Object { $_.Trim() }
    if ([version]$Parts[2] -lt [version]'570.65') { throw 'Update to NVIDIA driver 570.65 or newer for CUDA 12.8.' }
    if ([int]$Parts[1] -lt 20000) { Write-Host 'This MiniMax video setup was tested with 24 GB VRAM / 64 GB RAM; this smaller GPU may not fit video.' -ForegroundColor Yellow }
    $env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
    foreach ($Key in @('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH','VIRTUAL_ENV')) {
        Remove-Item -LiteralPath "Env:$Key" -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $Py)) {
        $Drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($RuntimeDirectory))
        $Need = if ($ImageOnly) { 48GB } else { 130GB }
        if ($Drive.AvailableFreeSpace -lt $Need) { throw "MiniMax setup needs approximately $([int]($Need/1GB)) GiB free including environment and model headroom. Use -RuntimeDirectory on a larger drive or clear space." }
        if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDirectory 'tools\install.ps1'))) {
            New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null
            foreach ($Item in @('mvm','tools','requirements.txt','requirements-torch.txt','pyproject.toml')) {
                Copy-Item -LiteralPath (Join-Path $Source $Item) -Destination (Join-Path $RuntimeDirectory $Item) -Recurse -Force
            }
        }
    }
    # Rerun package setup for the portable worker even if a failed first attempt
    # created python.exe. Existing standalone MiniMax installations are reused.
    $OwnRequirements = Join-Path $RuntimeDirectory 'requirements.txt'
    $PortableWorker = (Test-Path -LiteralPath $OwnRequirements) -and
        ((Get-FileHash -LiteralPath $OwnRequirements).Hash -eq (Get-FileHash -LiteralPath (Join-Path $Source 'requirements.txt')).Hash)
    if (-not (Test-Path -LiteralPath $Py) -or $PortableWorker) {
        & (Join-Path $RuntimeDirectory 'tools\install.ps1')
        if ($LASTEXITCODE) { throw 'MiniMax environment installation failed; rerun SETUP_MINIMAX.bat to resume.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RuntimeDirectory 'mvm\forge_bridge.py'))) {
        throw 'This MiniMax folder needs the Character Forge bridge code. Update it or choose a new -RuntimeDirectory for the portable worker.'
    }
    Push-Location -LiteralPath $RuntimeDirectory
    try {
        $Installed = (& $Py -m mvm forge-bridge | Select-Object -Last 1) | ConvertFrom-Json
        if ($LASTEXITCODE) { throw 'Could not check MiniMax installation.' }
        if (-not $ImageOnly -and -not $Installed.models.video) { & $Py -u -m mvm download video; if ($LASTEXITCODE) { throw 'Video download stopped; rerun SETUP_MINIMAX.bat to resume.' } }
        if (-not $Installed.models.image) {
            & $Py -u -m mvm download image
            if ($LASTEXITCODE) { throw 'Image download stopped; rerun SETUP_MINIMAX.bat to resume.' }
        }
    } finally { Pop-Location }
    $Config.minimax_dir = $RuntimeDirectory
    New-Item -ItemType Directory -Path (Split-Path -Parent $Settings) -Force | Out-Null
    $Config | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Settings -Encoding UTF8
    Write-Host "MiniMax ready in $RuntimeDirectory. Open START.bat and select MiniMax in the original Forge interface." -ForegroundColor Green
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
