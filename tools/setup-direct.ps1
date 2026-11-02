# Windows PowerShell 5.1 compatible. No administrator rights or global Python needed.
param(
    [string]$Models = 'qwen-rapid-aio',
    [switch]$EngineOnly,
    [switch]$Yes
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root 'runtime'
$Data = Join-Path $Root 'data'
$Venv = Join-Path $Runtime '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Stamp = Join-Path $Runtime '.requirements.sha256'
$Lock = $null
$Transcript = $false

function Invoke-Checked([string]$Program, [string[]]$Arguments, [string]$Failure) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw $Failure }
}

function Get-Sha256([string]$Path) {
    $Stream = [IO.File]::OpenRead($Path)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($Hasher.ComputeHash($Stream)).Replace('-', '') }
    finally { $Stream.Dispose(); $Hasher.Dispose() }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'Character Forge requires Windows 10/11 on an x64 PC with an NVIDIA GPU.'
    }
    foreach ($dir in @('logs', 'cache', 'input\forge_refs', 'output', 'models', 'user', 'temp')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Data $dir) | Out-Null
    }
    try { $Lock = [IO.File]::Open((Join-Path $Runtime '.install.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { throw 'Another Character Forge installer is running in this folder. Wait for it to finish.' }
    Start-Transcript -Path (Join-Path $Data 'logs\setup.log') -Append | Out-Null; $Transcript = $true
    Write-Host "`nCharacter Forge - preparing $Root" -ForegroundColor Cyan

    $Smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $SmiPath = if ($Smi) { $Smi.Source } else { $null }
    if (-not $SmiPath) {
        $SmiPath = Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
        if (-not (Test-Path -LiteralPath $SmiPath)) { $SmiPath = $null }
    }
    if (-not $SmiPath) { throw 'No NVIDIA driver found. Install the latest driver from https://www.nvidia.com/Download/index.aspx and rerun START.bat.' }
    $GpuRows = @(& $SmiPath --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or -not $GpuRows.Count) { throw 'NVIDIA GPU detection failed. Update the NVIDIA driver and restart Windows.' }
    $Gpu = $GpuRows[0].Split(',') | ForEach-Object { $_.Trim() }
    if ([version]$Gpu[1] -lt [version]'570.65') { throw "Driver $($Gpu[1]) is too old for the CUDA 12.8 runtime. Install NVIDIA driver 570.65 or newer, restart Windows, and rerun START.bat." }
    Write-Host "[1/4] NVIDIA: $($Gpu[0]), driver $($Gpu[1]), $($Gpu[2]) MiB VRAM" -ForegroundColor Green
    $Capabilities = @(& $SmiPath --query-gpu=compute_cap --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -eq 0 -and $Capabilities.Count -and [double]::Parse($Capabilities[0], [Globalization.CultureInfo]::InvariantCulture) -lt 8.0) {
        throw 'Qwen requires native BF16 support. Use an NVIDIA Ampere or newer GPU (RTX 30/40/50 series).'
    }
    if ([int]$Gpu[2] -le 12288) {
        Write-Host 'Low-VRAM GPU: the app will use CPU offload and smaller render sizes automatically.' -ForegroundColor Yellow
    }
    $Ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    if ($Ram -lt 60) { Write-Warning ('Qwen uses CPU offload. 64 GB system RAM is recommended; detected {0:N0} GB.' -f $Ram) }

    $Catalog = Get-Content -LiteralPath (Join-Path $Runtime 'models.json') -Raw | ConvertFrom-Json
    $Ids = @($Models.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $MissingBytes = 0L
    foreach ($id in $Ids) {
        $Model = $Catalog.models | Where-Object { $_.id -eq $id }
        if (-not $Model) { throw "Unknown model id: $id" }
        if (-not $EngineOnly) {
            foreach ($file in $Model.files) {
                $Dest = Join-Path (Join-Path $Data 'models') ($file.dir + '/' + $file.name)
                if ((Test-Path -LiteralPath $Dest) -and (Get-Item -LiteralPath $Dest).Length -eq $file.size) { continue }
                $Have = if (Test-Path -LiteralPath ($Dest + '.part')) { (Get-Item -LiteralPath ($Dest + '.part')).Length } else { 0 }
                # Both arguments must be Int64: PowerShell 5.1 otherwise picks
                # Math.Max(Int32, Int32) for a missing 28 GB checkpoint.
                $MissingBytes += [math]::Max([long]0, ([long]$file.size - [long]$Have))
            }
        }
    }
    $RuntimeReserve = if (Test-Path -LiteralPath (Join-Path $Venv 'Lib\site-packages\torch')) { 2GB } else { 12GB }
    $Free = (Get-Item -LiteralPath $Root).PSDrive.Free
    if ($Free -lt ($MissingBytes + $RuntimeReserve)) { throw ('Not enough disk space. Need at least {0:N1} GB free here; have {1:N1} GB. A new Qwen install needs about 45 GB free.' -f (($MissingBytes + $RuntimeReserve) / 1e9), ($Free / 1e9)) }

    $env:PYTHONNOUSERSITE = '1'; $env:PIP_REQUIRE_VIRTUALENV = 'true'
    $env:PIP_CACHE_DIR = Join-Path $Data 'cache\pip'; $env:HF_HOME = Join-Path $Data 'cache\huggingface'; $env:TORCH_HOME = Join-Path $Data 'cache\torch'
    $env:HF_HUB_OFFLINE = '0'; $env:TRANSFORMERS_OFFLINE = '0'
    $Bundled = Join-Path $Runtime 'python\python.exe'
    if (-not (Test-Path -LiteralPath $Bundled)) {
        $Bundled = Join-Path $Runtime 'python312\tools\python.exe'
        if (-not (Test-Path -LiteralPath (Join-Path $Runtime 'python312\.complete'))) {
            Write-Host '[2/4] Downloading local Python 3.12.10 (no system installation)...' -ForegroundColor Cyan
            $Archive = Join-Path $Data 'cache\python.3.12.10.nupkg'; $ExpectedHash = '0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8'
            if (-not (Test-Path -LiteralPath $Archive) -or (Get-Sha256 $Archive) -ne $ExpectedHash) {
                $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
                if (-not $Curl) { throw 'curl.exe is required. Use an up-to-date Windows 10 or Windows 11 installation.' }
                Invoke-Checked $Curl.Source @('--location', '--fail', '--retry', '5', '--retry-delay', '3', '--output', $Archive, 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg') 'Python download failed. Rerun START.bat to retry.'
                if ((Get-Sha256 $Archive) -ne $ExpectedHash) { throw 'Python archive checksum failed. Rerun START.bat to download it again.' }
            }
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $PythonDir = Join-Path $Runtime 'python312'
            New-Item -ItemType Directory -Force -Path $PythonDir | Out-Null
            $Zip = [IO.Compression.ZipFile]::OpenRead($Archive)
            try {
                foreach ($Entry in $Zip.Entries) {
                    $Target = [IO.Path]::GetFullPath((Join-Path $PythonDir $Entry.FullName))
                    if (-not $Target.StartsWith($PythonDir + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid path in Python archive.' }
                    if ($Entry.Name) {
                        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Target)) | Out-Null
                        [IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Target, $true)
                    }
                }
            } finally { $Zip.Dispose() }
            Invoke-Checked $Bundled @('--version') 'The downloaded Python interpreter could not start.'
            '3.12.10' | Set-Content -LiteralPath (Join-Path $PythonDir '.complete') -Encoding ASCII
        }
    }
    $Version = & $Bundled -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0) { throw 'The local Python interpreter could not start. Install the Microsoft Visual C++ x64 Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe and retry.' }
    if (-not (Test-Path -LiteralPath $Python)) {
        Invoke-Checked $Bundled @('-m', 'venv', $Venv) 'Could not create the local Python environment.'
    }
    [IO.File]::WriteAllLines((Join-Path $Venv 'pyvenv.cfg'),
        @("home = $(Split-Path -Parent $Bundled)", 'include-system-site-packages = false', "version = $Version"),
        (New-Object Text.UTF8Encoding($false)))
    & $Python -m pip --version
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked $Python @('-m', 'ensurepip', '--upgrade') 'Could not initialize pip in the local environment.'
    }

    Write-Host '[3/4] Checking/installing the pinned CUDA environment...' -ForegroundColor Cyan
    $Requirements = Join-Path $Runtime 'requirements.txt'
    $Hash = Get-Sha256 $Requirements
    $InstalledHash = if (Test-Path -LiteralPath $Stamp) { (Get-Content -LiteralPath $Stamp -Raw).Trim() } else { '' }
    if ($InstalledHash -ne $Hash) {
        Invoke-Checked $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip') 'Could not update local pip.'
        Invoke-Checked $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', $Requirements) 'Dependency installation failed. Rerun START.bat to resume.'
    } else {
        & $Python (Join-Path $Runtime 'check_runtime.py') --imports-only
        if ($LASTEXITCODE -ne 0) {
            Invoke-Checked $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '--force-reinstall', '-r', $Requirements) 'Could not repair the dependencies. Rerun START.bat.'
        }
    }
    Invoke-Checked $Python @((Join-Path $Runtime 'check_runtime.py')) 'CUDA compatibility check failed. See the diagnostic above; update the NVIDIA driver if needed.'
    $Hash | Set-Content -LiteralPath $Stamp -Encoding ASCII
    if (-not $EngineOnly) {
        Write-Host '[4/4] Downloading/checking Qwen and selected models (resumable)...' -ForegroundColor Cyan
        Invoke-Checked $Python @((Join-Path $Runtime 'downloader.py'), '--models-dir', (Join-Path $Data 'models'), '--ids', ($Ids -join ',')) 'A model download failed. Rerun START.bat to resume.'
        Invoke-Checked $Python @((Join-Path $Runtime 'prepare_models.py'), '--if-installed') 'Optional SDXL configuration setup failed. Rerun START.bat.'
    }
    Write-Host 'Character Forge is ready.' -ForegroundColor Green
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    if ($Lock) { $Lock.Dispose() }
}
