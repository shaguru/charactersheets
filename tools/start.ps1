param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $env:FORGE_PORT) { $env:FORGE_PORT = '8288' }
$Url = "http://127.0.0.1:$($env:FORGE_PORT)/forge/"
try {
    $Existing = $null
    try { $Existing = Invoke-RestMethod -Uri ($Url + 'api/config') -TimeoutSec 3 } catch {}
    if ($Existing) {
        if ($Existing.backend -ne 'direct-diffusers' -or $Existing.output_dir -ne (Join-Path $Root 'data\output')) {
            throw "Another application is using port $($env:FORGE_PORT). Close it, or set FORGE_PORT to another port."
        }
        if (-not $NoBrowser) { Start-Process $Url }
        Write-Host "Character Forge is already running: $Url"
        exit 0
    }
    $Client = New-Object Net.Sockets.TcpClient
    try {
        $Attempt = $Client.BeginConnect('127.0.0.1', [int]$env:FORGE_PORT, $null, $null)
        if ($Attempt.AsyncWaitHandle.WaitOne(500) -and $Client.Connected) {
            throw "Port $($env:FORGE_PORT) is in use. Close that application, or set FORGE_PORT to another port."
        }
    } finally { $Client.Close() }
    & (Join-Path $PSScriptRoot 'setup-direct.ps1') -Yes
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:FORGE_OPEN_BROWSER = if ($NoBrowser) { '0' } else { '1' }
    Write-Host "`nCharacter Forge - $Url" -ForegroundColor Cyan
    Write-Host 'Keep this window open while you work. Close it or press Ctrl+C to stop.'
    & (Join-Path $Root 'runtime\.venv\Scripts\python.exe') -u (Join-Path $Root 'runtime\direct_server.py')
    exit $LASTEXITCODE
} catch {
    Write-Host "`n$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
