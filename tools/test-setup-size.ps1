# Regression check for fresh installs, where the 28 GB checkpoint is absent.
# Extract the actual expression so the test exercises the shipping installer.
$ErrorActionPreference = 'Stop'
$Source = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'setup-direct.ps1') -Raw
$Calculation = [regex]::Match($Source, '(?m)^\s*\$MissingBytes \+= .+$').Value
if (-not $Calculation) { throw 'Missing disk-space calculation in setup-direct.ps1' }
foreach ($Case in @(
    @{ Size = 28431843583L; Have = 0L; Expected = 28431843583L },
    @{ Size = 28431843583L; Have = 5000000000L; Expected = 23431843583L },
    @{ Size = 28431843583L; Have = 28431843583L; Expected = 0L },
    @{ Size = 28431843583L; Have = 28431843584L; Expected = 0L },
    @{ Size = 7179L; Have = 0L; Expected = 7179L }
)) {
    # JSON parsing is how model sizes reach the real installer.
    $file = ('{"size":' + $Case.Size + '}') | ConvertFrom-Json
    $Have = $Case.Have
    $MissingBytes = 0L
    # Use the current scope, as setup does.
    . ([scriptblock]::Create($Calculation))
    if ($MissingBytes -ne $Case.Expected) { throw "Wrong remaining size: $MissingBytes; expected $($Case.Expected)" }
}
Write-Host 'PASS: missing/partial/complete/oversized 28 GB downloads and small files.'
