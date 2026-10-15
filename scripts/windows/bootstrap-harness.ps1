[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [string]$ScriptExtenderRoot,
    [string]$Modsettings,
    [string]$Python = "python",
    [string]$VenvPath,
    [switch]$InstallTestDependencies,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (-not $VenvPath) { $VenvPath = Join-Path $repoRoot ".venv-harness" }
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$venvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $Python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($InstallTestDependencies) {
    & $venvPython -m pip install -r (Join-Path $repoRoot "requirements-harness-test.txt")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$doctorArgs = @("-m", "integration.windows_preflight", "--game-dir", $GameDir)
if ($ScriptExtenderRoot) { $doctorArgs += @("--script-extender-root", $ScriptExtenderRoot) }
if ($Modsettings) { $doctorArgs += @("--modsettings", $Modsettings) }
if ($Json) { $doctorArgs += "--json" }

Push-Location $repoRoot
try {
    & $venvPython @doctorArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
