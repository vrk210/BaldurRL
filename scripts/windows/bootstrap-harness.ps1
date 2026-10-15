<#
.SYNOPSIS
Create an isolated harness venv and inspect a Windows BG3 installation read-only.
.DESCRIPTION
Does not install BG3/Script Extender, change settings/load orders, or launch BG3.
GameDir accepts the installation root or bin. ScriptExtenderRoot and Modsettings
default from LOCALAPPDATA in the Python preflight. -Json emits its v1 report;
present_unverified is static evidence only. The preflight exit code is preserved.
.PARAMETER InstallTestDependencies
Opt in to installing the pinned pytest/Lupa packages from requirements-harness-test.txt.
Omit this switch for standard-library-only runtime tools.
.EXAMPLE
.\scripts\windows\bootstrap-harness.ps1 -GameDir 'D:\Games\Baldurs Gate 3' -Json
#>
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
