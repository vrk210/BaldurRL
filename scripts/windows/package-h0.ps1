[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DivinePath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$source = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "bg3\h0"))
$divine = [System.IO.Path]::GetFullPath($DivinePath)
$output = [System.IO.Path]::GetFullPath($OutputPath)
$sourcePrefix = $source.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

if (-not (Test-Path -LiteralPath $divine -PathType Leaf)) { throw "Divine executable does not exist: $divine" }
if (-not (Test-Path -LiteralPath (Join-Path $source "Mods") -PathType Container)) { throw "Package source must contain Mods at its root: $source" }
if ($output.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Output must not be beneath package source: $output" }
if ([System.IO.Path]::GetExtension($output) -ne ".pak") { throw "OutputPath must end in .pak: $output" }
if ((Test-Path -LiteralPath $output) -and -not $Force -and -not $DryRun) { throw "Output exists; pass -Force to replace it: $output" }

$sourceFiles = Get-ChildItem -LiteralPath $source -Recurse -File | Sort-Object FullName
$sourceManifest = @($sourceFiles | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($sourcePrefix.Length).Replace("\", "/")
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes = $_.Length
    }
})

if ($DryRun) {
    [ordered]@{
        dry_run = $true
        argv = @($divine, "--game", "bg3", "--action", "create-package", "--source", $source, "--destination", $output)
        source = $source
        output = $output
        source_files = $sourceManifest
    } | ConvertTo-Json -Depth 5
    return
}

$outputDirectory = [System.IO.Path]::GetDirectoryName($output)
if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
$temporary = Join-Path $outputDirectory ("." + [System.IO.Path]::GetFileNameWithoutExtension($output) + "." + [guid]::NewGuid().ToString("N") + ".tmp.pak")
try {
    & $divine --game bg3 --action create-package --source $source --destination $temporary
    if ($LASTEXITCODE -ne 0) { throw "Divine exited with code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $temporary -PathType Leaf) -or (Get-Item -LiteralPath $temporary).Length -eq 0) {
        throw "Divine did not create a nonempty package: $temporary"
    }
    Move-Item -LiteralPath $temporary -Destination $output -Force:$Force
    $manifestPath = $output + ".manifest.json"
    [ordered]@{
        schema_version = 1
        source = $source
        source_files = $sourceManifest
        output = [ordered]@{
            path = $output
            sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $output).Length
        }
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Host "Created $output"
    Write-Host "Manifest $manifestPath"
}
finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
}
