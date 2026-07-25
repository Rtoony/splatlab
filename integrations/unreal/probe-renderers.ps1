[CmdletBinding()]
param(
    [string]$ProjectRoot = (Join-Path $PSScriptRoot "SplatLabUE56"),
    [string[]]$EngineRoot = @()
)

$ErrorActionPreference = "Stop"
$Tool = Join-Path $PSScriptRoot "bundle_tool.py"
$Python = Get-Command py -ErrorAction SilentlyContinue
$Prefix = @()
if ($null -ne $Python) {
    $Prefix = @("-3")
} else {
    $Python = Get-Command python -ErrorAction Stop
}

$Arguments = @($Tool, "probe-renderers", "--project-root", $ProjectRoot)
foreach ($Root in $EngineRoot) {
    $Arguments += @("--engine-root", $Root)
}

& $Python.Source @Prefix @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
