[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath
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

& $Python.Source @Prefix $Tool "verify-bundle" $BundlePath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
