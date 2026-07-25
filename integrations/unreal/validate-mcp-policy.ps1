[CmdletBinding()]
param(
    [string]$Policy = (Join-Path $PSScriptRoot "mcp-policy.json")
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

& $Python.Source @Prefix $Tool "validate-mcp-policy" $Policy
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
