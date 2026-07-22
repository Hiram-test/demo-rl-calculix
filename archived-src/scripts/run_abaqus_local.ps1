param(
    [ValidateSet("preflight", "solve", "train")]
    [string]$Mode = "solve",
    [string]$PythonCmd = "python",
    [string]$AbaqusCmd = $(if ($env:ABAQUS_CMD) { $env:ABAQUS_CMD } else { "abaqus" }),
    [string]$TemplateCae = "DEMO.cae",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
try {
    & $PythonCmd "rl_main_local.py" `
        --backend abaqus `
        --mode $Mode `
        --abaqus-cmd $AbaqusCmd `
        --template-cae-file $TemplateCae `
        --goal-file "examples/goal_local.json" `
        @RemainingArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
