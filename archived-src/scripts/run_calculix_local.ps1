param(
    [ValidateSet("preflight", "solve", "train")]
    [string]$Mode = "solve",
    [string]$PythonCmd = "python",
    [string]$GmshCmd = $(if ($env:GMSH_CMD) { $env:GMSH_CMD } else { "gmsh" }),
    [string]$CcxCmd = $(if ($env:CCX_CMD) { $env:CCX_CMD } else { "ccx" }),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
try {
    & $PythonCmd "rl_main_local.py" `
        --backend calculix `
        --mode $Mode `
        --gmsh-cmd $GmshCmd `
        --ccx-cmd $CcxCmd `
        --plate-config "examples/calculix_plate.json" `
        --goal-file "examples/goal_local.json" `
        @RemainingArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
