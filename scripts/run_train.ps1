# ToxPath training launcher for the lab workstation (Windows + RTX 4090).
#
# Why a wrapper instead of a bare python command:
#   - training runs for hours, so it must survive closing the RDP window
#     (Start-Process detaches it from the console);
#   - stdout/stderr go to timestamped files under logs3\ so nothing is lost;
#   - the four common invocations are easy to get wrong by hand.
#
# Modes:
#   smoke   one epoch, 8 samples, 224 px, CPU  -> verifies data/loss/ckpt in ~1 min
#   fold    train a single fold
#   all     train all folds (leave-one-slide-out)
#   resume  continue a fold from checkpoints3\foldN\last.pt
#
# Examples:
#   .\scripts\run_train.ps1 -Mode smoke
#   .\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12
#   .\scripts\run_train.ps1 -Mode all -Epochs 12
#   .\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 20
#   .\scripts\run_train.ps1 -Mode fold -Fold 1 -DataRoot "D:\data\toxpath"
#   .\scripts\run_train.ps1 -Mode fold -Foreground      # keep it attached (Ctrl+C works)
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
# unless the file has a UTF-8 BOM, and non-ASCII comments would be garbled.

[CmdletBinding()]
param(
    [ValidateSet("smoke", "fold", "all", "resume")]
    [string]$Mode = "fold",

    [int]$Fold = 1,

    [int]$Epochs = 12,

    # Empty = let paths.py resolve it (TOXPATH_DATA_ROOT / data\toxpath / ...)
    [string]$DataRoot = "",

    [string]$PythonExe = "python",

    [int]$NumWorkers = 4,

    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "logs3")) {
    New-Item -ItemType Directory -Path "logs3" | Out-Null
}
if (-not (Test-Path "train_3class.py")) {
    throw "train_3class.py not found in $Root - run this script from the repo."
}

$common = @()
if ($DataRoot -ne "") {
    # Quote it: the data path may contain spaces (e.g. "D:\my data\toxpath")
    $common += @("--data-root", ('"{0}"' -f $DataRoot))
}

$trainArgs = @()
switch ($Mode) {
    "smoke" {
        $trainArgs = @(
            "train_3class.py",
            "--fold", "1",
            "--epochs", "1",
            "--image-size", "224",
            "--max-train-samples", "8",
            "--max-val-samples", "4",
            "--samples-per-epoch", "8",
            "--log-every", "1",
            "--num-workers", "0",
            "--device", "cpu",
            "--init", "none",
            "--out-dir", "_smoke/ckpt",
            "--log-dir", "_smoke/log"
        )
    }
    "fold" {
        $trainArgs = @(
            "train_3class.py",
            "--fold", "$Fold",
            "--epochs", "$Epochs",
            "--num-workers", "$NumWorkers"
        ) + $common
    }
    "all" {
        $trainArgs = @(
            "train_3class.py",
            "--all-folds",
            "--epochs", "$Epochs",
            "--num-workers", "$NumWorkers"
        ) + $common
    }
    "resume" {
        $last = "checkpoints3\fold$Fold\last.pt"
        if (-not (Test-Path $last)) {
            throw "resume target not found: $last (train that fold first)"
        }
        $trainArgs = @(
            "train_3class.py",
            "--fold", "$Fold",
            "--epochs", "$Epochs",
            "--resume", $last,
            "--num-workers", "$NumWorkers"
        ) + $common
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = "logs3\train_$Mode`_$stamp.out.log"
$errLog = "logs3\train_$Mode`_$stamp.err.log"

# Some machines export the same variable twice in different letter cases
# (e.g. HTTPS_PROXY + https_proxy, after mixing cmd and git-bash profiles).
# .NET's process-env dictionary is case-insensitive, so Start-Process throws
# "An item with the same key has already been added" when it copies them.
# Fix: keep exactly ONE variable per case-insensitive name (prefer the
# ALL-CAPS spelling), drop the others. Never touch unique names - deleting
# e.g. SystemRoot would break python's random init.
$byUpper = @{}
foreach ($k in [System.Environment]::GetEnvironmentVariables('Process').Keys) {
    $u = $k.ToUpperInvariant()
    if (-not $byUpper.ContainsKey($u)) { $byUpper[$u] = @() }
    $byUpper[$u] += $k
}
$dropped = @()
foreach ($u in $byUpper.Keys) {
    if ($byUpper[$u].Count -le 1) { continue }
    $keep = $byUpper[$u] | Where-Object { $_ -ceq $u } | Select-Object -First 1
    if (-not $keep) { $keep = $byUpper[$u][0] }
    foreach ($k in $byUpper[$u]) {
        if ($k -cne $keep) {
            [System.Environment]::SetEnvironmentVariable($k, $null, 'Process')
            $dropped += $k
        }
    }
}
if ($dropped.Count) {
    Write-Host ("[env    ] dropped {0} duplicated env vars (case twins): {1}" -f $dropped.Count, ($dropped -join ", "))
}

# Force the child's stdout/stderr to UTF-8. Without this, redirected logs
# contain GBK-mangled Chinese text (the console codepage differs from the
# file encoding). live.jsonl is already written as UTF-8 by the trainer.
[System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')

if ($Foreground) {
    Write-Host ("[run    ] {0} {1}" -f $PythonExe, ($trainArgs -join " "))
    & $PythonExe @trainArgs
    exit $LASTEXITCODE
}

Write-Host ("[launch ] mode={0} fold={1} epochs={2}" -f $Mode, $Fold, $Epochs)
$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList $trainArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru

Write-Host ("[pid    ] {0}" -f $proc.Id)
Write-Host ("[stdout ] {0}" -f $outLog)
Write-Host ("[stderr ] {0}" -f $errLog)
Write-Host ""
Write-Host "[tips   ] live tail      : Get-Content $outLog -Wait -Tail 20"
Write-Host "[tips   ] still running  : Get-Process -Id $($proc.Id)"
Write-Host "[tips   ] monitor page   : python monitor.py   ->  http://127.0.0.1:8787"
Write-Host "[tips   ] stop training  : Stop-Process -Id $($proc.Id)"
Write-Host ""
Write-Host "Detached from this window: closing RDP will NOT kill it."
