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
#   tb      serve TensorBoard on logs3\tb (run this in a second window)
#           use -TbDir if you trained with a different --tb-dir
#
# Examples:
#   .\scripts\run_train.ps1 -Mode smoke
#   .\scripts\run_train.ps1 -Mode smoke -WithWeights    # also load HistoPLUS (271 MB)
#   .\scripts\run_train.ps1 -Mode smoke -DataRoot "D:\data\toxpath_73"   # validate new data
#   .\scripts\run_train.ps1 -Mode fold -Fold 1 -Epochs 12
#   .\scripts\run_train.ps1 -Mode all -Epochs 12
#   .\scripts\run_train.ps1 -Mode resume -Fold 1 -Epochs 20
#   .\scripts\run_train.ps1 -Mode tb                    # then open http://<lab-ip>:6006
#   .\scripts\run_train.ps1 -Mode tb -TbDir "logs3_73\tb"   # curves written elsewhere
#   .\scripts\run_train.ps1 -Mode fold -Fold 1 -DataRoot "D:\data\toxpath"
#   .\scripts\run_train.ps1 -Mode fold -Foreground      # keep it attached (Ctrl+C works)
#   .\scripts\run_train.ps1 -Mode all -Epochs 12 -NoGradCkpt   # 48 GB cards only
#   .\scripts\run_train.ps1 -Mode fold -ExtraArgs "--samples-per-epoch","3500","--val-every","3"
#
# Curves go to TensorBoard (logs3\tb\fold{N}); the plain-text log under logs3\
# is still the authoritative record of what happened.
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
# unless the file has a UTF-8 BOM, and non-ASCII comments would be garbled.

[CmdletBinding()]
param(
    [ValidateSet("smoke", "fold", "all", "resume", "tb")]
    [string]$Mode = "fold",

    [int]$Fold = 1,

    [int]$Epochs = 12,

    # TensorBoard port for -Mode tb
    [int]$Port = 6006,

    # Event dir served by -Mode tb. Must match the --tb-dir you trained with,
    # otherwise the page loads but shows nothing. Relative paths are resolved
    # against the repo root (this script always Set-Location there).
    [string]$TbDir = "logs3\tb",

    # Empty = let paths.py resolve it (TOXPATH_DATA_ROOT / data\toxpath / ...)
    [string]$DataRoot = "",

    [string]$PythonExe = "python",

    [int]$NumWorkers = 4,

    # 24 GB cards (4090) need gradient checkpointing at 1024/batch 1; pass this
    # to turn it off on 48 GB cards where the recompute just costs time.
    [switch]$NoGradCkpt,

    # Anything else you want to pass straight through to train_3class.py, e.g.
    #   -ExtraArgs "--samples-per-epoch","3500","--val-every","3"
    # Appended last, so a flag repeated here wins over the wrapper's default.
    # Not applied to -Mode smoke (it needs its own fixed, tiny arg list).
    [string[]]$ExtraArgs = @(),

    # Smoke runs --init none by default (fast, no 271 MB weight load). Pass this
    # to load HistoPLUS too, which is the step most likely to break on a new box
    # (timm version / key mismatch). Costs ~30 s instead of ~50 s on CPU.
    [switch]$WithWeights,

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
# 1024 input + batch 1 peaks around 22 GB without this, which is right at the
# 4090 limit. smoke runs on CPU and builds its own arg list, so it is unaffected.
if (-not $NoGradCkpt) {
    $common += "--grad-checkpointing"
}

# Accept both spellings of a pass-through list:
#   -ExtraArgs "--samples-per-epoch","3500"        -> already an array
#   -ExtraArgs "--samples-per-epoch,3500"          -> one string with commas
# The second form is what survives a nested `powershell -File` call, and
# silently reaching argparse as a single token is very hard to spot.
if ($ExtraArgs.Count -eq 1 -and $ExtraArgs[0] -like '*,*') {
    $ExtraArgs = @($ExtraArgs[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
}

$trainArgs = @()
switch ($Mode) {
    "smoke" {
        $smokeInit = if ($WithWeights) { "encoder+neck" } else { "none" }
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
            "--init", $smokeInit,
            "--out-dir", "_smoke/ckpt",
            "--log-dir", "_smoke/log",
            "--tb-dir", "_smoke/tb"
        )
        # smoke has its own fixed arg list, so -DataRoot must be appended here
        # explicitly - otherwise validating a dataset that lives on another
        # drive silently reads whatever paths.py happens to resolve.
        if ($DataRoot -ne "") {
            $trainArgs += @("--data-root", ('"{0}"' -f $DataRoot))
        }
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
    "tb" {
        # Serves TensorBoard; handled below once the environment is cleaned up.
        $trainArgs = @()
    }
}

# Pass-through handling: drop wrapper defaults that -ExtraArgs overrides, THEN
# append ExtraArgs. Order matters - filtering after appending would delete the
# user's own arguments too, since they look identical in the combined list.
# -Mode smoke keeps its own fixed list and ignores -ExtraArgs on purpose.
if ($Mode -ne "smoke") {
    if ($ExtraArgs.Count -gt 0 -and $trainArgs.Count -gt 0) {
        $extraOpts = @($ExtraArgs | Where-Object { $_ -like '--*' })
        if ($extraOpts.Count -gt 0) {
            $kept = @()
            $i = 0
            while ($i -lt $trainArgs.Count) {
                $tok = $trainArgs[$i]
                if ($tok -like '--*' -and ($extraOpts -contains $tok)) {
                    $hasValue = ($i + 1 -lt $trainArgs.Count) -and ($trainArgs[$i + 1] -notlike '--*')
                    if ($hasValue) { $i += 2 } else { $i += 1 }
                }
                else {
                    $kept += $tok
                    $i += 1
                }
            }
            $trainArgs = $kept
        }
    }
    $trainArgs = $trainArgs + $ExtraArgs
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
# file encoding). TensorBoard event files are binary and unaffected.
[System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')

# Reduce allocator fragmentation so a transient 10-20 MB request does not fail
# at the very end of an epoch. Only helps fragmentation, not a real peak overrun.
if (-not [System.Environment]::GetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF', 'Process')) {
    [System.Environment]::SetEnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True', 'Process')
}

# -Mode tb: serve TensorBoard, detached so closing RDP does not kill it.
if ($Mode -eq "tb") {
    # find_spec only checks presence - importing tensorboard prints warnings
    # to stderr, which PowerShell 5.1 turns into NativeCommandError.
    & $PythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tensorboard') else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "tensorboard is not installed for $PythonExe - run: `"$PythonExe`" -m pip install tensorboard"
    }
    if (-not (Test-Path $TbDir)) {
        Write-Host ("[tb     ] {0} not found - no run has written curves there yet." -f $TbDir)
        New-Item -ItemType Directory -Path $TbDir -Force | Out-Null
    }
    $tbStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $tbOut = "logs3\tb_$tbStamp.out.log"
    $tbErr = "logs3\tb_$tbStamp.err.log"
    # tensorboard wants forward slashes on Windows too; harmless if already so
    $tbDirArg = $TbDir.Replace('\', '/')
    $tbProc = Start-Process -FilePath $PythonExe `
        -ArgumentList @("-m", "tensorboard.main", "--logdir", $tbDirArg,
                        "--port", "$Port", "--host", "0.0.0.0") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $tbOut `
        -RedirectStandardError $tbErr `
        -PassThru
    Write-Host ("[pid    ] {0}" -f $tbProc.Id)
    Write-Host ("[logdir ] {0}" -f $tbDirArg)
    Write-Host "[url    ] http://127.0.0.1:$Port/    LAN: http://<this-pc-ip>:$Port/"
    Write-Host ("[stdout ] {0}" -f $tbOut)
    Write-Host "[tips   ] page is live after ~10 s; training appends to it automatically"
    Write-Host "[tips   ] firewall     : netsh advfirewall firewall add rule name=TensorBoard dir=in action=allow protocol=TCP localport=$Port"
    Write-Host "[tips   ] stop serving : Stop-Process -Id $($tbProc.Id)"
    exit 0
}

if ($Foreground) {
    Write-Host ("[run    ] {0} {1}" -f $PythonExe, ($trainArgs -join " "))
    # PS 5.1 converts ANY native stderr output into a terminating error while
    # $ErrorActionPreference is 'Stop', and torch/timm emit FutureWarnings on
    # every start. Relax it for the child and trust the exit code instead.
    $ErrorActionPreference = 'Continue'
    # The child writes UTF-8 (PYTHONUTF8=1); without this the console decodes
    # it as GBK and the Chinese log lines turn into mojibake.
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
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
Write-Host "[tips   ] curves         : .\scripts\run_train.ps1 -Mode tb   ->  http://127.0.0.1:6006"
Write-Host "[tips   ] stop training  : Stop-Process -Id $($proc.Id)"
Write-Host ""
Write-Host "Detached from this window: closing RDP will NOT kill it."
