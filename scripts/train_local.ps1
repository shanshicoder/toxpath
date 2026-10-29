<#
.SYNOPSIS
    ToxPath 本地训练一键启动（Windows + RTX 4090）。

.DESCRIPTION
    把 train_3class.py 的长参数收敛成 4 种模式，避免每次手敲一长串还敲错：

    smoke    冒烟：1 epoch、32 训练样本 / 8 验证样本，约 5 分钟，验证显存与管线
    fold     单折正式训练（默认 fold1、20 epoch）
    all      5 折顺序训练（按切片留一交叉验证）
    resume   从 last.pt 续训（默认 fold1）

    默认以后台方式启动：日志重定向到 logs3\ 下的文件，PowerShell 关掉、
    远程桌面断开、向日葵断连都不会中断训练。想前台看输出加 -Foreground。

.PARAMETER Mode
    smoke / fold / all / resume，默认 smoke。

.PARAMETER Fold
    折号 1~5，对应验证切片 train_001~train_005。

.PARAMETER Epochs
    训练轮数，默认 20。smoke 模式固定 1。

.PARAMETER DataRoot
    数据根目录。留空则交给 paths.py 自动解析（命令行 > 环境变量 > 项目内）。

.PARAMETER Ckpt
    HistoPLUS 权重。留空则自动解析（默认找 <项目根>\weights\）。

.PARAMETER Foreground
    前台运行，输出直接打到当前控制台。适合冒烟时盯日志；正式训练不要用，
    远程桌面一断窗口关了可能连带进程被杀。

.PARAMETER DryRun
    只打印最终拼出来的训练命令，不检查 torch、不启动训练。核对参数用。

.EXAMPLE
    .\scripts\train_local.ps1 -Mode smoke
    先冒烟，确认显存和 loss 正常。

.EXAMPLE
    .\scripts\train_local.ps1 -Mode fold -Fold 1 -Epochs 20
    单折正式训练，20 epoch。

.EXAMPLE
    .\scripts\train_local.ps1 -Mode all -Epochs 20
    5 折全跑，约 8~10 小时，建议下班前挂上。

.EXAMPLE
    .\scripts\train_local.ps1 -Mode resume -Fold 1
    断电/重启后从 checkpoints3\fold1\last.pt 接着跑。
#>

[CmdletBinding()]
param(
    [ValidateSet('smoke', 'fold', 'all', 'resume')]
    [string]$Mode = 'smoke',

    [ValidateRange(1, 5)]
    [int]$Fold = 1,

    [int]$Epochs = 20,

    [int]$BatchSize = 1,

    [int]$GradAccum = 8,

    [int]$SamplesPerEpoch = 800,

    [int]$ValEvery = 2,

    [int]$NumWorkers = 4,

    [int]$SaveEvery = 5,

    [string]$DataRoot = '',

    [string]$Ckpt = '',

    [string]$OutDir = '',

    [string]$LogDir = '',

    [switch]$FreezeEncoder,

    [switch]$NoCheckpointing,

    [switch]$Foreground,

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# 控制台统一 UTF-8，避免中文提示在 cmd / PowerShell 里乱码
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }

# --------------------------------------------------------------------------- #
# 0. 基本环境
# --------------------------------------------------------------------------- #
$Root = Split-Path -Parent $PSScriptRoot          # 项目根目录
Set-Location $Root

if (-not (Test-Path (Join-Path $Root 'train_3class.py'))) {
    throw "在 $Root 下找不到 train_3class.py，脚本位置不对？"
}

# python 来自当前激活的 conda 环境
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) {
    throw '找不到 python，先 conda activate 你的环境'
}

Write-Host "[env ] python  : $Python" -ForegroundColor Cyan
Write-Host "[env ] project : $Root" -ForegroundColor Cyan

$prevEap = $ErrorActionPreference
if ($DryRun) {
    Write-Host '[env ] DryRun：只打印将要执行的命令，不检查 torch、不启动训练' -ForegroundColor Yellow
    Write-Host ''
}
else {
    # 快速确认 torch 在、且是 CUDA 版（完整体检请跑 check_env.py）
    # 注意：PowerShell 5.1 在 ErrorActionPreference=Stop 下，原生命令往 stderr 写
    # 一句话就会抛 NativeCommandError，所以探针期间临时放宽为 Continue 再收回。
    $ErrorActionPreference = 'Continue'
    $probeOut = (& $Python -c "import torch,sys; sys.stdout.write(str(torch.__version__)+'|'+str(torch.cuda.is_available()))" 2>&1 | Out-String)
    $probeCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap

    if ($probeCode -ne 0) {
        Write-Host "[fail] 当前 python 里 import torch 失败（退出码 $probeCode）：" -ForegroundColor Red
        Write-Host ($probeOut.Trim()) -ForegroundColor Red
        Write-Host '       先激活装好 torch 的环境，或跑 python scripts\check_env.py 看缺什么' -ForegroundColor Yellow
        exit 1
    }

    $lastLine = ($probeOut.Trim() -split "`r?`n" | Select-Object -Last 1)
    if ($lastLine -notmatch '\|') {
        Write-Host '[fail] torch 探针输出异常，期望 "版本|cuda 布尔值"，实际：' -ForegroundColor Red
        Write-Host $probeOut.Trim() -ForegroundColor Red
        exit 1
    }
    $parts = $lastLine -split '\|'
    Write-Host "[env ] torch   : $($parts[0])  cuda=$($parts[1])" -ForegroundColor Cyan
    if ($parts[1] -ne 'True') {
        Write-Host '[fail] torch 看不到 CUDA，训练会退化成 CPU（慢 50 倍），已中止' -ForegroundColor Red
        Write-Host '       跑 python scripts\check_env.py 查原因' -ForegroundColor Yellow
        exit 1
    }
}

# --------------------------------------------------------------------------- #
# 1. 目录与日志
# --------------------------------------------------------------------------- #
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $Root 'checkpoints3' }
if ([string]::IsNullOrWhiteSpace($LogDir)) { $LogDir = Join-Path $Root 'logs3' }

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
switch ($Mode) {
    'smoke'  { $ExpName = "smoke_$Stamp" }
    'fold'   { $ExpName = "fold${Fold}_$Stamp" }
    'all'    { $ExpName = "all5_$Stamp" }
    'resume' { $ExpName = "fold${Fold}_resume_$Stamp" }
}

$RunLogDir = Join-Path $LogDir '_runs'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --------------------------------------------------------------------------- #
# 2. 拼参数
# --------------------------------------------------------------------------- #
$tokens = New-Object System.Collections.Generic.List[string]
$tokens.Add('train_3class.py')

function Add-Arg([string]$Name, [string]$Value) {
    $tokens.Add($Name)
    $tokens.Add($Value)
}
function Add-ArgQ([string]$Name, [string]$Value) {
    # 路径可能含空格，加引号交给子进程自己解析
    $tokens.Add($Name)
    $tokens.Add('"' + $Value + '"')
}

switch ($Mode) {
    'smoke' {
        Add-Arg '--fold' '1'
        Add-Arg '--epochs' '1'
        Add-Arg '--batch-size' "$BatchSize"
        Add-Arg '--grad-accum' '4'
        Add-Arg '--max-train-samples' '32'
        Add-Arg '--max-val-samples' '8'
        Add-Arg '--num-workers' '2'
        $OutDir = Join-Path $Root 'checkpoints3_smoke'
        $LogDir = Join-Path $Root 'logs3_smoke'
        New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    }
    'fold' {
        Add-Arg '--fold' "$Fold"
        Add-Arg '--epochs' "$Epochs"
        Add-Arg '--batch-size' "$BatchSize"
        Add-Arg '--grad-accum' "$GradAccum"
        Add-Arg '--samples-per-epoch' "$SamplesPerEpoch"
        Add-Arg '--val-every' "$ValEvery"
        Add-Arg '--num-workers' "$NumWorkers"
        Add-Arg '--save-every' "$SaveEvery"
    }
    'all' {
        $tokens.Add('--all-folds')
        Add-Arg '--epochs' "$Epochs"
        Add-Arg '--batch-size' "$BatchSize"
        Add-Arg '--grad-accum' "$GradAccum"
        Add-Arg '--samples-per-epoch' "$SamplesPerEpoch"
        Add-Arg '--val-every' "$ValEvery"
        Add-Arg '--num-workers' "$NumWorkers"
        Add-Arg '--save-every' "$SaveEvery"
    }
    'resume' {
        $resumePath = Join-Path $OutDir "fold$Fold\last.pt"
        if (-not (Test-Path $resumePath)) {
            Write-Host "[fail] 找不到 $resumePath，没有可续训的 checkpoint" -ForegroundColor Red
            Write-Host "       先跑：.\scripts\train_local.ps1 -Mode fold -Fold $Fold" -ForegroundColor Yellow
            exit 2
        }
        Add-Arg '--fold' "$Fold"
        Add-Arg '--epochs' "$Epochs"
        Add-Arg '--batch-size' "$BatchSize"
        Add-Arg '--grad-accum' "$GradAccum"
        Add-Arg '--samples-per-epoch' "$SamplesPerEpoch"
        Add-Arg '--val-every' "$ValEvery"
        Add-Arg '--num-workers' "$NumWorkers"
        Add-Arg '--save-every' "$SaveEvery"
        Add-ArgQ '--resume' $resumePath
    }
}

$tokens.Add('--amp')
if (-not $NoCheckpointing) { $tokens.Add('--grad-checkpointing') }
if ($FreezeEncoder) { $tokens.Add('--freeze-encoder') }

Add-ArgQ '--out-dir' $OutDir
Add-ArgQ '--log-dir' $LogDir
if (-not [string]::IsNullOrWhiteSpace($DataRoot)) { Add-ArgQ '--data-root' $DataRoot }
if (-not [string]::IsNullOrWhiteSpace($Ckpt)) { Add-ArgQ '--histoplus-ckpt' $Ckpt }

$argLine = ($tokens -join ' ')

# smoke 模式会改写 $OutDir / $LogDir，所以日志路径等参数定稿后再算
$RunLogDir = Join-Path $LogDir '_runs'
New-Item -ItemType Directory -Force -Path $RunLogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $RunLogDir "$ExpName.log"
$ErrFile = Join-Path $RunLogDir "$ExpName.err.log"

# --------------------------------------------------------------------------- #
# 3. 启动
# --------------------------------------------------------------------------- #
Write-Host ''
Write-Host '=== 启动训练 ===' -ForegroundColor Green
Write-Host "[mode ] $Mode"
Write-Host "[cmd  ] python $argLine"
Write-Host "[out  ] $OutDir"
Write-Host "[log  ] $LogFile"
Write-Host ''

if ($DryRun) {
    Write-Host '[dry  ] 只打印命令，未启动。确认无误后去掉 -DryRun 重跑' -ForegroundColor Yellow
    exit 0
}

if ($Foreground) {
    Write-Host '[run  ] 前台运行，Ctrl+C 可中断' -ForegroundColor Yellow
    # 训练过程会往 stderr 写 warning，同样需要临时放宽 ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python @tokens 2>&1 | Tee-Object -FilePath $LogFile
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    exit $code
}

# 后台：独立进程 + 日志重定向，父窗口关掉也不影响
$proc = Start-Process -FilePath $Python `
    -ArgumentList $argLine `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrFile `
    -WindowStyle Hidden `
    -PassThru

Write-Host "[run  ] 已在后台启动，PID = $($proc.Id)" -ForegroundColor Green
Write-Host ''
Write-Host '--- 常用命令 ---' -ForegroundColor Cyan
Write-Host "  # 实时看日志（Ctrl+C 只退出 tail，不影响训练）"
Write-Host "  Get-Content '$LogFile' -Wait -Tail 30"
Write-Host ''
Write-Host "  # 看还在不在跑"
Write-Host "  Get-Process -Id $($proc.Id) -ErrorAction SilentlyContinue"
Write-Host ''
Write-Host "  # 看 GPU 占用（每 2 秒刷新，Ctrl+C 退出）"
Write-Host '  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 2'
Write-Host ''
Write-Host "  # 要停就停（会丢掉当前 epoch，但有 last.pt 可续训）"
Write-Host "  Stop-Process -Id $($proc.Id)"
Write-Host ''
if ($Mode -eq 'all') {
    $hoursLo = [math]::Round($Epochs * 0.1 * 5, 1)
    Write-Host "[预计] 5 折 × $Epochs epoch，约 $hoursLo 小时（4090，800 样本/epoch）" -ForegroundColor Yellow
}
Write-Host '[提示] 远程桌面/向日葵断开不影响该进程；但电脑不能睡眠，见 docs\本地训练SOP.md 第 2 节' -ForegroundColor Yellow
