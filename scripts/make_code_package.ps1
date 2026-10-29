<#
.SYNOPSIS
    打包代码（可选带 HistoPLUS 权重），用于拷到实验室电脑。

.DESCRIPTION
    实验室电脑上一般已经有数据（train_patches / train_masks_classes），但**权重
    往往没有**（271 MB，单独一个 .pt，最容易漏）。所以这里给两种包：

    1) 纯代码包（默认，约 250 KB）
       包含：model\ dataset\ scripts\ tests\ docs\ + 根目录 4 个 .py
             + requirements.txt + README.md
       适合：机器上已经有权重，只是同步代码改动

    2) 代码 + 权重包（加 -IncludeWeights，约 271 MB）
       在 1) 的基础上，把 histoplus_cellvit_segmentor_20x.pt 放进包内
       weights\ 目录。解压后 paths.py 会自动找到它，不用配任何环境变量。
       适合：第一次上机，或换了一台没有权重的机器

    两种包都排除：data\ checkpoints3\ logs3\ eval3\ _deprecated\ .idea\ .git\
    __pycache__ *.png *.log

    产出默认放在项目上一级目录，文件名带时间戳，反复打包不会互相覆盖。
    脚本会打印每个文件的大小、总数、权重的 SHA256，打包完还会重新扫一遍
    压缩包，确认除了 weights\ 下的权重之外没有混进数据文件。

.PARAMETER OutFile
    输出 zip 路径。默认 ..\toxpath_code_<时间戳>.zip（带权重时为 toxpath_full_<时间戳>.zip）。

.PARAMETER IncludeWeights
    把 HistoPLUS 权重一起打进去。约 271 MB，压缩后体积基本不变（.pt 本身已紧凑），
    Compress-Archive 会花一两分钟。

.PARAMETER WeightsPath
    显式指定权重文件路径。留空则按顺序找：
    <项目根>\weights\<文件名> → <项目根>\<文件名> → <项目根>\..\<文件名>

.PARAMETER IncludeGit
    额外把 .git 一起打进去（想在新机器上继续用 git 时才需要，体积会大很多）。

.EXAMPLE
    .\scripts\make_code_package.ps1
    纯代码包，约 250 KB。

.EXAMPLE
    .\scripts\make_code_package.ps1 -IncludeWeights
    代码 + 权重，约 271 MB。第一次拷到实验室机器时用这个。

.EXAMPLE
    .\scripts\make_code_package.ps1 -IncludeWeights -OutFile E:\toxpath_full.zip
    带权重并直接输出到 U 盘根目录。
#>

[CmdletBinding()]
param(
    [string]$OutFile = '',

    [switch]$IncludeWeights,

    [string]$WeightsPath = '',

    [switch]$IncludeGit
)

$ErrorActionPreference = 'Stop'

# 控制台统一 UTF-8，避免中文提示在 cmd / PowerShell 里乱码
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }

# 关掉进度条：PS 5.1 下进度条会让 Compress-Archive 慢好几倍，输出也更干净
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path (Join-Path $Root 'train_3class.py'))) {
    throw "在 $Root 下找不到 train_3class.py，脚本位置不对？"
}

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $prefix = if ($IncludeWeights) { 'toxpath_full' } else { 'toxpath_code' }
    $OutFile = Join-Path (Split-Path -Parent $Root) "${prefix}_$Stamp.zip"
}
$OutFile = [IO.Path]::GetFullPath($OutFile)
$OutDir = Split-Path -Parent $OutFile
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

# --------------------------------------------------------------------------- #
# 0. 权重定位（带权重时先找，找不到立刻退出，别等打完包才发现）
# --------------------------------------------------------------------------- #
$WeightName = 'histoplus_cellvit_segmentor_20x.pt'
$WeightFile = $null
if ($IncludeWeights) {
    $cands = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($WeightsPath)) { $cands.Add($WeightsPath) }
    $cands.Add((Join-Path $Root "weights\$WeightName"))
    $cands.Add((Join-Path $Root $WeightName))
    $cands.Add((Join-Path (Split-Path -Parent $Root) $WeightName))

    foreach ($c in $cands) {
        if (Test-Path -LiteralPath $c -PathType Leaf) {
            $WeightFile = (Get-Item -LiteralPath $c).FullName
            break
        }
    }

    if (-not $WeightFile) {
        Write-Host '[fail] 指定了 -IncludeWeights，但找不到权重文件。已尝试：' -ForegroundColor Red
        foreach ($c in $cands) { Write-Host "       $c" -ForegroundColor Red }
        Write-Host "       用 -WeightsPath 指定完整路径，或把 $WeightName 放到 <项目根>\weights\ 下" -ForegroundColor Yellow
        exit 3
    }

    $wMB = [math]::Round((Get-Item -LiteralPath $WeightFile).Length / 1MB, 1)
    Write-Host "[weight] $WeightFile  ($wMB MB)" -ForegroundColor Cyan
    if ($wMB -lt 250) {
        Write-Host '[warn] 权重不足 250 MB，正常约 271 MB，可能没拷完整' -ForegroundColor Yellow
    }
}

# --------------------------------------------------------------------------- #
# 1. 白名单：目录 + 根目录文件
# --------------------------------------------------------------------------- #
$IncludeDirs = @('model', 'dataset', 'scripts', 'tests', 'docs')
$IncludeFiles = @(
    'train_3class.py',
    'evaluate.py',
    'inference.py',
    'paths.py',
    'requirements.txt',
    'README.md'
)
if ($IncludeGit) { $IncludeDirs += '.git' }

# 任何一层路径命中下面的模式都跳过
$SkipDirNames = @('__pycache__', '.idea', '.vscode', '.pytest_cache',
                  'data', 'checkpoints3', 'checkpoints3_smoke', 'logs3',
                  'logs3_smoke', 'eval3', 'runs3', '_deprecated')
$SkipSuffix = @('.pyc', '.pt', '.pth', '.log', '.zip', '.png', '.jpg', '.jpeg', '.tar', '.gz')

$stage = Join-Path $env:TEMP "toxpath_code_stage_$Stamp"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Test-Skip([IO.FileInfo]$f) {
    if ($SkipSuffix -contains $f.Extension.ToLower()) { return $true }
    $parts = $f.FullName.Substring($Root.Length).Split([IO.Path]::DirectorySeparatorChar)
    foreach ($p in $parts) {
        if ($SkipDirNames -contains $p) { return $true }
    }
    return $false
}

# --------------------------------------------------------------------------- #
# 2. 复制到暂存目录
# --------------------------------------------------------------------------- #
$copied = New-Object System.Collections.Generic.List[object]
$skipped = 0

foreach ($item in $IncludeFiles) {
    $src = Join-Path $Root $item
    if (Test-Path $src) {
        $fi = Get-Item $src
        if (-not (Test-Skip $fi)) {
            Copy-Item -Path $src -Destination (Join-Path $stage $item) -Force
            $copied.Add([pscustomobject]@{ Path = $item; KB = [math]::Round($fi.Length / 1KB, 1) })
        }
    }
    else {
        Write-Host "[warn] 缺文件：$item" -ForegroundColor Yellow
    }
}

foreach ($dir in $IncludeDirs) {
    $srcDir = Join-Path $Root $dir
    if (-not (Test-Path $srcDir)) {
        Write-Host "[warn] 缺目录：$dir" -ForegroundColor Yellow
        continue
    }
    $files = Get-ChildItem -Path $srcDir -Recurse -File -Force
    foreach ($f in $files) {
        if (Test-Skip $f) { $skipped++; continue }
        $rel = $f.FullName.Substring($Root.Length + 1)
        $dst = Join-Path $stage $rel
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
        Copy-Item -Path $f.FullName -Destination $dst -Force
        $copied.Add([pscustomobject]@{ Path = $rel; KB = [math]::Round($f.Length / 1KB, 1) })
    }
}

# 权重单独放，绕开 Test-Skip 的 *.pt 排除规则
if ($WeightFile) {
    $wDest = Join-Path $stage "weights\$WeightName"
    $wDestDir = Split-Path -Parent $wDest
    if (-not (Test-Path $wDestDir)) { New-Item -ItemType Directory -Force -Path $wDestDir | Out-Null }
    Copy-Item -LiteralPath $WeightFile -Destination $wDest -Force
    $wLen = (Get-Item -LiteralPath $wDest).Length
    $copied.Add([pscustomobject]@{ Path = "weights\$WeightName"; KB = [math]::Round($wLen / 1KB, 1) })
}

# --------------------------------------------------------------------------- #
# 3. 清单 + 压缩
# --------------------------------------------------------------------------- #
Write-Host ''
Write-Host '=== 打包清单 ===' -ForegroundColor Green
$copied | Sort-Object Path | Format-Table -AutoSize -Property Path, KB | Out-String -Width 120 | Write-Host

$totalKB = ($copied | Measure-Object -Property KB -Sum).Sum
if ($totalKB -gt 1024) {
    $sizeText = "{0:N1} MB" -f ($totalKB / 1024)
}
else {
    $sizeText = "{0:N1} KB" -f $totalKB
}
Write-Host ("共 {0} 个文件，{1}（跳过 {2} 个被排除项）" -f $copied.Count, $sizeText, $skipped)
Write-Host ''

if (Test-Path $OutFile) { Remove-Item $OutFile -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $OutFile -CompressionLevel Optimal
$zipMB = [math]::Round((Get-Item $OutFile).Length / 1MB, 2)
Write-Host "[done] 已生成：$OutFile  ($zipMB MB)" -ForegroundColor Green

# --------------------------------------------------------------------------- #
# 4. 回扫：确认没混进数据（weights\ 下那一个权重是预期内的）
# --------------------------------------------------------------------------- #
$allowedWeight = "weights/$WeightName"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($OutFile)
$bad = $zip.Entries | Where-Object {
    $n = $_.FullName -replace '\\', '/'
    if ($n -eq $allowedWeight) { return $false }
    ($n -like '*.pt' -or $n -like '*.jpeg' -or $n -like '*.jpg' -or
     $n -match '(^|/)data/' -or $n -match 'checkpoints' -or $n -match 'logs3')
}
$entryCount = $zip.Entries.Count
$zip.Dispose()

if ($bad) {
    Write-Host '[warn] 压缩包里出现了疑似数据文件：' -ForegroundColor Yellow
    $bad | ForEach-Object { Write-Host "        $($_.FullName)" -ForegroundColor Yellow }
}
else {
    $tail = if ($WeightFile) { '（含 weights\ 下 1 个权重，符合预期）' } else { '，无数据/权重文件' }
    Write-Host "[check] 回扫 $entryCount 个条目，干净$tail" -ForegroundColor Green
}

Remove-Item -Recurse -Force $stage

# --------------------------------------------------------------------------- #
# 4.5 权重指纹：到实验室机器上用它核对文件是否完整
# --------------------------------------------------------------------------- #
if ($WeightFile) {
    $hash = (Get-FileHash -LiteralPath $WeightFile -Algorithm SHA256).Hash
    Write-Host ''
    Write-Host "[sha256] $WeightName" -ForegroundColor Cyan
    Write-Host "         $hash"
    Write-Host '         到实验室机器核对：Get-FileHash .\weights\histoplus_cellvit_segmentor_20x.pt -Algorithm SHA256'
}

# --------------------------------------------------------------------------- #
# 5. git 信息（方便在新机器上对照改了哪些）
# --------------------------------------------------------------------------- #
if (Test-Path (Join-Path $Root '.git')) {
    $commit = (& git -C $Root rev-parse --short HEAD 2>$null)
    $branch = (& git -C $Root rev-parse --abbrev-ref HEAD 2>$null)
    $dirty = (& git -C $Root status --porcelain 2>$null | Measure-Object -Line).Lines
    Write-Host ''
    Write-Host "[git  ] 分支 $branch  提交 $commit  未提交改动 $dirty 个文件"
    if ($dirty -gt 0 -and -not $IncludeGit) {
        Write-Host '        注意：zip 里是工作区当前内容（含未提交改动），不是 commit 快照' -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host '--- 在实验室电脑上 ---' -ForegroundColor Cyan
Write-Host '  1) 解压到任意目录，例如 D:\work\toxpath'
Write-Host '  2) conda activate 你的环境（python 3.10 + torch cu124 + timm==1.0.8）'
Write-Host '  3) python scripts\check_env.py --data-root "D:\data\toxpath"'
if ($WeightFile) {
    Write-Host '       权重已随包解压到 <项目根>\weights\，paths.py 会自动找到，无需配置'
}
else {
    Write-Host '       本包不含权重；没有权重时加 --IncludeWeights 重新打包，'
    Write-Host '       或手动把 .pt 放到 <项目根>\weights\ 下'
}
Write-Host '  4) .\scripts\train_local.bat -Mode smoke'
Write-Host '  详见 docs\本地训练SOP.md'
