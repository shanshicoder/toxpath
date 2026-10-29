<#
.SYNOPSIS
    从 GitHub Release 下载 HistoPLUS 预训练权重（271 MB），并校验完整性。

.DESCRIPTION
    权重体积超过 GitHub 单文件 100 MB 硬限制，所以不进 Git 仓库，
    而是作为 Release 附件分发。本脚本负责：

      1. 按 仓库 + tag + 文件名 拼出下载地址
      2. 有 token 就带认证（私有仓库必须），没 token 先试匿名（公开仓库可用）
      3. 用 curl 断点续传下载到临时文件，中断后重跑能接着下
      4. 校验字节数与 SHA256，对不上就报错并保留损坏文件供排查
      5. 校验通过才落到 weights\，避免半截文件被训练误用

    私有仓库需要 token：GitHub → Settings → Developer settings →
    Personal access tokens → 勾选 repo 权限。放进环境变量：
        setx GITHUB_TOKEN "ghp_xxx"
    或者每次 -Token "ghp_xxx" 传进来。

.PARAMETER Repo
    仓库全名，默认 shanshicoder/toxpath

.PARAMETER Tag
    Release 的 tag，默认 weights-v1（建 Release 时的标签名）

.PARAMETER Asset
    Release 里的文件名，默认 histoplus_cellvit_segmentor_20x.pt

.PARAMETER Dest
    保存目录，默认 <项目根>\weights

.PARAMETER Token
    GitHub PAT。默认读环境变量 GITHUB_TOKEN

.PARAMETER Force
    目标文件已存在且校验通过时也重新下载

.EXAMPLE
    .\scripts\fetch_weights.ps1

.EXAMPLE
    .\scripts\fetch_weights.ps1 -Token "ghp_xxxxxxxx"

.EXAMPLE
    .\scripts\fetch_weights.ps1 -Tag weights-v2 -Force
#>

[CmdletBinding()]
param(
    [string]$Repo = 'shanshicoder/toxpath',

    [string]$Tag = 'weights-v1',

    [string]$Asset = 'histoplus_cellvit_segmentor_20x.pt',

    [string]$Dest = '',

    [string]$Token = $env:GITHUB_TOKEN,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# 控制台统一 UTF-8，避免中文提示在 cmd / PowerShell 里乱码
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }

# 进度条会让 Invoke-WebRequest / 大文件操作慢好几倍
$ProgressPreference = 'SilentlyContinue'

# --------------------------------------------------------------------------- #
# 0. 基准值（与 make_code_package.ps1 打印的一致）
# --------------------------------------------------------------------------- #
$ExpectedSizeBytes = 284507558
$ExpectedSha256 = 'AA19DC31C109C0A4FCF7BD773559C45480CEE792A331AE0A9FFE42EB08267BFC'

$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Dest)) { $Dest = Join-Path $Root 'weights' }
if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Force -Path $Dest | Out-Null }

$Target = Join-Path $Dest $Asset

Write-Host "[repo] $Repo" -ForegroundColor Cyan
Write-Host "[tag ] $Tag" -ForegroundColor Cyan
Write-Host "[file] $Asset" -ForegroundColor Cyan
Write-Host "[dest] $Target" -ForegroundColor Cyan

# --------------------------------------------------------------------------- #
# 1. 已存在且完整就跳过
# --------------------------------------------------------------------------- #
function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToUpper()
}

if ((Test-Path $Target) -and -not $Force) {
    $len = (Get-Item $Target).Length
    if ($len -eq $ExpectedSizeBytes) {
        Write-Host '[skip] 目标文件已存在且大小正确，正在核对哈希...' -ForegroundColor Yellow
        $h = Get-FileSha256 $Target
        if ($h -eq $ExpectedSha256) {
            Write-Host '[done] 权重已就位，无需下载' -ForegroundColor Green
            exit 0
        }
        Write-Host '[warn] 哈希不符，可能是坏的，重新下载' -ForegroundColor Yellow
    }
    else {
        Write-Host "[warn] 已存在但大小不对（$len / $ExpectedSizeBytes），重新下载" -ForegroundColor Yellow
    }
}

# --------------------------------------------------------------------------- #
# 2. 下载
# --------------------------------------------------------------------------- #
$Url = "https://github.com/$Repo/releases/download/$Tag/$Asset"
$Tmp = "$Target.part"

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Host '[fail] 找不到 curl.exe（Windows 10 1803+ 自带），请改用浏览器下载：' -ForegroundColor Red
    Write-Host "       $Url" -ForegroundColor Yellow
    exit 3
}

$curlArgs = @(
    '-L',                    # 跟随重定向（GitHub 会跳到 objects.githubusercontent.com）
    '--fail',                # 404/403 直接失败，不把错误页存成文件
    '--retry', '3',          # 网络抖动重试
    '--retry-delay', '5',
    '-C', '-',               # 断点续传
    '--connect-timeout', '20',
    '--no-progress-meter',   # 关掉进度表；否则 PowerShell 会把 stderr 当错误刷红
    '-o', $Tmp
)

if (-not [string]::IsNullOrWhiteSpace($Token)) {
    # Release 附件对私有仓库需要 token；Bearer 是当前推荐的写法
    $curlArgs += @('-H', "Authorization: Bearer $Token")
    Write-Host '[auth] 使用 GITHUB_TOKEN 认证' -ForegroundColor DarkGray
}
else {
    Write-Host '[auth] 未提供 token，按公开仓库尝试' -ForegroundColor DarkGray
}

$curlArgs += $Url

Write-Host ''
Write-Host '[down] 开始下载（271 MB，视网速 1-15 分钟）...' -ForegroundColor Cyan
Write-Host "       中途可以用 dir `"$Tmp`" 看进度；断了重跑本脚本会续传" -ForegroundColor DarkGray

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
# 2>&1 把 curl 的 stderr 收进变量，避免 PowerShell 把每行进度当错误记录刷屏
$curlOut = & curl.exe @curlArgs 2>&1 | Out-String
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEap

if ($code -ne 0) {
    Write-Host ''
    Write-Host "[fail] 下载失败（curl 退出码 $code）" -ForegroundColor Red

    $tail = ($curlOut -split "`r?`n" | Where-Object { $_ -match '^curl:' } | Select-Object -Last 3)
    if ($tail) {
        Write-Host '       curl 最后几行：' -ForegroundColor DarkGray
        $tail | ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
    }

    switch ($code) {
        22 {
            Write-Host '       HTTP 4xx/5xx：Release 或文件不存在，或没有权限' -ForegroundColor Yellow
            Write-Host "       打开确认：https://github.com/$Repo/releases/tag/$Tag" -ForegroundColor Yellow
            Write-Host '       私有仓库请设 token：setx GITHUB_TOKEN "ghp_xxx" 后重开窗口' -ForegroundColor Yellow
        }
        6 {  Write-Host '       域名解析失败：检查网络 / 代理' -ForegroundColor Yellow }
        7 {  Write-Host '       连不上 github.com:443：该端口可能被封（国内常见），' -ForegroundColor Yellow
             Write-Host '       挂代理后重跑；或改用浏览器下载（方式 B）；或 U 盘拷（方式 C）' -ForegroundColor Yellow }
        18 { Write-Host '       传输中断（文件下了一部分）：直接重跑本脚本，会从断点续传' -ForegroundColor Yellow }
        28 { Write-Host '       超时：若是 443 被封，脚本无法自行绕过，需代理或浏览器下载；' -ForegroundColor Yellow
             Write-Host '       否则重跑即可续传' -ForegroundColor Yellow }
        35 { Write-Host '       TLS 握手失败：多半是代理/防火墙拦截，检查代理设置' -ForegroundColor Yellow }
        56 { Write-Host '       连接被重置（TLS close_notify）：GitHub 附件域经常这样，' -ForegroundColor Yellow
             Write-Host '       直接重跑 2-3 次通常能续传完成；一直不行就用浏览器下载' -ForegroundColor Yellow }
        default { Write-Host '       重跑一次通常会续传；反复失败就用浏览器手动下载' -ForegroundColor Yellow }
    }
    exit 4
}

# --------------------------------------------------------------------------- #
# 3. 校验
# --------------------------------------------------------------------------- #
$len = (Get-Item $Tmp).Length
Write-Host ''
Write-Host "[size] $len 字节" -ForegroundColor Cyan

if ($len -ne $ExpectedSizeBytes) {
    Write-Host "[fail] 大小不符：期望 $ExpectedSizeBytes，实际 $len" -ForegroundColor Red
    Write-Host "       半截文件保留在 $Tmp，重跑脚本会续传" -ForegroundColor Yellow
    exit 5
}

Write-Host '[hash] 正在算 SHA256...' -ForegroundColor Cyan
$sha = Get-FileSha256 $Tmp
if ($sha -ne $ExpectedSha256) {
    Write-Host '[fail] SHA256 不符，文件可能在传输中损坏：' -ForegroundColor Red
    Write-Host "       期望 $ExpectedSha256" -ForegroundColor Red
    Write-Host "       实际 $sha" -ForegroundColor Red
    Write-Host "       可删掉 $Tmp 后重新下载" -ForegroundColor Yellow
    exit 6
}

if (Test-Path $Target) { Remove-Item $Target -Force }
Move-Item $Tmp $Target
Write-Host ''
Write-Host "[done] 校验通过，权重就位：$Target" -ForegroundColor Green
Write-Host "       SHA256 $sha" -ForegroundColor DarkGray
Write-Host ''
Write-Host '下一步：python scripts\check_env.py --data-root "你的数据目录"' -ForegroundColor Cyan
