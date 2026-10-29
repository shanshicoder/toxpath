<#
.SYNOPSIS
    把本地改动提交并推送到 GitHub（笔记本 → 仓库 → 实验室机器拉取）。

.DESCRIPTION
    笔记本改完代码后跑这一条，实验室机器那边 git pull 就能拿到。
    脚本做四件事：

      1. 检查仓库/身份/远程是否配好
      2. 扫描待提交文件，拦住不该进仓库的东西（大文件、权重、数据、日志）
      3. 有改动就 add + commit（消息不给就自动按时间生成）
      4. 本地领先远程时 push，并打印实验室机器该执行的命令

    提交信息用 -Message 传，不传就用「sync: 2026-10-29 10:30 自动同步」。
    想写正经的 commit message，就先自己 git commit，再跑本脚本（它会只 push）。

.PARAMETER Message
    提交信息。含空格要加引号。

.PARAMETER DryRun
    只打印将要做的操作，不提交不推送。

.PARAMETER NoCommit
    只推送，不自动提交（适合已经手动 commit 过的情况）。

.PARAMETER Via
    推送协议。auto（默认）先探 SSH 能否免密认证，不行自动改用 HTTPS；
    ssh / https 强制指定。私钥有 passphrase 又没有 ssh-agent 时，用 https
    可以直接借用 Windows 凭据管理器里已存的 GitHub 凭据，不用输密码。

.EXAMPLE
    .\scripts\sync_github.ps1

.EXAMPLE
    .\scripts\sync_github.ps1 -Message "fix: 修正 fold5 采样权重"

.EXAMPLE
    .\scripts\sync_github.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [string]$Message = '',

    [switch]$DryRun,

    [switch]$NoCommit,

    # auto  = 先探 SSH 能否免密认证，不行就自动改用 HTTPS
    # ssh   = 强制走 SSH（需要私钥已解锁或能手动输 passphrase）
    # https = 强制走 HTTPS（凭据由 Windows 凭据管理器提供，不需要输密码）
    [ValidateSet('auto', 'ssh', 'https')]
    [string]$Via = 'auto'
)

$ErrorActionPreference = 'Stop'

# PS 5.1 在 Stop 模式下会把原生命令写 stderr 当成终止性错误，而 git 往 stderr
# 写的东西（进度、warning、"To <url>"、认证提示）全是正常输出。所有 git 调用
# 统一走这个包装器：临时放宽 EAP，用退出码判断成败。
function Invoke-Git {
    param([string[]]$GitArgs)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & git @GitArgs
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $old
    }
}

# 控制台统一 UTF-8，避免中文提示在 cmd / PowerShell 里乱码
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }

# --------------------------------------------------------------------------- #
# 0. 定位仓库
# --------------------------------------------------------------------------- #
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {

if (-not (Test-Path (Join-Path $Root '.git'))) {
    Write-Host "[fail] $Root 不是 git 仓库" -ForegroundColor Red
    exit 1
}

Write-Host "[repo] $Root" -ForegroundColor Cyan

# --------------------------------------------------------------------------- #
# 1. 身份与远程
# --------------------------------------------------------------------------- #
$userName = (git config user.name) 2>$null
$userEmail = (git config user.email) 2>$null
if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Host '[fail] 没配 git 身份，先执行：' -ForegroundColor Red
    Write-Host '       git config --global user.name  "你的名字"' -ForegroundColor Yellow
    Write-Host '       git config --global user.email "你的邮箱@example.com"' -ForegroundColor Yellow
    exit 1
}

$remoteUrl = (git remote get-url origin) 2>$null
if ([string]::IsNullOrWhiteSpace($remoteUrl)) {
    Write-Host '[fail] 没有 origin 远程，先执行：' -ForegroundColor Red
    Write-Host '       git remote add origin git@github.com:shanshicoder/toxpath.git' -ForegroundColor Yellow
    exit 1
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host "[git ] $userName <$userEmail>  分支 $branch" -ForegroundColor Cyan
Write-Host "[orig] $remoteUrl" -ForegroundColor Cyan
Write-Host ''

# --------------------------------------------------------------------------- #
# 2. 提交前护栏：不该进仓库的东西
# --------------------------------------------------------------------------- #
$changes = @(git status --porcelain)
$hasChanges = $changes.Count -gt 0

$blocked = New-Object System.Collections.Generic.List[string]
$bigFiles = New-Object System.Collections.Generic.List[string]

if ($hasChanges) {
    # 只看会被提交的（?? 和 M/A），排除删除项
    $candidates = @()
    foreach ($line in $changes) {
        $code = $line.Substring(0, 2)
        $path = $line.Substring(3).Trim().Trim('"')
        if ($code -match '^D' -or $code -match 'D$') { continue }
        $full = Join-Path $Root $path
        if (Test-Path $full -PathType Leaf) { $candidates += $full }
    }

    $blockExt = @('.pt', '.pth', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.zip',
                  '.npy', '.npz', '.log', '.ckpt', '.onnx')
    foreach ($f in $candidates) {
        $rel = $f.Substring($Root.Length + 1)
        $ext = [IO.Path]::GetExtension($f).ToLower()
        $sizeMB = [math]::Round((Get-Item $f).Length / 1MB, 2)

        # 数据目录整棵树都不许进
        if ($rel -match '^(data|weights|checkpoints3?|logs3?|eval3|runs3?)\\') {
            $blocked.Add("$rel（数据/产物目录）")
        }
        elseif ($blockExt -contains $ext -and $sizeMB -gt 1) {
            $blocked.Add("$rel（$sizeMB MB，二进制/权重）")
        }
        elseif ($sizeMB -gt 5) {
            $bigFiles.Add("$rel（$sizeMB MB）")
        }
    }
}

if ($blocked.Count -gt 0) {
    Write-Host '[fail] 这些文件不该进 Git 仓库：' -ForegroundColor Red
    $blocked | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
    Write-Host ''
    Write-Host '       权重走 GitHub Release（tag weights-v1），数据与 checkpoint 留在本机。' -ForegroundColor Yellow
    Write-Host '       若是 .gitignore 漏了规则，补上后重跑。已误 add 的用 git restore --staged <文件> 撤回。' -ForegroundColor Yellow
    exit 2
}

if ($bigFiles.Count -gt 0) {
    Write-Host '[warn] 以下文件较大（>5 MB），确认是有意提交：' -ForegroundColor Yellow
    $bigFiles | ForEach-Object { Write-Host "       $_" -ForegroundColor Yellow }
}

# --------------------------------------------------------------------------- #
# 3. 提交
# --------------------------------------------------------------------------- #
if (-not $hasChanges) {
    Write-Host '[git ] 工作区干净，无新改动' -ForegroundColor Gray
}
elseif ($NoCommit) {
    Write-Host '[git ] -NoCommit：跳过提交，只推送已有 commit' -ForegroundColor Gray
}
else {
    $msg = $Message
    if ([string]::IsNullOrWhiteSpace($msg)) {
        $msg = 'sync: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm') + ' 自动同步'
    }

    $n = $changes.Count
    Write-Host "[git ] 待提交 $n 项" -ForegroundColor Cyan

    if ($DryRun) {
        Write-Host "[dry ] 将执行：git add -A；git commit -m `"$msg`"" -ForegroundColor Yellow
    }
    else {
        $addCode = Invoke-Git @('add', '-A')
        if ($addCode -ne 0) {
            Write-Host '[fail] git add 失败' -ForegroundColor Red
            exit 3
        }
        $commitCode = Invoke-Git @('commit', '-q', '-m', $msg)
        if ($commitCode -ne 0) {
            Write-Host '[fail] git commit 失败（可能触发 pre-commit hook）' -ForegroundColor Red
            exit 3
        }
        Write-Host "[git ] 已提交：$msg" -ForegroundColor Green
    }
}

# --------------------------------------------------------------------------- #
# 4. 推送
# --------------------------------------------------------------------------- #
# SSH 地址 → HTTPS 地址：git@github.com:owner/repo.git → https://github.com/owner/repo.git
function ConvertTo-HttpsUrl {
    param([string]$Url)
    if ($Url -match '^git@([^:]+):(.+)$') {
        return "https://$($Matches[1])/$($Matches[2])"
    }
    if ($Url -match '^ssh://git@([^/]+)/(.+)$') {
        return "https://$($Matches[1])/$($Matches[2])"
    }
    return $Url
}

# 探测 SSH 能否免密认证：BatchMode 让 ssh 直接失败而不是卡在密码提示
function Test-SshAuth {
    param([string]$Url)
    $oldSsh = $env:GIT_SSH_COMMAND
    $oldEap = $ErrorActionPreference
    $env:GIT_SSH_COMMAND = 'ssh -o BatchMode=yes -o ConnectTimeout=8'
    # 认证失败时 git 会往 stderr 写一行，PS 5.1 在 Stop 模式下会把它当终止性错误。
    # 这里只在本次探测期间放宽，靠 $LASTEXITCODE 判断结果。
    $ErrorActionPreference = 'Continue'
    try {
        $null = git ls-remote --heads $Url 2>&1
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $oldEap
        if ($null -eq $oldSsh) { Remove-Item Env:\GIT_SSH_COMMAND -ErrorAction SilentlyContinue }
        else { $env:GIT_SSH_COMMAND = $oldSsh }
    }
}

# 先看有没有远程分支可比较；没有说明是首次推送
$hasUpstream = $false
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$null = git rev-parse --abbrev-ref "$branch@{upstream}" 2>&1
$hasUpstream = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

$ahead = 0
if ($hasUpstream) {
    $ahead = [int](git rev-list --count "$branch@{upstream}..$branch")
    Write-Host "[git ] 本地领先远程 $ahead 个提交" -ForegroundColor Cyan
}

# 选协议：auto 模式下先探 SSH 能否免密认证，不行就换 HTTPS
# DryRun 时前缀用 [plan]，免得看起来像真的推了
$actTag = '[push]'
if ($DryRun) { $actTag = '[plan]' }

$useHttps = $false
switch ($Via) {
    'https' { $useHttps = $true }
    'ssh' { $useHttps = $false }
    default {
        if ($remoteUrl -match '^https?://') {
            $useHttps = $true
        }
        else {
            $useHttps = -not (Test-SshAuth $remoteUrl)
            if ($useHttps) {
                Write-Host "$actTag SSH 无法免密认证（私钥有 passphrase 且没有 ssh-agent），自动改用 HTTPS" -ForegroundColor Yellow
            }
        }
    }
}

$pushUrl = $remoteUrl
$proxyArgs = @()
if ($useHttps) {
    $pushUrl = ConvertTo-HttpsUrl $remoteUrl
    # 实测踩过的坑：全局 http.proxy 写死成 127.0.0.1:7890，代理没开时连 443 都连不上，
    # 报 "Failed to connect to github.com:443 over proxy"。这里显式清空走直连。
    $proxyArgs = @('-c', 'http.proxy=', '-c', 'https.proxy=')
    Write-Host "$actTag HTTPS $pushUrl" -ForegroundColor Cyan
    Write-Host '       凭据取自 Windows 凭据管理器，不需要输密码' -ForegroundColor DarkGray
}
else {
    Write-Host "$actTag SSH $pushUrl" -ForegroundColor Cyan
    Write-Host '       若提示 Enter passphrase：输入 SSH 私钥密码（输入时不显示）' -ForegroundColor DarkGray
}

$pushArgs = $proxyArgs + @('push', '-u', $pushUrl, $branch)

if ($DryRun) {
    Write-Host ("[dry ] 将执行：git " + ($pushArgs -join ' ')) -ForegroundColor Yellow
    Write-Host ''
    Write-Host '[dry ] DryRun 结束，什么都没改' -ForegroundColor Yellow
    exit 0
}

if ($hasUpstream -and $ahead -eq 0) {
    Write-Host '[done] 远程已是最新，无需推送' -ForegroundColor Green
    exit 0
}

$pushCode = Invoke-Git $pushArgs

if ($pushCode -ne 0) {
    Write-Host ''
    Write-Host "[fail] 推送失败（退出码 $pushCode），常见原因：" -ForegroundColor Red
    Write-Host '       1) passphrase 输错 / 取消 → 重跑，或加 -Via https 改用凭据管理器' -ForegroundColor Yellow
    Write-Host '       2) Permission denied (publickey) → 公钥没加到 GitHub：' -ForegroundColor Yellow
    Write-Host '          复制 ~/.ssh/id_rsa.pub → GitHub Settings → SSH and GPG keys → New SSH key' -ForegroundColor Yellow
    Write-Host '       3) 远程有新提交（non-fast-forward）→ 先 git pull --rebase，再重跑' -ForegroundColor Yellow
    Write-Host '       4) HTTPS 连不上 → 本脚本已显式清空 proxy 走直连；若本机只能经代理上网，' -ForegroundColor Yellow
    Write-Host '          手动执行 git -c http.proxy=http://127.0.0.1:<可用端口> push <https地址> <分支>' -ForegroundColor Yellow
    exit 4
}

# push 到具体 URL 不会更新 refs/remotes/origin/*，补一次 fetch。
# 否则 git status 一直显示 ahead，脚本下次又会重复推送。
if ($useHttps) {
    $refspec = 'refs/heads/' + $branch + ':refs/remotes/origin/' + $branch
    $fetchArgs = $proxyArgs + @('fetch', $pushUrl, $refspec)
    $null = Invoke-Git $fetchArgs
}

Write-Host ''
Write-Host '[done] 已推送' -ForegroundColor Green
Write-Host ''
Write-Host '--- 实验室机器上 ---' -ForegroundColor Cyan
Write-Host '  首次：git clone git@github.com:shanshicoder/toxpath.git D:\work\toxpath'
Write-Host '  之后：cd D:\work\toxpath ; git pull'
Write-Host '  权重不从仓库拉（271 MB），从 Release weights-v1 下载后放 weights\'

}
finally {
    Pop-Location
}
