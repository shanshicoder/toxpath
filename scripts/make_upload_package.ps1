# Build the cloud upload packages for ToxPath (see the cloud SOP in docs/, section 3).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1 -CodeOnly
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1 -WithMasks
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1 -OutDir D:\dist
#
# Output (default E:\project\toxpath\dist):
#   toxpath_code.tar.gz   git archive of HEAD, ~97 KB
#   toxpath_data.tar      train_patches + train_masks_classes + _patch_labels_3class.csv
#   SHA256SUMS            `sha256sum -c SHA256SUMS` compatible
#   CHECKSUMS.txt         human readable manifest
#
# Extracted layout on the server:
#   tar -xzf toxpath_code.tar.gz -C toxpath
#   tar -xf  toxpath_data.tar    -C toxpath/data/toxpath
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 parses it without a
# UTF-8 BOM; adding Chinese characters here breaks the parser.

param(
    [string]$RepoRoot = "E:\project\toxpath",
    [string]$OutDir   = "",
    [switch]$CodeOnly,
    [switch]$WithMasks
)

$ErrorActionPreference = "Stop"

if (-not $OutDir) { $OutDir = Join-Path $RepoRoot "dist" }
$DataRoot = Join-Path $RepoRoot "data\toxpath"
$CsvName  = "_patch_labels_3class.csv"

# ---------- preflight ----------

foreach ($exe in @("git.exe", "tar.exe")) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "$exe not found in PATH (Windows 10/11 ships tar.exe; Git for Windows ships git.exe)."
    }
}
if (-not (Test-Path -LiteralPath $RepoRoot)) { throw "Repo root not found: $RepoRoot" }
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot ".git"))) { throw "Not a git repo: $RepoRoot" }
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$commit = (& git -C $RepoRoot rev-parse --short HEAD).Trim()
$branch = (& git -C $RepoRoot rev-parse --abbrev-ref HEAD).Trim()
$dirty  = (& git -C $RepoRoot status --porcelain)
if ($dirty) {
    Write-Host "WARNING: working tree has uncommitted changes; the code package uses HEAD only." -ForegroundColor Yellow
    Write-Host "         Commit first if those changes matter for the run." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "commit $commit ($branch)  ->  $OutDir" -ForegroundColor Cyan
Write-Host ""

# ---------- code package ----------

$codeTar = Join-Path $OutDir "toxpath_code.tar.gz"
if (Test-Path -LiteralPath $codeTar) { Remove-Item -LiteralPath $codeTar -Force }

Write-Host "== building toxpath_code.tar.gz ==" -ForegroundColor Cyan
& git -C $RepoRoot archive --format=tar.gz -o $codeTar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed with exit code $LASTEXITCODE" }

$codeFiles = (& git -C $RepoRoot ls-files | Measure-Object).Count
$codeEntry = (& tar.exe -tf $codeTar | Measure-Object).Count
$codeInfo  = Get-Item -LiteralPath $codeTar
Write-Host ("   {0:N0} B, {1} entries (git tracks {2})" -f $codeInfo.Length, $codeEntry, $codeFiles)

# ---------- data package ----------

$dataTar = $null
$dataItems = @()
if ($CodeOnly) {
    Write-Host ""
    Write-Host "== skipping data package (-CodeOnly) ==" -ForegroundColor Yellow
}
else {
    if (-not (Test-Path -LiteralPath $DataRoot)) { throw "Data root not found: $DataRoot" }

    $dataItems = @("train_patches", "train_masks_classes")
    if ($WithMasks) { $dataItems += "train_masks" }

    Write-Host ""
    Write-Host "== building toxpath_data.tar ==" -ForegroundColor Cyan
    foreach ($it in ($dataItems + $CsvName)) {
        $p = Join-Path $DataRoot $it
        if (-not (Test-Path -LiteralPath $p)) { throw "Missing data item: $p" }
        $f = Get-Item -LiteralPath $p
        if ($f.PSIsContainer) {
            $s = Get-ChildItem -LiteralPath $p -Recurse -File | Measure-Object Length -Sum
            Write-Host ("   {0,-22} {1,10:N1} MB  {2,6} files" -f $it, ($s.Sum / 1MB), $s.Count)
        }
        else {
            Write-Host ("   {0,-22} {1,10:N2} MB" -f $it, ($f.Length / 1MB))
        }
    }

    $dataTar = Join-Path $OutDir "toxpath_data.tar"
    if (Test-Path -LiteralPath $dataTar) { Remove-Item -LiteralPath $dataTar -Force }

    $t0 = Get-Date
    $tarArgs = @("-cf", $dataTar, "-C", $DataRoot) + $dataItems + @($CsvName)
    & tar.exe @tarArgs
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $dataInfo = Get-Item -LiteralPath $dataTar
    Write-Host ("   {0:N0} B, packed in {1} s" -f $dataInfo.Length, $secs)
}

# ---------- verification ----------
# The three counts below must agree. A truncated upload shows up here first.

Write-Host ""
Write-Host "== verifying archive contents ==" -ForegroundColor Cyan
$ok = $true

if ($dataTar) {
    $listing = & tar.exe -tf $dataTar

    $nPatch = ($listing | Where-Object { $_ -match '^train_patches/.*\.(jpeg|jpg|png)$' }).Count
    $nMask  = ($listing | Where-Object { $_ -match '^train_masks_classes/.*\.png$' }).Count
    $nCsv   = ($listing | Where-Object { $_ -eq $CsvName }).Count

    $srcPatch = (Get-ChildItem -LiteralPath (Join-Path $DataRoot "train_patches") -Recurse -File | Measure-Object).Count
    $srcMask  = (Get-ChildItem -LiteralPath (Join-Path $DataRoot "train_masks_classes") -Recurse -File | Measure-Object).Count
    $csvLines = (Get-Content -LiteralPath (Join-Path $DataRoot $CsvName) | Measure-Object).Count

    Write-Host ("   patches in tar : {0,6}   on disk: {1,6}" -f $nPatch, $srcPatch)
    Write-Host ("   masks   in tar : {0,6}   on disk: {1,6}" -f $nMask, $srcMask)
    Write-Host ("   csv     in tar : {0,6}   rows   : {1,6} (header + {2} patches)" -f $nCsv, $csvLines, ($csvLines - 1))

    if ($nPatch -ne $srcPatch) { Write-Host "   FAIL: patch count mismatch" -ForegroundColor Red; $ok = $false }
    if ($nMask -ne $srcMask) { Write-Host "   FAIL: mask count mismatch" -ForegroundColor Red; $ok = $false }
    if ($nPatch -ne $nMask) { Write-Host "   FAIL: patches != masks" -ForegroundColor Red; $ok = $false }
    if ($nCsv -ne 1) { Write-Host "   FAIL: csv not found in archive" -ForegroundColor Red; $ok = $false }
    if ($csvLines -ne ($nPatch + 1)) { Write-Host "   FAIL: csv rows != patches + header" -ForegroundColor Red; $ok = $false }

    if ($nMask -eq 0 -or $nPatch -eq 0) { $ok = $false }
}
else {
    $nPatch = 0; $nMask = 0; $nCsv = 0; $csvLines = 0
}

if ($codeEntry -lt 10) { Write-Host "   FAIL: code archive looks empty" -ForegroundColor Red; $ok = $false }
if (-not $ok) { throw "verification failed; do not upload these packages" }
Write-Host "   all checks passed" -ForegroundColor Green

# ---------- checksums ----------

Write-Host ""
Write-Host "== checksums ==" -ForegroundColor Cyan

$packages = @($codeTar)
if ($dataTar) { $packages += $dataTar }

$sumLines = foreach ($p in $packages) {
    $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()
    "$h  $(Split-Path -Path $p -Leaf)"
}
Set-Content -LiteralPath (Join-Path $OutDir "SHA256SUMS") -Value $sumLines -Encoding ascii
foreach ($l in $sumLines) { Write-Host "   $l" }

$manifest = @()
$manifest += "ToxPath cloud upload packages"
$manifest += "generated : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$manifest += "commit    : $commit ($branch)"
$manifest += "host      : $env:COMPUTERNAME"
$manifest += ""
$manifest += "[toxpath_code.tar.gz]"
$manifest += "  size    : $($codeInfo.Length) B"
$manifest += "  entries : $codeEntry"
$manifest += "  extract : tar -xzf toxpath_code.tar.gz -C toxpath"
$manifest += ""
if ($dataTar) {
    $manifest += "[toxpath_data.tar]"
    $manifest += "  size    : $($dataInfo.Length) B"
    $manifest += "  items   : $($dataItems -join ', '), $CsvName"
    $manifest += "  patches : $nPatch"
    $manifest += "  masks   : $nMask"
    $manifest += "  csv rows: $csvLines"
    $manifest += "  extract : tar -xf toxpath_data.tar -C toxpath/data/toxpath"
    $manifest += ""
}
$manifest += "Verify on the server:  sha256sum -c SHA256SUMS"
$manifest += "Check counts on the server:"
$manifest += "  find train_patches -type f | wc -l"
$manifest += "  find train_masks_classes -type f | wc -l"
$manifest += "  wc -l _patch_labels_3class.csv"
Set-Content -LiteralPath (Join-Path $OutDir "CHECKSUMS.txt") -Value $manifest -Encoding ascii

# ---------- next step ----------

Write-Host ""
Write-Host "== done ==" -ForegroundColor Green
Write-Host "   $OutDir"
Write-Host ""
Write-Host "Upload (replace <port> and <host>):" -ForegroundColor Cyan
Write-Host "   scp -P <port> `"$codeTar`" root@<host>:/root/autodl-tmp/work/"
if ($dataTar) {
    Write-Host "   rsync -avP -e `"ssh -p <port>`" `"$dataTar`" root@<host>:/root/autodl-tmp/work/"
    Write-Host "   # no rsync? use WinSCP / SFTP, it resumes broken transfers"
}
Write-Host "   scp -P <port> `"E:\project\histoplus_cellvit_segmentor_20x.pt`" root@<host>:/root/autodl-tmp/"
Write-Host ""
Write-Host "Then on the server: follow the cloud SOP in docs/ (section 3)." -ForegroundColor Cyan
