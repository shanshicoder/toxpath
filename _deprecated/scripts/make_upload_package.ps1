# Build the AutoDL upload package (training-only files).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\make_upload_package.ps1 -Out D:\toxpath_upload.tar
#
# Output:
#   E:\project\toxpath_upload.tar  (~1.5 GB, uncompressed; JPEGs are already compressed)
#
# Extracted layout:
#   toxpath/
#   |-- model/ dataset/ scripts/ tests/
#   |-- train_3class.py  requirements.txt  README.md
#   `-- data/toxpath/{train_patches,train_masks_classes,_patch_labels_3class.csv}
#   histoplus_cellvit_segmentor_20x.pt
#
# NOTE: keep this file ASCII-only so Windows PowerShell 5.1 can parse it
# without a UTF-8 BOM.

param(
    [string]$Out = "E:\project\toxpath_upload.tar"
)

$ErrorActionPreference = "Stop"

$project = "E:\project\toxpath"
$weights = "E:\project\histoplus_cellvit_segmentor_20x.pt"

if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    throw "tar.exe not found (Windows 10/11 ships it). Use 7-Zip instead."
}
if (-not (Test-Path $project)) { throw "Project dir not found: $project" }
if (-not (Test-Path $weights)) { throw "Weights not found: $weights" }

# Paths relative to E:\project
$items = @(
    "toxpath\model",
    "toxpath\dataset",
    "toxpath\scripts",
    "toxpath\tests",
    "toxpath\train_3class.py",
    "toxpath\requirements.txt",
    "toxpath\README.md",
    "toxpath\data\toxpath\train_patches",
    "toxpath\data\toxpath\train_masks_classes",
    "toxpath\data\toxpath\_patch_labels_3class.csv",
    "histoplus_cellvit_segmentor_20x.pt"
)

Write-Host "== Items to package ==" -ForegroundColor Cyan
foreach ($it in $items) {
    $p = Join-Path "E:\project" $it
    if (-not (Test-Path $p)) { throw "Missing: $p" }
    $f = Get-Item $p
    if ($f.PSIsContainer) {
        $s = (Get-ChildItem $p -Recurse -File | Measure-Object Length -Sum)
        "{0,-52} {1,10:N1} MB  {2,6} files" -f $it, ($s.Sum / 1MB), $s.Count | Write-Host
    } else {
        "{0,-52} {1,10:N1} MB" -f $it, ($f.Length / 1MB) | Write-Host
    }
}

if (Test-Path $Out) {
    Write-Host "`nRemoving existing: $Out" -ForegroundColor Yellow
    Remove-Item -LiteralPath $Out -Force
}

Write-Host "`n== Creating tar (no compression) ==" -ForegroundColor Cyan
$t0 = Get-Date
$tarArgs = @("-cf", $Out, "-C", "E:\project", "--exclude=*__pycache__*", "--exclude=*.pyc") + $items
& tar.exe @tarArgs
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }

$file = Get-Item $Out
$hash = (Get-FileHash -LiteralPath $Out -Algorithm SHA256).Hash
$secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

Write-Host "`n== Done ==" -ForegroundColor Green
Write-Host ("File  : {0}" -f $file.FullName)
Write-Host ("Size  : {0:N2} GB ({1:N0} bytes)" -f ($file.Length / 1GB), $file.Length)
Write-Host ("Time  : {0} s" -f $secs)
Write-Host ("SHA256: {0}" -f $hash)
Write-Host "`nUpload example:" -ForegroundColor Cyan
Write-Host ("  scp -P <port> `"{0}`" root@<host>:/root/autodl-tmp/" -f $file.FullName)
Write-Host "Verify on server:"
Write-Host ("  sha256sum {0}" -f $file.Name)
