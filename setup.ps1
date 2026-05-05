# ============================================================
# CosyVoice3 Model & ttsfrd Download Script (Windows)
# ============================================================
# Usage: Open terminal in this folder and run:
#   powershell -ExecutionPolicy Bypass .\setup.ps1
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ModelDir = Join-Path $ScriptDir "pretrained_models\Fun-CosyVoice3-0.5B"
$TtsfrdDir = Join-Path $ScriptDir "pretrained_models\CosyVoice-ttsfrd"

Write-Host "===================================" -ForegroundColor Cyan
Write-Host " CosyVoice3 Model + ttsfrd Setup    " -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# ─── Find Python (skip WindowsApps fake python) ───
$python = $null
# Method 1: from where.exe output (skip WindowsApps)
$paths = where.exe python 2>$null | Where-Object { $_ -notmatch "WindowsApps" }
if ($paths) {
    $python = @($paths)[0]
}
# Method 2: check common install paths
if (-not (Test-Path $python)) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python310\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "C:\Python310\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $python = $c
            break
        }
    }
}
# Method 3: fallback to 'python' command
if (-not $python) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $python) {
    Write-Host "[ERROR] Python not found. Install Python 3.10 from python.org" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python: $python" -ForegroundColor Green

# ─── 1. Download CosyVoice3 model ───
$modelFlag = Join-Path $ModelDir "llm.pt"
if (Test-Path $modelFlag) {
    Write-Host "[OK] Model already exists, skip download" -ForegroundColor Green
} else {
    Write-Host "[..] Downloading CosyVoice3 model..." -ForegroundColor Yellow
    Write-Host "     Local path: $ModelDir"
    $tmpPy = Join-Path $ScriptDir "_tmp_download_model.py"
    $pyCode = "from modelscope import snapshot_download`nsnapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$ModelDir')`n"
    [System.IO.File]::WriteAllText($tmpPy, $pyCode, [System.Text.Encoding]::UTF8)
    & $python $tmpPy
    Remove-Item $tmpPy -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Model downloaded" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Model download failed" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# ─── 2. Download ttsfrd resource ───
$ttsfrdFlag = Join-Path $TtsfrdDir "ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl"
if (Test-Path $ttsfrdFlag) {
    Write-Host "[OK] ttsfrd resource already exists, skip download" -ForegroundColor Green
} else {
    Write-Host "[..] Downloading ttsfrd resource..." -ForegroundColor Yellow
    Write-Host "     Local path: $TtsfrdDir"
    $tmpPy = Join-Path $ScriptDir "_tmp_download_ttsfrd.py"
    $pyCode = "from modelscope import snapshot_download`nsnapshot_download('iic/CosyVoice-ttsfrd', local_dir='$TtsfrdDir')`n"
    [System.IO.File]::WriteAllText($tmpPy, $pyCode, [System.Text.Encoding]::UTF8)
    & $python $tmpPy
    Remove-Item $tmpPy -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] ttsfrd resource downloaded" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] ttsfrd download failed" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# ─── 3. Extract resource.zip ───
if (Test-Path $TtsfrdDir) {
    $resourceZip = Join-Path $TtsfrdDir "resource.zip"
    $resourceDir = Join-Path $TtsfrdDir "resource"
    if ((Test-Path $resourceZip) -and -not (Test-Path $resourceDir)) {
        Write-Host "[..] Extracting resource.zip..." -ForegroundColor Yellow
        Expand-Archive -Path $resourceZip -DestinationPath $TtsfrdDir -Force
        Write-Host "[OK] Extracted" -ForegroundColor Green
    } else {
        Write-Host "[OK] resource ready" -ForegroundColor Green
    }
}
Write-Host ""

Write-Host "===================================" -ForegroundColor Cyan
Write-Host " All done!                         " -ForegroundColor Cyan
Write-Host "                                   " -ForegroundColor Cyan
Write-Host " Start with Docker:                " -ForegroundColor Cyan
Write-Host "   docker compose up -d --build    " -ForegroundColor Cyan
Write-Host "                                   " -ForegroundColor Cyan
Write-Host " Or directly:                      " -ForegroundColor Cyan
Write-Host "   python server-opus.py           " -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
