<#  tools/quality.ps1
Runs lint, format, type-check, and tests in one go.

Usage (PowerShell 7):
  pwsh tools/quality.ps1
  pwsh tools/quality.ps1 -UnsafeFix          # allow ruff's --unsafe-fixes
  pwsh tools/quality.ps1 -NoTests            # skip pytest
  pwsh tools/quality.ps1 -NoTypecheck        # skip mypy
  pwsh tools/quality.ps1 -Docs               # also (re)build docs
  pwsh tools/quality.ps1 -ReinstallTools     # force (re)install dev tools
#>

param(
  [switch]$UnsafeFix = $false,
  [switch]$NoTests   = $false,
  [switch]$NoTypecheck = $false,
  [switch]$Docs = $false,
  [switch]$ReinstallTools = $false
)

$ErrorActionPreference = "Stop"

# ---- Helpers ---------------------------------------------------------------

function Write-Phase($msg) { Write-Host "▶ $msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "⚠ $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "✗ $msg" -ForegroundColor Red }

# Ensure we're on PowerShell 7+
if ($PSVersionTable.PSVersion.Major -lt 7) {
  Write-Err "Please run with PowerShell 7 (pwsh)."
  exit 1
}

# Move to repo root if script was called from elsewhere
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

# ---- Python / venv ---------------------------------------------------------

# Use repo-local venv
$venvAct = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path ".\.venv")) {
  Write-Phase "Creating virtual environment (.venv)"
  python -m venv .venv
}
# Activate (in-process)
& $venvAct

# Confirm interpreter path
$py = (python -c "import sys; print(sys.executable)") 2>$null
if (-not $py) { Write-Err "Python not found inside venv."; exit 1 }
Write-OK "Using Python: $py"

# ---- Dev tools -------------------------------------------------------------

$devPkgs = @("ruff","black","mypy","pytest","pytest-cov")
if ($ReinstallTools) {
  Write-Phase "Reinstalling dev tools"
  python -m pip install -U pip
  python -m pip install -U $devPkgs
} else {
  # install any missing
  $need = @()
  foreach ($p in $devPkgs) {
    $found = (python -m pip show $p 2>$null)
    if (-not $found) { $need += $p }
  }
  if ($need.Count -gt 0) {
    Write-Phase "Installing dev tools: $($need -join ', ')"
    python -m pip install -U $need
  }
}

# ---- Lint & format ---------------------------------------------------------

Write-Phase "Ruff (lint) + Black (format)"
$ruffArgs = @("check",".","--fix")
if ($UnsafeFix) { $ruffArgs += "--unsafe-fixes" }
python -m ruff @ruffArgs
python -m black .

# ---- Type checking ---------------------------------------------------------

if (-not $NoTypecheck) {
  Write-Phase "Mypy (type-check)"
  python -m mypy .
} else {
  Write-Warn "Skipping mypy (--NoTypecheck)"
}

# ---- Tests ----------------------------------------------------------------

if (-not $NoTests) {
  Write-Phase "Pytest (unit tests)"
  python -m pytest -q
} else {
  Write-Warn "Skipping pytest (--NoTests)"
}

# ---- Docs (optional) -------------------------------------------------------

if ($Docs) {
  if (Test-Path "tools\build_docs.py") {
    Write-Phase "Building docs (pdoc)"
    python tools\build_docs.py
    Write-OK "Docs built → docs\site"
  } else {
    Write-Warn "Docs not built (tools/build_docs.py not found)"
  }
}

Write-OK "Quality pipeline complete."