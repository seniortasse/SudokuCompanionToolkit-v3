
param(
  [string[]]$Pairs = @("3,1","5,4","8,6","2,7"),
  # Optional: pass pairs as a single semicolon-separated string, e.g. "3,1;5,4;8,6;2,7"
  [string]$PairsStr = "",
  [string]$Src = ".\demo_export",
  [string]$Model = ".\vision\train\checkpoints\best.pt",
  [string]$Calib = ".\vision\train\calibration.json",
  [double]$InnerCrop = 0.92,
  [double]$Low = 0.85,
  [double]$Margin = 0.10,
  [double]$WideMargin = 0.15,
  [int]$ValEvery = 10,
  [string]$InferOut = ".\runs\infer_pairs",
  [string]$MineRoot = ".\vision\data\mine_pairs",
  [string]$RealRoot = ".\vision\data\real",
  [string]$SynthMetaTrain = ".\vision\data\synth\meta\train.jsonl",
  [string]$RealMetaTrain  = ".\vision\data\real\meta\train.jsonl",
  [string]$RealMetaVal    = ".\vision\data\real\meta\val.jsonl",
  [string]$SaveDir  = ".\vision\train\checkpoints",
  [int]$EpochsPerPair = 4,
  [int]$Batch = 512,
  [double]$LR = 0.0007,
  [string]$TrainMix = "0.7,0.3",
  [int]$Workers = 0
)

$ErrorActionPreference = "Stop"

function Run-Step([string]$Title, [scriptblock]$Block) {
  Write-Host "`n=== $Title ===" -ForegroundColor Cyan
  & $Block
  if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    throw "Step failed: $Title (exit $LASTEXITCODE)"
  }
}

# Parse PairsStr if provided (semicolon-separated)
if ($PairsStr -and $PairsStr.Trim().Length -gt 0) {
  $Pairs = @()
  foreach ($tok in ($PairsStr -split ';')) {
    $t = $tok.Trim()
    if ($t.Length -gt 0) { $Pairs += $t }
  }
}

# Validate pairs
if (-not $Pairs -or $Pairs.Count -eq 0) {
  throw 'No pairs provided. Use -PairsStr "3,1;5,4;8,6;2,7" or -Pairs "3,1","5,4","8,6","2,7"'
}
foreach ($p in $Pairs) {
  if ($p -notmatch '^\s*\d,\d\s*$') {
    throw "Invalid pair '$p'. Expected format 'd,d' (e.g., 3,1)"
  }
}

# Ensure output roots exist
New-Item -ItemType Directory -Force -Path $InferOut | Out-Null
New-Item -ItemType Directory -Force -Path $MineRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $SaveDir -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path ".\runs\eval_pairs" | Out-Null

# 0) One inference for all pairs (shared all_preds.csv)
Run-Step "0) Shared inference for all pairs -> $InferOut\all_preds.csv" {
  & python ".\vision\infer\predict_cells.py" `
    --src $Src `
    --model $Model `
    --img 28 `
    --device cpu `
    --calib $Calib `
    --low $Low `
    --margin $Margin `
    --inner-crop $InnerCrop `
    --out $InferOut
  if (-not (Test-Path (Join-Path $InferOut "all_preds.csv"))) {
    throw "Inference did not produce $InferOut\all_preds.csv"
  }
}

# 1) Iterate pairs, run the loop with --pair-only and warm-start chaining
$currentModel = $Model
foreach ($pair in $Pairs) {
  $tag = ($pair -replace ",","_")
  $mineOut = Join-Path $MineRoot "mine_$tag"
  $evalOut = Join-Path ".\runs\eval_pairs" "eval_$tag"

  Write-Host "`n---- Pair $pair (mine: $mineOut) ----" -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path $mineOut | Out-Null

  & .\tools\mining\run_loop.ps1 `
    -Src $Src `
    -Model $currentModel `
    -Calib $Calib `
    -Digits $pair `
    -PairOnly `
    -InnerCrop $InnerCrop `
    -Low $Low `
    -Margin $Margin `
    -WideMargin $WideMargin `
    -ValEvery $ValEvery `
    -InferOut $InferOut `
    -MineOut $mineOut `
    -RealRoot $RealRoot `
    -SynthMetaTrain $SynthMetaTrain `
    -RealMetaTrain $RealMetaTrain `
    -RealMetaVal $RealMetaVal `
    -SaveDir $SaveDir `
    -Epochs $EpochsPerPair `
    -Batch $Batch `
    -LR $LR `
    -TrainMix $TrainMix `
    -Workers $Workers `
    -EvalOut $evalOut `
    -SkipInfer

  # Chain warm-start to the latest model for the next pair
  $candidate = Join-Path $SaveDir "best.pt"
  if (Test-Path $candidate) {
    $currentModel = $candidate
  } else {
    Write-Host "WARN: $candidate not found; continuing with previous model" -ForegroundColor DarkYellow
  }
}

Write-Host "`nAll pairs processed ✅  Latest model: $currentModel" -ForegroundColor Green
Write-Host "Eval outputs per pair under: .\runs\eval_pairs" -ForegroundColor Green
