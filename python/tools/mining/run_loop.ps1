param(
  [string]$Src = ".\demo_export",
  [string]$Model = ".\vision\train\checkpoints\best.pt",
  [string]$Calib = ".\vision\train\calibration.json",
  [string]$Digits = "3,1,2,5,4,8,6,7",
  [double]$InnerCrop = 0.92,
  [double]$Low = 0.85,
  [double]$Margin = 0.10,
  [double]$WideMargin = 0.15,
  [string]$InferOut = ".\runs\infer_loop",
  [string]$MineOut = ".\vision\data\mine_loop",
  [string]$RealRoot = ".\vision\data\real",
  [string]$SynthMetaTrain = ".\vision\data\synth\meta\train.jsonl",
  [string]$RealMetaTrain  = ".\vision\data\real\meta\train.jsonl",
  [string]$RealMetaVal    = ".\vision\data\real\meta\val.jsonl",
  [string]$SaveDir  = ".\vision\train\checkpoints",
  [int]$Epochs = 4,
  [int]$Batch = 512,
  [double]$LR = 0.0007,
  [string]$TrainMix = "0.7,0.3",
  [int]$Workers = 0,
  [string]$EvalOut = ".\runs\eval_loop",
  [int]$ValEvery = 10,
  [switch]$IncludeLow = $true,
  [switch]$PairOnly = $false,
  [switch]$SkipInfer = $false
)

$ErrorActionPreference = "Stop"

function Run-Step([string]$Title, [scriptblock]$Block) {
  Write-Host "`n=== $Title ===" -ForegroundColor Cyan
  & $Block
  if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    throw "Step failed: $Title (exit $LASTEXITCODE)"
  }
}

# 1) Inference (can be skipped to reuse an existing all_preds.csv)
if (-not $SkipInfer) {
  Run-Step "1) Batch inference -> all_preds.csv" {
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
  }
} else {
  Write-Host "Skipping inference (using existing $InferOut\all_preds.csv)" -ForegroundColor DarkYellow
}

# 2) Mine interesting digits (with optional --pair-only)
Run-Step "2) Mine hard examples for digits/pairs: $Digits" {
  $incFlag = $null; if ($IncludeLow) { $incFlag = "--include-low" }
  $pairFlag = $null; if ($PairOnly) { $pairFlag = "--pair-only" }
  & python ".\tools\mining\mine_digits_pack.py" `
    --preds "$InferOut\all_preds.csv" `
    --digits $Digits `
    --out $MineOut `
    --val-every $ValEvery `
    $incFlag `
    --wide-margin $WideMargin `
    $pairFlag
}

# 3) Label queue (fast)
Run-Step "3) Label queue (fast digit labeler)" {
  $queueDir = Join-Path $MineOut "queue"
  if (Test-Path $queueDir) {
    $count = (Get-ChildItem -Recurse -File $queueDir | Measure-Object).Count
  } else {
    $count = 0
  }
  if ($count -gt 0) {
    Write-Host "Queue has $count tiles at $queueDir" -ForegroundColor Yellow
    & python ".\tools\labeler\fast_digit_labeler.py" `
      --src $queueDir `
      --out $RealRoot `
      --val-every $ValEvery
  } else {
    Write-Host "No new tiles to label; skipping." -ForegroundColor DarkYellow
  }
}

# 4) Rebuild manifests + fine-tune briefly
Run-Step "4a) Build real manifests" {
  & python ".\tools\labeler\build_manifests.py" `
    --root $RealRoot `
    --out "$RealRoot\meta" `
    --source real
}

Run-Step "4b) Fine-tune (warm-start)" {
  & python -m vision.train.train_cells `
    --train-manifests "$SynthMetaTrain,$RealMetaTrain" `
    --train-mix $TrainMix `
    --val-manifest $RealMetaVal `
    --img 28 `
    --epochs $Epochs `
    --batch $Batch `
    --lr $LR `
    --warm-start $Model `
    --class-weights auto `
    --inner-crop $InnerCrop `
    --save-dir $SaveDir `
    --workers $Workers
}

# 5) Evaluate
Run-Step "5) Evaluate on gold val slice" {
  & python ".\vision\train\eval_confusion.py" `
    --model "$SaveDir\best.pt" `
    --val-manifest $RealMetaVal `
    --img 28 `
    --inner-crop $InnerCrop `
    --device cpu `
    --out $EvalOut
}

Write-Host "`nAll done ✅  Check:" -ForegroundColor Green
Write-Host " - Inference: $InferOut" -ForegroundColor Green
Write-Host " - Mined queue: $MineOut\queue" -ForegroundColor Green
Write-Host " - New checkpoint: $SaveDir\best.pt" -ForegroundColor Green
Write-Host " - Eval outputs: $EvalOut" -ForegroundColor Green
