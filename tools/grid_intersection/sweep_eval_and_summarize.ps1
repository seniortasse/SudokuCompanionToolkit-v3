# --- Paths ----------------------------------------------------
$CKPT = "runs/grid_intersection/L1_easy/best_mje.pt"
$VAL  = "datasets/grids/synth_L1_easy/manifests/val_synth.jsonl"
$BASE = "runs/grid_intersection/L1_easy_eval_sweep_v2"

# --- Decode grids ---------------------------------------------
$Modes = @("quadfit","softargmax")
$Temps = @("0.3","0.5","0.7")        # used only for softargmax; ignored by quadfit
$TJs   = @("0.70","0.65","0.60")
$Confs = @("0.05","0.04","0.03")
$TopKs = @("150","180","220")
$IMG_SIZES = @("768")                # add "384" for device parity if you want

# --- CPU only (safe) -------------------------------------------
$DEVICE = "cpu"

# --- Make base outdir -----------------------------------------
New-Item -ItemType Directory -Force -Path $BASE | Out-Null

# --- Sweep -----------------------------------------------------
foreach ($img in $IMG_SIZES) {
  foreach ($mode in $Modes) {
    $temps = $mode -eq "softargmax" ? $Temps : @("NA")
    foreach ($temp in $temps) {
      foreach ($tj in $TJs) {
        foreach ($cf in $Confs) {
          foreach ($tk in $TopKs) {

            $tag = "mode-$mode`_temp-$temp`_tj-$tj`_conf-$cf`_topk-$tk`_img-$img"
            $OD  = Join-Path $BASE $tag
            New-Item -ItemType Directory -Force -Path $OD | Out-Null

            Write-Host ">>> Running $tag" -ForegroundColor Cyan
            python -m python.vision.train.grid_intersection.train `
              --eval_only `
              --resume $CKPT `
              --val_manifest $VAL `
              --train_manifest $VAL `
              --outdir $OD `
              --image_size $img `
              --subpixel $mode `
              --softargmax_temp ($temp -ne "NA" ? $temp : "0.5") `
              --eval_epoch 15 `
              --log_every 0 `
              --tj $tj `
              --j_conf $cf `
              --j_topk $tk `
              --device $DEVICE `
              --base_ch 24
          }
        }
      }
    }
  }
}

Write-Host "`n>>> Sweep finished. Aggregating..." -ForegroundColor Yellow

# --- Aggregate with the helper --------------------------------
# NOTE: the helper script path below assumes you downloaded/kept it at /mnt/data.
# If you moved it into your repo (e.g., tools/grid_intersection), update the path accordingly.
$Summarizer = "/mnt/data/summarize_eval_reports.py"
python $Summarizer --root $BASE --out $BASE

# --- Print top 5 by our composite sort (J_MJE asc, AP2 desc, LE8 desc) ---
$CSV = Join-Path $BASE "summary.csv"
if (Test-Path $CSV) {
  Write-Host "`n>>> Top 5 configs (best first)" -ForegroundColor Green
  Import-Csv $CSV | Sort-Object `
    @{Expression='J_MJE'; Ascending=$true}, `
    @{Expression='AP2';   Ascending=$false}, `
    @{Expression='LE8';   Ascending=$false} `
    | Select-Object -First 5 `
    | Format-Table -AutoSize

  # Optional: open the CSV in your default app
  try { Start-Process $CSV } catch {}
} else {
  Write-Host "summary.csv not found at $CSV" -ForegroundColor Red
}