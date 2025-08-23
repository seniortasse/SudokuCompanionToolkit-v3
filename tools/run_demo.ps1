param(
  [string]$Image = "samples\sample_puzzle.jpg",
  [string]$Out   = "demo_export",
  [int]$MaxMoves = 5
)

$ErrorActionPreference = "Stop"

# 1) Ensure output dir exists
New-Item -ItemType Directory -Path $Out -Force > $null

# 2) Generate JSON directly from the CLI (UTF-8 no BOM)
python -m apps.cli.demo_cli_overlay `
  --image $Image `
  --out   $Out `
  --mode  demo `
  --max_moves $MaxMoves `
  --json "$Out\moves.json"

# 3) Build storyboard + animations
python -m apps.cli.storyboard_sheet --dir $Out --out "$Out\storyboard" --paper letter --cols 2 --json "$Out\moves.json"
python -m apps.cli.animate_gif      --dir $Out --out "$Out\moves.gif"
python -m apps.cli.animate_mp4      --dir $Out --out "$Out\moves.mp4"

Write-Host "Done. See:"
Write-Host " - $Out\moves.json"
Write-Host " - $Out\storyboard\storyboard.pdf"
Write-Host " - $Out\moves.gif"
Write-Host " - $Out\moves.mp4"