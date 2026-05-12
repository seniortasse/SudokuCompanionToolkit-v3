# resize_banner_models.ps1
# Resizes banner model images to exactly 970 x 600 px.
# Run from repo root.

Add-Type -AssemblyName System.Drawing

$TargetWidth = 970
$TargetHeight = 600

$Files = @(
    "assets\backgrounds\B02_Banner_2_Ready_To_Resize.png"
)

foreach ($RelativePath in $Files) {
    $InputPath = Join-Path (Get-Location) $RelativePath

    if (-not (Test-Path $InputPath)) {
        Write-Host "Missing file: $RelativePath" -ForegroundColor Red
        continue
    }

    $Directory = Split-Path $InputPath -Parent
    $BaseName = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
    $OutputPath = Join-Path $Directory "$BaseName`_970x600.png"

    Write-Host "Resizing $RelativePath -> $OutputPath"

    $SourceImage = [System.Drawing.Image]::FromFile($InputPath)

    try {
        $Bitmap = New-Object System.Drawing.Bitmap $TargetWidth, $TargetHeight
        $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)

        try {
            $Graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
            $Graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $Graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
            $Graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

            $DestinationRect = New-Object System.Drawing.Rectangle 0, 0, $TargetWidth, $TargetHeight
            $Graphics.DrawImage($SourceImage, $DestinationRect)

            $Bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally {
            $Graphics.Dispose()
            $Bitmap.Dispose()
        }
    }
    finally {
        $SourceImage.Dispose()
    }
}

Write-Host "Done. Resized banners were saved next to the originals with suffix _970x600.png." -ForegroundColor Green