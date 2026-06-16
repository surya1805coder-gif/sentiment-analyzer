param(
    [int]$Epochs = 3,
    [int]$BatchSize = 8
)

Write-Host "=== Sentiment Analysis Training ===" -ForegroundColor Cyan
Write-Host "Epochs: $Epochs, Batch Size: $BatchSize" -ForegroundColor Yellow

python main.py --epochs $Epochs --batch-size $BatchSize

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
