$files = @(
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json"
)

$model_dir = "D:\eOCR\trocr-small"
New-Item -ItemType Directory -Force -Path $model_dir

foreach ($file in $files) {
    $url = "https://huggingface.co/microsoft/trocr-small-printed/resolve/main/$file"
    $out_path = "$model_dir\$file"
    
    Write-Host "Downloading $file..."
    $success = $false
    while (-not $success) {
        $p = Start-Process -FilePath "curl.exe" -ArgumentList "-C - -L -o `"$out_path`" `"$url`"" -Wait -PassThru
        if ($p.ExitCode -eq 0) {
            $success = $true
            Write-Host "Successfully downloaded $file!"
        } else {
            Write-Host "Connection dropped for $file! Resuming download from last byte..."
            Start-Sleep -Seconds 2
        }
    }
}
Write-Host "All files downloaded successfully to $model_dir!"
