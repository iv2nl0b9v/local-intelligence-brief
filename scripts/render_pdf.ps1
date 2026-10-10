param([string]$Chrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe')
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path $Chrome)) { throw "Chrome not found: $Chrome" }
$Html = [Uri]::new((Join-Path $Root 'index.html')).AbsoluteUri
$Pdf = Join-Path $Root 'local-ai-field-guide.pdf'
$Process = Start-Process -FilePath $Chrome -ArgumentList @('--headless=new', '--no-pdf-header-footer', "--print-to-pdf=`"$Pdf`"", "`"$Html`"") -WindowStyle Hidden -Wait -PassThru
if ($Process.ExitCode -ne 0) { throw "Chrome exited with code $($Process.ExitCode)" }
if (-not (Test-Path $Pdf)) { throw 'PDF was not created' }
Write-Output "Generated $Pdf. Inspect page count and clipping after layout changes."
