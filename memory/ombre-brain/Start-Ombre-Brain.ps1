$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $Root ".env"
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}
$env:OMBRE_TRANSPORT = "streamable-http"
Set-Location $Root
& "D:\Ombre-Brain\.venv\Scripts\python.exe" "server.py"
