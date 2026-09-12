# openLLM Agent launcher (PowerShell)
# Run: right-click "Run with PowerShell", or: powershell -File I:\openllm\launch.ps1
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  openLLM  -  AI experience system"
Write-Host "  Six bodies: IAI IAX ISA IOS ISN IKO"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Agent mode (/model to switch, /exit to quit)..."
Write-Host ""
wsl -e bash -lc "cd /mnt/i/openllm && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c 'from openllm.cli.main import main; main()'"
Write-Host ""
Write-Host "openLLM exited."
Read-Host "Press Enter to close"
