# openLLM Agent 启动脚本（PowerShell）
# 用法：右键"使用 PowerShell 运行"，或 powershell -File 启动OpenLLM.ps1
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  openLLM · AI 经验积累系统"
Write-Host "  六体架构：IAI IAX ISA IOS ISN IKO"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "正在启动 Agent 模式（/model 切换模型，/exit 退出）..."
Write-Host ""
wsl -e bash -lc "cd /mnt/i/openllm && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openllm.cli.main"
Write-Host ""
Write-Host "openLLM 已退出。"
Read-Host "按回车关闭"
