@echo off
chcp 65001 >nul
title openLLM Agent
echo.
echo  ============================================
echo    openLLM  ·  AI 经验积累系统
echo    六体架构：IAI · IAX · ISA · IOS · ISN · IKO
echo  ============================================
echo.
echo  正在启动 Agent 模式（交互对话）...
echo  命令：/help 查看帮助，/exit 退出，直接打字对话
echo.
wsl -e bash -lc "cd /mnt/i/openllm && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openllm.cli.main"
echo.
echo  openLLM 已退出。
pause
