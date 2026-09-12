@echo off
chcp 65001 >nul
title openLLM Agent
echo.
echo  ============================================
echo    openLLM  -  AI experience system
echo    Six bodies: IAI IAX ISA IOS ISN IKO
echo  ============================================
echo.
echo  Starting Agent mode (/model to switch, /exit to quit)...
echo.
wsl -e bash -lc "cd /mnt/i/openllm && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c 'from openllm.cli.main import main; main()'"
echo.
echo  openLLM exited.
pause
