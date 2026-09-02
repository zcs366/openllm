# 确保tests能找到ilm目录的模块
import sys
import os

# ilm目录 = tests的父目录
ilm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ilm_dir not in sys.path:
    sys.path.insert(0, ilm_dir)
