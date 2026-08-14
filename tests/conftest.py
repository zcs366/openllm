"""pytest配置。"""
import sys
from pathlib import Path

# 确保openllm包可导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
