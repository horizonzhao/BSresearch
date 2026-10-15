"""pytest 配置：将 src 加入导入路径，并强制无 GUI 绘图后端。"""

import os

# 避免 Windows/miniconda 下 Tk 缺失导致 matplotlib 测试失败
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
