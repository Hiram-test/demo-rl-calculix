"""Isolated shaft-grid discovery experiment package."""  # 声明本目录只承载圆轴标线与网格方向实验。
from __future__ import annotations  # 启用现代类型注解并保持包初始化兼容。

import os  # 调整当前进程的求解器搜索路径。
from pathlib import Path  # 定位随包提交的 CalculiX 输入规范化包装器。

_PACKAGE_DIR = Path(__file__).resolve().parent  # 定位当前隔离实验包目录。
_CCX_WRAPPER = _PACKAGE_DIR / "ccx"  # 定位与包一起提交的求解器包装入口。
if _CCX_WRAPPER.exists():  # 检查包装器文件已经随当前提交存在。
    _CCX_WRAPPER.chmod(0o755)  # 在临时 runner 检出目录中授予执行权限。
    os.environ["PATH"] = str(_PACKAGE_DIR) + os.pathsep + os.environ.get("PATH", "")  # 让后端优先发现包装器再调用系统真实 CalculiX。
