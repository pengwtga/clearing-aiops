"""清算对账归因服务层（归因 / 门禁 / 评测）。

分层说明：
- `scripts/` = 数据与对账引擎层（纯脚本、零第三方依赖）
- `app/`     = 归因服务层（LLM 归因、置信度门禁、降级、评测）

两层共用同一份差错字典 `scripts/taxonomy.py`（单一事实来源），
这里把它挂进 import 路径，避免字典被复制成两份后与实现漂移。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = ROOT / "scripts"

for _p in (str(ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
