"""配置 —— 手写 .env 解析，不引 python-dotenv（保持零第三方依赖）

优先级：环境变量 > .env 文件 > 默认值。环境变量优先是为了让 CI / 容器能直接覆盖。
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    """只支持 KEY=VALUE 与 # 注释，够用且不会引入依赖"""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_DOTENV = _read_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key) or _DOTENV.get(key) or default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True)
class Settings:
    # ---- 归因门禁（核心参数，面试必问"为什么是 0.85"）
    gate_threshold: float = 0.85
    # ---- LLM 调用
    temperature: float = 0.1          # 归因是判别任务，压低随机性
    timeout: int = 60
    max_retries: int = 1
    # 全局最小请求间隔（毫秒）：免费档通常有 RPM 限制，靠它把并发压到限额以下
    min_interval_ms: int = 0

    @property
    def providers(self) -> list[ProviderConfig]:
        """按配置构建容灾链，缺 Key 的 Provider 自动跳过"""
        chain: list[ProviderConfig] = []
        primary_key = _get("RECON_LLM_API_KEY")
        if primary_key:
            chain.append(ProviderConfig(
                "primary",
                _get("RECON_LLM_BASE_URL", "https://api.deepseek.com"),
                _get("RECON_LLM_MODEL", "deepseek-chat"),
                primary_key,
            ))
        zhipu_key = _get("RECON_ZHIPU_API_KEY")
        if zhipu_key:
            chain.append(ProviderConfig(
                "zhipu",
                _get("RECON_ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
                _get("RECON_ZHIPU_MODEL", "glm-4.5-flash"),
                zhipu_key,
            ))
        silicon_key = _get("RECON_SILICONFLOW_API_KEY")
        if silicon_key:
            chain.append(ProviderConfig(
                "siliconflow",
                _get("RECON_SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
                _get("RECON_SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                silicon_key,
            ))
        return chain


def load_settings() -> Settings:
    return Settings(
        gate_threshold=_get_float("RECON_GATE_THRESHOLD", 0.85),
        temperature=_get_float("RECON_LLM_TEMPERATURE", 0.1),
        timeout=_get_int("RECON_LLM_TIMEOUT", 60),
        max_retries=_get_int("RECON_LLM_MAX_RETRIES", 1),
        min_interval_ms=_get_int("RECON_MIN_INTERVAL_MS", 0),
    )
