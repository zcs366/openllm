"""
监督矩阵配置加载器

子产方案：多对多互补监督（非一一对应）。
核心监督者固定，辅助监督者六轮轮值。

用法：
    from openllm.governance.supervision_matrix import SupervisionMatrix
    sm = SupervisionMatrix()
    supervisor = sm.get_supervisor("IAX", round_number=1)
"""
import yaml
from pathlib import Path
from typing import Optional


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "supervision_matrix.yaml"


class SupervisionMatrix:
    """六体监督矩阵——核心固定+辅助轮值"""

    def __init__(self, config_path: str = None):
        path = Path(config_path) if config_path else _CONFIG_PATH
        if path.exists():
            with open(path) as f:
                self._config = yaml.safe_load(f)
        else:
            # 默认配置（不依赖YAML文件）
            self._config = self._default_config()

        self._core = self._config.get("core_supervisors", {})
        self._rotating = self._config.get("rotating_supervisors", {})
        self._thresholds = self._config.get("health_thresholds", {
            "healthy": 0.7, "degraded": 0.5, "critical": 0.3,
        })

    def _default_config(self) -> dict:
        return {
            "core_supervisors": {
                "IAX": "ISN", "IAI": "IKO", "ISA": "IOS",
                "IOS": "IAI", "ISN": "IAX", "IKO": "ISN",
            },
            "rotating_supervisors": {},
            "health_thresholds": {"healthy": 0.7, "degraded": 0.5, "critical": 0.3},
        }

    def get_core_supervisor(self, body: str) -> Optional[str]:
        """获取某个体的核心监督者"""
        return self._core.get(body)

    def get_rotating_supervisor(self, body: str, round_number: int) -> Optional[str]:
        """获取某个体在指定轮次的辅助监督者"""
        round_key = f"round_{((round_number - 1) % 6) + 1}"
        round_config = self._rotating.get(round_key, {})
        return round_config.get(body)

    def get_supervisor(self, body: str, round_number: int = 1) -> dict:
        """获取某个体的完整监督信息"""
        core = self.get_core_supervisor(body)
        rotating = self.get_rotating_supervisor(body, round_number)
        return {
            "body": body,
            "core_supervisor": core,
            "rotating_supervisor": rotating,
            "all_supervisors": list(set(filter(None, [core, rotating]))),
        }

    def classify_health(self, score: float) -> str:
        """根据分数分类健康状态"""
        if score >= self._thresholds["healthy"]:
            return "healthy"
        elif score >= self._thresholds["degraded"]:
            return "degraded"
        else:
            return "critical"

    def get_all_bodies(self) -> list[str]:
        """获取所有六体名称"""
        return list(self._core.keys())

    def get_matrix_summary(self) -> dict:
        """获取监督矩阵摘要"""
        summary = {}
        for body in self._core:
            summary[body] = {
                "core": self._core[body],
                "rotating_rounds": {},
            }
            for r in range(1, 7):
                rot = self.get_rotating_supervisor(body, r)
                if rot:
                    summary[body]["rotating_rounds"][r] = rot
        return summary
