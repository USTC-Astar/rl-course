"""第 20 课行为克隆协变量偏移实验的正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.imitation import run_imitation_demo


def test_bc_matches_expert_on_covered_starts() -> None:
    result = run_imitation_demo()
    # 分布内：BC 与专家接近（本网格宽容，两者都应大多成功）。
    assert result["bc_success_covered"] >= 0.85
    assert result["expert_success_covered"] >= 0.9


def test_bc_collapses_off_distribution() -> None:
    result = run_imitation_demo()
    # 分布外：专家照常工作，BC 因没有数据而显著崩溃——协变量偏移真实存在。
    assert result["expert_success_uncovered"] - result["bc_success_uncovered"] >= 0.3
    assert result["uncovered_starts"] > 0
