"""IKO 去手钉子（决策记录动作6）——测/评/改三分离。

IKO 是**镜子**：能评测（should_rollback）、能提议（propose_*），
**不能改状态**。执行只接受 IOS 的授权；未授权必须拒绝并吼一声。
"""
import logging

import pytest

from openllm.iko.feedback_collector import FeedbackSignal
from openllm.iko.lambda_calibrator import (LAMBDA_MAX, LAMBDA_MIN,
                                           LambdaCalibrator)


@pytest.fixture()
def cal(tmp_path):
    return LambdaCalibrator(storage_path=tmp_path / "lambda.json")


def _arm_rollback(cal):
    """把 λ 压到触发线以下。"""
    for _ in range(20):
        cal.update(FeedbackSignal.REJECTED, 0.9)
    assert cal.should_rollback() is True


# ── 手已经拆掉 ──

def test_unguarded_mutator_is_gone(cal):
    assert not hasattr(cal, "trigger_transparency_rollback"), \
        "镜子不许留后门：无授权的改写口必须不存在"


# ── 提议不动手 ──

def test_propose_does_not_touch_lambda(cal):
    _arm_rollback(cal)
    before = cal.get_current_lambda()
    p = cal.propose_transparency_rollback("general")
    assert cal.get_current_lambda() == before, "提议阶段 λ 一个字节都不许动"
    assert p["proposal"] == "transparency_rollback"
    assert p["domain"] == "general"
    assert p["target_lambda"] >= 0.6
    assert p["evidence"]["should_rollback"] is True
    assert p["reason"], "提议书必须说明理由"


# ── 未授权必须拒绝且吼一声 ──

@pytest.mark.parametrize("who", ["ISN", "IKO", "", "ios", "user"])
def test_unauthorized_apply_is_refused_and_shouts(cal, caplog, who):
    _arm_rollback(cal)
    before = cal.get_current_lambda()
    p = cal.propose_transparency_rollback("general")
    with caplog.at_level(logging.ERROR):
        ok = cal.apply_transparency_rollback(p, authorized_by=who)
    assert ok is False
    assert cal.get_current_lambda() == before, "未授权却改了状态＝镜子又长手了"
    assert any("拒绝执行回滚" in r.message for r in caplog.records), "拒绝必须吼一声"


def test_malformed_proposal_is_refused(cal, caplog):
    before = cal.get_current_lambda()
    with caplog.at_level(logging.ERROR):
        assert cal.apply_transparency_rollback({"proposal": "别的"}, authorized_by="IOS") is False
        assert cal.apply_transparency_rollback("不是字典", authorized_by="IOS") is False
    assert cal.get_current_lambda() == before


# ── IOS 授权才执行 ──

def test_ios_authorized_apply_works_and_stays_in_bounds(cal):
    _arm_rollback(cal)
    p = cal.propose_transparency_rollback("general")
    assert cal.apply_transparency_rollback(p, authorized_by="IOS") is True
    lam = cal.get_current_lambda()
    assert LAMBDA_MIN <= lam <= LAMBDA_MAX
    assert lam >= 0.6
    assert cal.should_rollback() is False, "执行后应重置恶化标记"


def test_authorized_apply_never_lowers_lambda(cal):
    """回滚的方向是「更透明」，不许把 λ 拉低。"""
    for _ in range(2):
        cal.update(FeedbackSignal.ACCEPTED, 0.9)
    before = cal.get_current_lambda()
    p = cal.propose_transparency_rollback("general")
    cal.apply_transparency_rollback(p, authorized_by="IOS")
    assert cal.get_current_lambda() >= before
