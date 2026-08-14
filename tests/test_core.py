"""openLLM核心模块测试。"""
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock


# ── ExperimentEngine ──

class TestExperimentEngine:
    def setup_method(self):
        from openllm.tools.experiment_engine import ExperimentEngine
        self.tmpdir = tempfile.mkdtemp()
        self.engine = ExperimentEngine(storage_dir=Path(self.tmpdir))

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_add_hypothesis(self):
        from openllm.tools.experiment_engine import HypothesisStatus
        h = self.engine.add_hypothesis("h1", "claim", "pred")
        assert h.id == "h1"
        assert h.status == HypothesisStatus.UNVERIFIED

    def test_design_experiment(self):
        self.engine.add_hypothesis("h1", "claim", "pred")
        exp = self.engine.design_experiment("exp1", "h1", "method")
        assert exp.status == "designed"
        assert exp.hypothesis_id == "h1"

    def test_hypothesis_auto_testing(self):
        self.engine.add_hypothesis("h1", "claim", "pred")
        self.engine.design_experiment("exp1", "h1", "method")
        h = self.engine.get_hypothesis("h1")
        from openllm.tools.experiment_engine import HypothesisStatus
        assert h.status == HypothesisStatus.TESTING

    def test_record_result(self):
        self.engine.add_hypothesis("h1", "claim", "pred")
        self.engine.design_experiment("exp1", "h1", "method")
        r = self.engine.record_result("exp_001", True, conclusion="ok")
        assert r.success

    def test_proven_after_3(self):
        from openllm.tools.experiment_engine import HypothesisStatus
        self.engine.add_hypothesis("h1", "c", "p")
        self.engine.design_experiment("e", "h1", "m")
        for _ in range(3):
            self.engine.record_result("exp_001", True)
        assert self.engine.get_hypothesis("h1").status == HypothesisStatus.PROVEN

    def test_disproven_after_2(self):
        from openllm.tools.experiment_engine import HypothesisStatus
        self.engine.add_hypothesis("h1", "c", "p")
        self.engine.design_experiment("e", "h1", "m")
        for _ in range(2):
            self.engine.record_result("exp_001", False)
        assert self.engine.get_hypothesis("h1").status == HypothesisStatus.DISPROVEN

    def test_persistence(self):
        self.engine.add_hypothesis("h1", "c", "p")
        from openllm.tools.experiment_engine import ExperimentEngine
        engine2 = ExperimentEngine(storage_dir=Path(self.tmpdir))
        assert "h1" in engine2.hypotheses

    def test_summary(self):
        self.engine.add_hypothesis("h1", "c", "p")
        s = self.engine.summary()
        assert s["total_hypotheses"] == 1


# ── PaperKnowledge ──

class TestPaperKnowledge:
    def setup_method(self):
        from openllm.tools.paper_knowledge import PaperKnowledge
        self.tmpdir = tempfile.mkdtemp()
        self.pk = PaperKnowledge(storage_dir=Path(self.tmpdir))

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_add_paper(self):
        p = self.pk.add_paper("p1", "Test", tags=["ai"])
        assert p.id == "p1"
        assert p.tags == ["ai"]

    def test_search_by_tag(self):
        self.pk.add_paper("p1", "H2O", tags=["attention", "kv"])
        r = self.pk.search("attention")
        assert len(r) == 1
        assert r[0].id == "p1"

    def test_search_by_title(self):
        self.pk.add_paper("p1", "Machine Learning")
        r = self.pk.search("machine")
        assert len(r) == 1

    def test_add_concept(self):
        self.pk.add_concept("fuel", "info gain", papers=["p1"])
        assert "fuel" in self.pk.concepts

    def test_persistence(self):
        self.pk.add_paper("p1", "Test", tags=["x"])
        from openllm.tools.paper_knowledge import PaperKnowledge
        pk2 = PaperKnowledge(storage_dir=Path(self.tmpdir))
        assert "p1" in pk2.papers


# ── ResearchLoop ──

class TestResearchLoop:
    def setup_method(self):
        from openllm.tools.research_loop import ResearchLoop
        self.rl = ResearchLoop()

    def test_observe(self):
        obs = self.rl.observe("obs1")
        assert obs["text"] == "obs1"
        assert len(self.rl.state.observations) == 1

    def test_hypothesize(self):
        h = self.rl.hypothesize("claim", "pred")
        assert h.id.startswith("h_")
        assert self.rl.state.current_hypothesis_id == h.id

    def test_design_and_run(self):
        self.rl.hypothesize("c", "p")
        eid = self.rl.design_experiment("exp", "method")
        assert eid.startswith("exp_")
        r = self.rl.run_experiment(True, conclusion="ok")
        assert r["success"]

    def test_conclude(self):
        c = self.rl.conclude("finding")
        assert c["finding"] == "finding"

    def test_iterate(self):
        st = self.rl.iterate()
        assert st["iterations"] == 1

    def test_to_heartbeat_context(self):
        ctx = self.rl.to_heartbeat_context()
        assert ctx["research_mode"] is True


# ── Bridge ──

class TestHermesBridge:
    def test_bridge_message_json(self):
        from openllm.bridge.hermes_bridge import BridgeMessage
        msg = BridgeMessage(source="openllm", target="hermes", action="search")
        j = msg.to_json()
        msg2 = BridgeMessage.from_json(j)
        assert msg2.source == "openllm"
        assert msg2.action == "search"

    def test_signal_consumer(self):
        from openllm.bridge.hermes_bridge import SignalConsumer, BridgeMessage
        tmpdir = tempfile.mkdtemp()
        consumer = SignalConsumer(signal_dir=Path(tmpdir))
        received = []
        consumer.register("test", lambda p: received.append(p))
        # 写信号文件
        msg = BridgeMessage(source="openllm", target="hermes",
                           action="test", payload={"key": "val"})
        (Path(tmpdir) / f"{msg.message_id}.json").write_text(msg.to_json())
        # 消费
        results = consumer.consume()
        assert len(results) == 1
        assert received[0] == {"key": "val"}
        shutil.rmtree(tmpdir)


# ── BodyProtocol ──

class TestBodyProtocol:
    def test_registry(self):
        from openllm.core.body_protocol import BodyRegistry, IAIBody, IOSBody
        reg = BodyRegistry()
        reg.register(IAIBody())
        reg.register(IOSBody())
        assert reg.names() == ["IAI", "IOS"]
        assert reg.get("IAI").name == "IAI"

    def test_unified_heartbeat(self):
        from openllm.core.body_protocol import UnifiedHeartbeat
        hc = UnifiedHeartbeat(user_message="test")
        hc.log_phase("listen", "ok")
        assert len(hc.phase_log) == 1
        assert hc.phase_log[0]["phase"] == "listen"


# ── Protocol ──

class TestProtocol:
    def test_heartbeat_context_research_field(self):
        from openllm.core.protocol import HeartbeatContext
        hc = HeartbeatContext()
        assert hasattr(hc, "research")
        hc.research = {"phase": "observe"}
        assert hc.research == {"phase": "observe"}


# ── Agent ──

class TestAgent:
    def test_agent_has_research(self):
        from openllm.core.main_loop import Agent
        a = Agent(mode="silent")
        assert hasattr(a, "research")
        assert type(a.research).__name__ == "ResearchLoop"

    def test_agent_has_five_bodies(self):
        from openllm.core.main_loop import Agent
        a = Agent(mode="silent")
        assert a.isa is not None
        assert a.octopus is not None
        assert a.ios is not None
        assert a.isn is not None
        assert a.iko is not None
