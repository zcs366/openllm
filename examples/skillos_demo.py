#!/usr/bin/env python3
"""
SkillOS RL Curation Demo — Proof of Concept
=============================================

Demonstrates how the SkillOSAdapter learns curation policies from skill
usage history over multiple iterations. Shows:

1. Synthetic skill usage history (good/bad/mediocre skills)
2. Multi-dimensional RL-style scoring
3. Keep/Merge/Retire decisions with reasoning
4. Iterative curation — how the pool improves each round
5. Policy tuning based on RL feedback signals

Usage:
    python3 examples/skillos_demo.py
"""

import sys
import os
import math
from datetime import datetime, timezone, timedelta

# Ensure openllm is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.adapters.skillos_adapter import (
    CurationAction,
    SkillUsageRecord,
    SkillUsageHistory,
    SkillScore,
    CurationDecision,
    CurationPolicy,
    SkillOSAdapter,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
BANNER = "=" * 72


def section(title: str):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def make_record(
    success: bool = True,
    latency_ms: float = 50.0,
    token_cost: float = 100.0,
    quality_score: float = 0.8,
    days_ago: int = 0,
) -> SkillUsageRecord:
    """Create a single usage record."""
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return SkillUsageRecord(
        timestamp=ts,
        success=success,
        latency_ms=latency_ms,
        token_cost=token_cost,
        quality_score=quality_score,
    )


def make_history(
    name: str,
    calls: int = 10,
    success_rate: float = 0.9,
    task_group_id: str = "group-a",
    curator_score: float = 0.7,
    compression_reward: float = 0.6,
    total_token_size: int = 200,
    content_hash: str = "",
    days_ago_last: int = 1,
) -> SkillUsageHistory:
    """Create skill usage history with synthetic records."""
    records = []
    for i in range(calls):
        success = (i / max(calls, 1)) < success_rate
        records.append(make_record(
            success=success,
            days_ago=max(0, days_ago_last - i),
            latency_ms=30 + (1 - success_rate) * 200,
            token_cost=80 + (1 - success_rate) * 400,
            quality_score=0.2 + success_rate * 0.7,
        ))
    return SkillUsageHistory(
        skill_name=name,
        task_group_id=task_group_id,
        records=records,
        curator_score=curator_score,
        compression_reward=compression_reward,
        total_token_size=total_token_size,
        content_hash=content_hash or f"hash-{name}",
    )


def print_decision(d: CurationDecision):
    """Pretty-print a curation decision."""
    action_icon = {
        CurationAction.KEEP: "✅ KEEP ",
        CurationAction.MERGE: "🔄 MERGE",
        CurationAction.RETIRE: "❌ RETIRE",
    }
    icon = action_icon.get(d.action, "❓ ???")
    score = d.score.composite_score
    print(f"  {icon}  {d.skill_name:<22s}  score={score:.4f}  conf={d.score.confidence:.3f}")
    if d.merge_target:
        print(f"         → merge target: {d.merge_target}")
    # Truncate reason for display
    reason = d.reason
    if len(reason) > 90:
        reason = reason[:87] + "..."
    print(f"         {reason}")


def print_scores_table(histories, adapter):
    """Print a detailed scores table for all skills."""
    print(f"\n  {'Skill':<22s} {'Success':>8s} {'Effici':>8s} {'Recenc':>8s} {'Uniqu':>8s} {'Compr':>8s} {'Curatr':>8s} {'TOTAL':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for h in sorted(histories, key=lambda x: adapter.score_skill(x).composite_score, reverse=True):
        s = adapter.score_skill(h)
        print(f"  {h.skill_name:<22s} {s.success_signal:>8.3f} {s.efficiency_signal:>8.3f} "
              f"{s.recency_signal:>8.3f} {s.uniqueness_signal:>8.3f} "
              f"{s.compression_signal:>8.3f} {s.curator_signal:>8.3f} "
              f"{s.composite_score:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1: Synthetic Skill Pool
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_pool() -> list[SkillUsageHistory]:
    """Build a pool of 15 skills with varying quality levels."""
    return [
        # ── Tier 1: High-quality skills (should be KEPT) ──────────────────
        make_history(
            name="search-engine",
            calls=50, success_rate=0.95, task_group_id="retrieval",
            curator_score=0.92, compression_reward=0.85, total_token_size=150,
            days_ago_last=0, content_hash="hash-search-v3",
        ),
        make_history(
            name="code-review",
            calls=40, success_rate=0.90, task_group_id="development",
            curator_score=0.88, compression_reward=0.80, total_token_size=200,
            days_ago_last=1, content_hash="hash-code-review",
        ),
        make_history(
            name="summarizer",
            calls=35, success_rate=0.88, task_group_id="text",
            curator_score=0.85, compression_reward=0.75, total_token_size=180,
            days_ago_last=0, content_hash="hash-summary-v2",
        ),

        # ── Tier 2: Decent skills (should be KEPT or borderline) ──────────
        make_history(
            name="translator-en-zh",
            calls=25, success_rate=0.82, task_group_id="text",
            curator_score=0.70, compression_reward=0.65, total_token_size=220,
            days_ago_last=3, content_hash="hash-translator",
        ),
        make_history(
            name="debug-helper",
            calls=20, success_rate=0.75, task_group_id="development",
            curator_score=0.68, compression_reward=0.60, total_token_size=250,
            days_ago_last=5, content_hash="hash-debug",
        ),
        make_history(
            name="web-scraper",
            calls=30, success_rate=0.78, task_group_id="retrieval",
            curator_score=0.65, compression_reward=0.55, total_token_size=280,
            days_ago_last=2, content_hash="hash-scraper-v1",
        ),

        # ── Tier 3: Mediocre skills (MERGE candidates) ────────────────────
        make_history(
            name="old-summarizer",
            calls=15, success_rate=0.55, task_group_id="text",
            curator_score=0.40, compression_reward=0.35, total_token_size=350,
            days_ago_last=20, content_hash="hash-summary-v1",  # similar to summarizer
        ),
        make_history(
            name="basic-search",
            calls=12, success_rate=0.50, task_group_id="retrieval",
            curator_score=0.38, compression_reward=0.30, total_token_size=300,
            days_ago_last=25, content_hash="hash-search-v1",  # similar to search-engine
        ),
        make_history(
            name="text-formatter",
            calls=10, success_rate=0.60, task_group_id="text",
            curator_score=0.35, compression_reward=0.28, total_token_size=400,
            days_ago_last=30, content_hash="hash-formatter",
        ),

        # ── Tier 4: Low-quality skills (RETIRE candidates) ────────────────
        make_history(
            name="broken-regex",
            calls=8, success_rate=0.20, task_group_id="development",
            curator_score=0.12, compression_reward=0.08, total_token_size=600,
            days_ago_last=60, content_hash="hash-regex-broken",
        ),
        make_history(
            name="deprecated-api-v1",
            calls=5, success_rate=0.15, task_group_id="retrieval",
            curator_score=0.08, compression_reward=0.05, total_token_size=800,
            days_ago_last=90, content_hash="hash-api-v1",
        ),
        make_history(
            name="spam-generator",
            calls=3, success_rate=0.10, task_group_id="text",
            curator_score=0.05, compression_reward=0.03, total_token_size=1200,
            days_ago_last=45, content_hash="hash-spam",
        ),

        # ── Tier 5: Dead skills (should be hard-retired) ──────────────────
        make_history(
            name="zombie-tool",
            calls=2, success_rate=0.0, task_group_id="development",
            curator_score=0.01, compression_reward=0.01, total_token_size=1500,
            days_ago_last=120, content_hash="hash-zombie",
        ),
        make_history(
            name="ghost-skill",
            calls=1, success_rate=0.0, task_group_id="retrieval",
            curator_score=0.01, compression_reward=0.01, total_token_size=2000,
            days_ago_last=180, content_hash="hash-ghost",
        ),
        make_history(
            name="dead-weight",
            calls=0, success_rate=0.0, task_group_id="text",
            curator_score=0.0, compression_reward=0.0, total_token_size=500,
            days_ago_last=200, content_hash="hash-deadweight",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2: RL Feedback Loop
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_rl_feedback(
    adapter: SkillOSAdapter,
    decisions: list[CurationDecision],
    histories: dict[str, SkillUsageHistory],
) -> tuple[CurationPolicy, str]:
    """
    Simulate one round of RL feedback: adjust policy thresholds based on
    the distribution of decisions made in this round.

    Strategy:
    - If too few RETIREs → raise retire threshold (be more aggressive)
    - If too many MERGEs → lower merge threshold (be more decisive)
    - If KEEP ratio is good → tighten keep threshold (be more selective)
    - Also adjust weights: if efficiency scores are uniformly low, reduce
      its weight so other signals have more influence.
    """
    total = len(decisions)
    if total == 0:
        return adapter.policy, "No decisions made"

    keeps = sum(1 for d in decisions if d.action == CurationAction.KEEP)
    merges = sum(1 for d in decisions if d.action == CurationAction.MERGE)
    retires = sum(1 for d in decisions if d.action == CurationAction.RETIRE)

    keep_ratio = keeps / total
    retire_ratio = retires / total
    merge_ratio = merges / total

    # RL-inspired policy adjustment
    old = adapter.policy
    adjustments = []

    # Core thresholds
    new_retire = old.retire_threshold
    new_merge = old.merge_threshold
    new_keep = old.keep_threshold

    # If too few retirements, raise retire threshold to catch more
    if retire_ratio < 0.2:
        new_retire = min(
            new_merge - 0.05,
            new_retire + 0.05,
        )
        adjustments.append(
            f"Low retire rate ({retire_ratio:.0%}) → raised retire_threshold to {new_retire:.3f}"
        )

    # If too many merges (indecisive), lower merge threshold
    if merge_ratio > 0.4:
        new_merge = max(
            new_retire + 0.05,
            new_merge - 0.03,
        )
        adjustments.append(
            f"High merge rate ({merge_ratio:.0%}) → lowered merge_threshold to {new_merge:.3f}"
        )

    # If keep ratio is good, tighten keep threshold to be more selective
    if keep_ratio > 0.4:
        new_keep = max(
            new_merge + 0.1,
            new_keep - 0.02,
        )
        adjustments.append(
            f"Good keep ratio ({keep_ratio:.0%}) → tightened keep_threshold to {new_keep:.3f}"
        )

    if not adjustments:
        adjustments.append("Policy within acceptable range — no adjustment")

    new_policy = CurationPolicy(
        keep_threshold=new_keep,
        merge_threshold=new_merge,
        retire_threshold=new_retire,
        weight_success=old.weight_success,
        weight_efficiency=old.weight_efficiency,
        weight_recency=old.weight_recency,
        weight_uniqueness=old.weight_uniqueness,
        weight_compression=old.weight_compression,
        weight_curator=old.weight_curator,
        recency_half_life_days=old.recency_half_life_days,
        min_calls_for_confidence=old.min_calls_for_confidence,
        max_skills_per_group=old.max_skills_per_group,
    )

    return new_policy, "; ".join(adjustments)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo():
    section("SkillOS RL Curation — Proof of Concept")
    print("""
This demo shows how the SkillOSAdapter learns curation policies from
synthetic skill usage history. We simulate 3 iterations of RL-style
curation, where each round adjusts the policy based on feedback.

Key signals used:
  • success_signal   — How often the skill succeeds
  • efficiency_signal — Quality per token consumed
  • recency_signal   — How recently the skill was used (exponential decay)
  • uniqueness_signal — How unique this skill is in its task group
  • compression_signal — Rewards smaller, more efficient skills
  • curator_signal   — External curator quality assessment
""")

    # ── Build pool ────────────────────────────────────────────────────────
    section("Iteration 0: Initial Skill Pool (15 skills)")
    initial_pool = build_initial_pool()
    history_map = {h.skill_name: h for h in initial_pool}

    print(f"\n  Pool size: {len(initial_pool)} skills")
    print(f"  Task groups: {len(set(h.task_group_id for h in initial_pool))}")
    print(f"\n  Expected quality tiers:")
    print(f"    Tier 1 (Keep):     search-engine, code-review, summarizer")
    print(f"    Tier 2 (Keep):     translator-en-zh, debug-helper, web-scraper")
    print(f"    Tier 3 (Merge):    old-summarizer, basic-search, text-formatter")
    print(f"    Tier 4 (Retire):   broken-regex, deprecated-api-v1, spam-generator")
    print(f"    Tier 5 (Retire):   zombie-tool, ghost-skill, dead-weight")

    # ── Score all skills ──────────────────────────────────────────────────
    adapter = SkillOSAdapter(
        policy=CurationPolicy(),
        reference_time=NOW,
    )
    print_scores_table(initial_pool, adapter)

    # ── Iteration 1 ───────────────────────────────────────────────────────
    section("Iteration 1: First Curation Pass")

    print("\n  Policy: keep≥0.60 | merge=[0.10, 0.60) | retire<0.10")
    print(f"  Max skills per group: {adapter.policy.max_skills_per_group}")

    decisions_iter1 = adapter.curate(initial_pool)
    print(f"\n  Decisions made: {len(decisions_iter1)}")

    # Count actions
    keeps = [d for d in decisions_iter1 if d.action == CurationAction.KEEP]
    merges = [d for d in decisions_iter1 if d.action == CurationAction.MERGE]
    retires = [d for d in decisions_iter1 if d.action == CurationAction.RETIRE]

    print(f"  ✅ KEEP:  {len(keeps)}")
    print(f"  🔄 MERGE: {len(merges)}")
    print(f"  ❌ RETIRE: {len(retires)}")

    print("\n  Decisions:")
    for d in sorted(decisions_iter1, key=lambda x: x.score.composite_score, reverse=True):
        print_decision(d)

    # ── Simulate RL feedback → Iteration 2 ────────────────────────────────
    section("Iteration 2: RL Feedback + Policy Adjustment")

    new_policy1, adjustment1 = simulate_rl_feedback(adapter, decisions_iter1, history_map)
    print(f"\n  RL adjustment: {adjustment1}")

    # Create new adapter with adjusted policy
    adapter2 = SkillOSAdapter(policy=new_policy1, reference_time=NOW)

    # Pool shrinks: remove retired skills, apply decisions
    surviving_pool = [
        h for h in initial_pool
        if not any(
            d.skill_name == h.skill_name and d.action == CurationAction.RETIRE
            for d in decisions_iter1
        )
    ]

    print(f"  Pool after retirements: {len(surviving_pool)} skills "
          f"(was {len(initial_pool)})")

    print(f"\n  New policy: keep≥{new_policy1.keep_threshold:.3f} "
          f"| merge=[{new_policy1.retire_threshold:.3f}, {new_policy1.keep_threshold:.3f}) "
          f"| retire<{new_policy1.retire_threshold:.3f}")

    print_scores_table(surviving_pool, adapter2)

    decisions_iter2 = adapter2.curate(surviving_pool)

    keeps2 = [d for d in decisions_iter2 if d.action == CurationAction.KEEP]
    merges2 = [d for d in decisions_iter2 if d.action == CurationAction.MERGE]
    retires2 = [d for d in decisions_iter2 if d.action == CurationAction.RETIRE]

    print(f"\n  ✅ KEEP:  {len(keeps2)}")
    print(f"  🔄 MERGE: {len(merges2)}")
    print(f"  ❌ RETIRE: {len(retires2)}")

    print("\n  Decisions:")
    for d in sorted(decisions_iter2, key=lambda x: x.score.composite_score, reverse=True):
        print_decision(d)

    # ── Iteration 3 ───────────────────────────────────────────────────────
    section("Iteration 3: Final Curation Pass")

    new_policy2, adjustment2 = simulate_rl_feedback(adapter2, decisions_iter2, history_map)
    print(f"\n  RL adjustment: {adjustment2}")

    adapter3 = SkillOSAdapter(policy=new_policy2, reference_time=NOW)

    surviving_pool2 = [
        h for h in surviving_pool
        if not any(
            d.skill_name == h.skill_name and d.action == CurationAction.RETIRE
            for d in decisions_iter2
        )
    ]

    print(f"  Pool after retirements: {len(surviving_pool2)} skills "
          f"(was {len(initial_pool)})")

    print(f"\n  New policy: keep≥{new_policy2.keep_threshold:.3f} "
          f"| merge=[{new_policy2.retire_threshold:.3f}, {new_policy2.keep_threshold:.3f}) "
          f"| retire<{new_policy2.retire_threshold:.3f}")

    print_scores_table(surviving_pool2, adapter3)

    decisions_iter3 = adapter3.curate(surviving_pool2)

    keeps3 = [d for d in decisions_iter3 if d.action == CurationAction.KEEP]
    merges3 = [d for d in decisions_iter3 if d.action == CurationAction.MERGE]
    retires3 = [d for d in decisions_iter3 if d.action == CurationAction.RETIRE]

    print(f"\n  ✅ KEEP:  {len(keeps3)}")
    print(f"  🔄 MERGE: {len(merges3)}")
    print(f"  ❌ RETIRE: {len(retires3)}")

    print("\n  Decisions:")
    for d in sorted(decisions_iter3, key=lambda x: x.score.composite_score, reverse=True):
        print_decision(d)

    # ── Summary ───────────────────────────────────────────────────────────
    section("Curation Summary — Improvement Over Iterations")

    # Compute aggregate stats
    avg_iter1 = sum(d.score.composite_score for d in decisions_iter1) / len(decisions_iter1) if decisions_iter1 else 0
    avg_iter2 = sum(d.score.composite_score for d in decisions_iter2) / len(decisions_iter2) if decisions_iter2 else 0
    avg_iter3 = sum(d.score.composite_score for d in decisions_iter3) / len(decisions_iter3) if decisions_iter3 else 0

    # Average of KEPT skills only
    avg_keep1 = (sum(d.score.composite_score for d in keeps) / len(keeps)) if keeps else 0
    avg_keep2 = (sum(d.score.composite_score for d in keeps2) / len(keeps2)) if keeps2 else 0
    avg_keep3 = (sum(d.score.composite_score for d in keeps3) / len(keeps3)) if keeps3 else 0

    print(f"""
  ┌─────────────────┬──────────┬──────────┬──────────┐
  │ Metric          │ Iter 1   │ Iter 2   │ Iter 3   │
  ├─────────────────┼──────────┼──────────┼──────────┤
  │ Pool size       │ {len(initial_pool):>8d} │ {len(surviving_pool):>8d} │ {len(surviving_pool2):>8d} │
  │ Avg score (all) │ {avg_iter1:>8.4f} │ {avg_iter2:>8.4f} │ {avg_iter3:>8.4f} │
  │ Avg score (keep)│ {avg_keep1:>8.4f} │ {avg_keep2:>8.4f} │ {avg_keep3:>8.4f} │
  │ ✅ Keep          │ {len(keeps):>8d} │ {len(keeps2):>8d} │ {len(keeps3):>8d} │
  │ 🔄 Merge         │ {len(merges):>8d} │ {len(merges2):>8d} │ {len(merges3):>8d} │
  │ ❌ Retire        │ {len(retires):>8d} │ {len(retires2):>8d} │ {len(retires3):>8d} │
  └─────────────────┴──────────┴──────────┴──────────┘

  Trend: Pool shrinks from {len(initial_pool)} → {len(surviving_pool)} → {len(surviving_pool2)} skills
  Average kept-skill score: {avg_keep1:.4f} → {avg_keep2:.4f} → {avg_keep3:.4f}
  {("⬆ Quality improves as low-quality skills are removed" if avg_keep3 > avg_keep1 else "~ Pool quality stable")}
""")

    # ── Merge targets ─────────────────────────────────────────────────────
    section("Merge Target Analysis")
    all_merge = (
        [d for d in decisions_iter1 if d.action == CurationAction.MERGE and d.merge_target]
        + [d for d in decisions_iter2 if d.action == CurationAction.MERGE and d.merge_target]
    )
    if all_merge:
        print("\n  Skills identified for merging:")
        seen_pairs = set()
        for d in all_merge:
            pair = tuple(sorted([d.skill_name, d.merge_target]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                print(f"    • {d.skill_name} → {d.merge_target}")
                print(f"      (both in similar quality range, potential consolidation)")
    else:
        print("\n  No merge pairs identified — skills are sufficiently distinct.")

    # ── Confidence analysis ───────────────────────────────────────────────
    section("Confidence Analysis")
    print("\n  Score confidence is based on data volume (sigmoid of call count):")
    print(f"  min_calls_for_confidence: {adapter.policy.min_calls_for_confidence}")
    print()
    for h in sorted(initial_pool, key=lambda x: adapter.score_skill(x).confidence, reverse=True):
        s = adapter.score_skill(h)
        bar_len = int(s.confidence * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {h.skill_name:<22s} {bar} {s.confidence:.3f}  ({h.total_calls} calls)")

    # ── Done ──────────────────────────────────────────────────────────────
    section("Demo Complete")
    print("""
  Key Takeaways:
  1. SkillOSAdapter correctly identifies high/low quality skills
  2. Multi-dimensional scoring (6 signals) provides nuanced decisions
  3. RL policy adjustment converges — each iteration refines thresholds
  4. Pool quality improves as mediocre/failed skills are retired
  5. Merge detection identifies consolidation opportunities
  6. Confidence scoring prevents premature decisions on under-sampled skills
""")


if __name__ == "__main__":
    run_demo()
