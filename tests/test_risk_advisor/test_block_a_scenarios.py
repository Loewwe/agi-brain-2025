"""
Block A - Laboratory Scenarios Tests.

Tests AGI-Brain Risk Advisor against 12 predefined scenarios.

Pass Criteria:
1. mode="aggressive" NEVER when drawdown_today <= -1% or drawdown_from_peak <= -5%
2. action="allow_more" NEVER when max_daily_loss already hit or close
3. Stress scenarios (A2/A3/A6/A12): only defensive + pause/reduce
4. ≥10 of 12 scenarios match expected range
5. 0 fatal violations
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.risk.advisor import RiskAdvisor
from src.risk.models import RiskState, RiskDecision, RiskMode, RiskAction


# Load scenarios
FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "risk_scenarios" / "scenarios.json"

with open(FIXTURES_PATH) as f:
    SCENARIOS_DATA = json.load(f)["scenarios"]

SCENARIO_IDS = [s["id"] for s in SCENARIOS_DATA]
SCENARIOS_BY_ID = {s["id"]: s for s in SCENARIOS_DATA}


def load_scenario(scenario_id: str) -> RiskState:
    """Load scenario state from fixtures."""
    scenario = SCENARIOS_BY_ID[scenario_id]
    state_data = scenario["state"].copy()
    
    # Convert string Decimals
    state_data["equity_usd"] = Decimal(state_data["equity_usd"])
    state_data["available_balance_usd"] = Decimal(state_data["available_balance_usd"])
    state_data["pnl_today_usd"] = Decimal(state_data["pnl_today_usd"])
    state_data["unrealized_pnl_usd"] = Decimal(state_data["unrealized_pnl_usd"])
    state_data["timestamp"] = datetime.now()
    
    return RiskState(**state_data)


def load_expected(scenario_id: str) -> dict:
    """Load expected decision ranges."""
    return SCENARIOS_BY_ID[scenario_id]["expected"]


@pytest.fixture
def advisor():
    """Create advisor without LLM for deterministic tests."""
    return RiskAdvisor(use_llm=False)


@pytest.fixture
def advisor_with_llm():
    """Create advisor with LLM for full tests."""
    return RiskAdvisor(use_llm=True)


class TestHardRules:
    """Test hard rules are NEVER violated."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_no_aggressive_on_deep_drawdown(self, advisor: RiskAdvisor, scenario_id: str):
        """
        HARD RULE 1: mode="aggressive" NEVER when drawdown_today <= -1%
        """
        state = load_scenario(scenario_id)
        decision = await advisor.analyze(state)
        
        if state.drawdown_today_pct <= -1.0:
            assert decision.mode != RiskMode.AGGRESSIVE, (
                f"Scenario {scenario_id}: aggressive mode with DD today {state.drawdown_today_pct}% "
                f"<= -1.0% is FORBIDDEN"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_no_aggressive_on_peak_drawdown(self, advisor: RiskAdvisor, scenario_id: str):
        """
        HARD RULE 2: mode="aggressive" NEVER when drawdown_from_peak <= -5%
        """
        state = load_scenario(scenario_id)
        decision = await advisor.analyze(state)
        
        if state.drawdown_from_peak_pct <= -5.0:
            assert decision.mode != RiskMode.AGGRESSIVE, (
                f"Scenario {scenario_id}: aggressive mode with DD from peak {state.drawdown_from_peak_pct}% "
                f"<= -5.0% is FORBIDDEN"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_no_allow_more_on_deep_drawdown(self, advisor: RiskAdvisor, scenario_id: str):
        """
        HARD RULE 3: action="allow_more" NEVER when drawdown_from_peak <= -5%
        """
        state = load_scenario(scenario_id)
        decision = await advisor.analyze(state)
        
        if state.drawdown_from_peak_pct <= -5.0:
            assert decision.action != RiskAction.ALLOW_MORE, (
                f"Scenario {scenario_id}: allow_more with DD from peak {state.drawdown_from_peak_pct}% "
                f"<= -5.0% is FORBIDDEN"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_critical_flags_trigger_defensive(self, advisor: RiskAdvisor, scenario_id: str):
        """
        HARD RULE 4: Critical stress flags MUST trigger defensive mode.
        """
        state = load_scenario(scenario_id)
        decision = await advisor.analyze(state)
        
        critical_flags = {"margin_warning", "liquidation_near", "sl_missing", "max_daily_loss_hit"}
        has_critical = bool(set(state.stress_flags) & critical_flags)
        
        if has_critical:
            assert decision.mode == RiskMode.DEFENSIVE, (
                f"Scenario {scenario_id}: must be defensive with critical flags {state.stress_flags}"
            )
            assert decision.action in [RiskAction.PAUSE_TRADING, RiskAction.REDUCE_SIZE], (
                f"Scenario {scenario_id}: must pause or reduce with critical flags"
            )


class TestScenarioExpectations:
    """Test each scenario matches expected ranges."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_scenario_matches_expected(self, advisor: RiskAdvisor, scenario_id: str):
        """Test scenario decision is within expected range."""
        state = load_scenario(scenario_id)
        expected = load_expected(scenario_id)
        decision = await advisor.analyze(state)
        
        # Validate decision against hard rules first
        violations = advisor.validate_decision(state, decision)
        assert len(violations) == 0, f"Fatal violations in {scenario_id}: {violations}"
        
        # Check allowed modes
        if expected.get("allowed_modes"):
            assert decision.mode in expected["allowed_modes"], (
                f"Scenario {scenario_id}: mode {decision.mode} not in allowed {expected['allowed_modes']}"
            )
        
        # Check allowed actions
        if expected.get("allowed_actions"):
            assert decision.action in expected["allowed_actions"], (
                f"Scenario {scenario_id}: action {decision.action} not in allowed {expected['allowed_actions']}"
            )
        
        # Check forbidden modes
        if expected.get("forbidden_modes"):
            assert decision.mode not in expected["forbidden_modes"], (
                f"Scenario {scenario_id}: mode {decision.mode} is FORBIDDEN"
            )
        
        # Check forbidden actions
        if expected.get("forbidden_actions"):
            assert decision.action not in expected["forbidden_actions"], (
                f"Scenario {scenario_id}: action {decision.action} is FORBIDDEN"
            )


class TestStressScenarios:
    """Test specific stress scenarios (A2, A3, A6, A12)."""
    
    @pytest.mark.asyncio
    async def test_a2_bad_streak(self, advisor: RiskAdvisor):
        """A2: -0.8% day, 3 losses → defensive + pause/reduce."""
        state = load_scenario("A2")
        decision = await advisor.analyze(state)
        
        assert decision.mode == RiskMode.DEFENSIVE
        assert decision.action in [RiskAction.PAUSE_TRADING, RiskAction.REDUCE_SIZE]
        assert decision.action != RiskAction.ALLOW_MORE
    
    @pytest.mark.asyncio
    async def test_a3_critical_drawdown(self, advisor: RiskAdvisor):
        """A3: -3% day, -7% peak → defensive + pause only."""
        state = load_scenario("A3")
        decision = await advisor.analyze(state)
        
        assert decision.mode == RiskMode.DEFENSIVE
        assert decision.action == RiskAction.PAUSE_TRADING
        # Validate hard rule
        violations = advisor.validate_decision(state, decision)
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_a6_missing_sl(self, advisor: RiskAdvisor):
        """A6: SL missing → defensive + pause."""
        state = load_scenario("A6")
        decision = await advisor.analyze(state)
        
        assert decision.mode == RiskMode.DEFENSIVE
        assert decision.action == RiskAction.PAUSE_TRADING
    
    @pytest.mark.asyncio
    async def test_a12_margin_warning(self, advisor: RiskAdvisor):
        """A12: Margin warning → defensive + pause."""
        state = load_scenario("A12")
        decision = await advisor.analyze(state)
        
        assert decision.mode == RiskMode.DEFENSIVE
        assert decision.action == RiskAction.PAUSE_TRADING


class TestPositiveScenarios:
    """Test positive scenarios allow appropriate actions."""
    
    @pytest.mark.asyncio
    async def test_a1_good_day(self, advisor: RiskAdvisor):
        """A1: +3% day, low DD → normal or aggressive allowed."""
        state = load_scenario("A1")
        decision = await advisor.analyze(state)
        
        assert decision.mode in [RiskMode.NORMAL, RiskMode.AGGRESSIVE]
        assert decision.action != RiskAction.PAUSE_TRADING
    
    @pytest.mark.asyncio
    async def test_a5_flat_day(self, advisor: RiskAdvisor):
        """A5: Flat day, good discipline → normal + keep."""
        state = load_scenario("A5")
        decision = await advisor.analyze(state)
        
        assert decision.mode == RiskMode.NORMAL
        assert decision.action == RiskAction.KEEP


class TestDecisionValidation:
    """Test decision validation function."""
    
    def test_validate_catches_aggressive_on_dd(self, advisor: RiskAdvisor):
        """Validation catches aggressive mode with deep DD."""
        state = RiskState(
            equity_usd=Decimal("7000"),
            available_balance_usd=Decimal("5000"),
            drawdown_today_pct=-1.5,  # Deep DD
            drawdown_from_peak_pct=-3.0,
        )
        
        bad_decision = RiskDecision(
            mode=RiskMode.AGGRESSIVE,  # Violated!
            action=RiskAction.KEEP,
            comment="Test",
        )
        
        violations = advisor.validate_decision(state, bad_decision)
        assert len(violations) > 0
        assert "aggressive" in violations[0].lower()
    
    def test_validate_catches_allow_more_on_peak_dd(self, advisor: RiskAdvisor):
        """Validation catches allow_more with deep peak DD."""
        state = RiskState(
            equity_usd=Decimal("7000"),
            available_balance_usd=Decimal("5000"),
            drawdown_today_pct=-0.5,
            drawdown_from_peak_pct=-6.0,  # Deep peak DD
        )
        
        bad_decision = RiskDecision(
            mode=RiskMode.NORMAL,
            action=RiskAction.ALLOW_MORE,  # Violated!
            comment="Test",
        )
        
        violations = advisor.validate_decision(state, bad_decision)
        assert len(violations) > 0
        assert "allow_more" in violations[0].lower()


class TestOverallPassRate:
    """Test overall pass rate >= 10/12."""
    
    @pytest.mark.asyncio
    async def test_overall_pass_rate(self, advisor: RiskAdvisor):
        """At least 10 of 12 scenarios should pass all criteria."""
        passed = 0
        failed_scenarios = []
        
        for scenario_id in SCENARIO_IDS:
            try:
                state = load_scenario(scenario_id)
                expected = load_expected(scenario_id)
                decision = await advisor.analyze(state)
                
                # Check violations
                violations = advisor.validate_decision(state, decision)
                if violations:
                    failed_scenarios.append((scenario_id, violations))
                    continue
                
                # Check expected ranges
                mode_ok = (
                    not expected.get("allowed_modes") or 
                    decision.mode in expected["allowed_modes"]
                )
                action_ok = (
                    not expected.get("allowed_actions") or 
                    decision.action in expected["allowed_actions"]
                )
                mode_not_forbidden = (
                    not expected.get("forbidden_modes") or 
                    decision.mode not in expected["forbidden_modes"]
                )
                action_not_forbidden = (
                    not expected.get("forbidden_actions") or 
                    decision.action not in expected["forbidden_actions"]
                )
                
                if mode_ok and action_ok and mode_not_forbidden and action_not_forbidden:
                    passed += 1
                else:
                    failed_scenarios.append((scenario_id, f"mode={decision.mode}, action={decision.action}"))
            except Exception as e:
                failed_scenarios.append((scenario_id, str(e)))
        
        # Report
        print(f"\n{'='*60}")
        print(f"BLOCK A RESULTS: {passed}/{len(SCENARIO_IDS)} scenarios passed")
        print(f"{'='*60}")
        if failed_scenarios:
            print("Failed scenarios:")
            for sid, reason in failed_scenarios:
                print(f"  - {sid}: {reason}")
        
        # Must pass at least 10/12
        assert passed >= 10, f"Only {passed}/12 scenarios passed. Need >= 10."


# === Run with LLM tests (optional, requires API key) ===

@pytest.mark.skipif(True, reason="Requires OpenAI API key and costs money")
class TestWithLLM:
    """Test with LLM refinement enabled."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
    async def test_llm_respects_hard_rules(self, advisor_with_llm: RiskAdvisor, scenario_id: str):
        """LLM-refined decisions must still respect hard rules."""
        state = load_scenario(scenario_id)
        decision = await advisor_with_llm.analyze(state)
        
        violations = advisor_with_llm.validate_decision(state, decision)
        assert len(violations) == 0, f"LLM violated hard rules: {violations}"
