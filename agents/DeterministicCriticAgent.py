# agents/DeterministicCriticAgent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from formats.data_formats import PipelineState, Episode, ExpertIntent
from utils.PromptLogger import PromptLogger
from typing import Dict, List, Optional

from agents.BaseLLMClient import BaseLLMClient
import json
import re

class DeterministicCriticAgent:
    """
    Universal critic agent.
    Parses ANY expert comment → expected peak change → checks if achieved.
    """
    
    def __init__(
        self,
        llm,
        max_retries=3,
        enable_logging: bool = True,
        log_format: str = "json",
        position_threshold: float = 0.0,
        height_threshold_relative: float = 0.0,
        position_tolerance: float = 5.0,
        height_tolerance_relative: float = 0.05,
    ):
        self.llm_client = llm if isinstance(llm, BaseLLMClient) else None
        self.llm = llm
        self.max_retries = max_retries
        self.history: List[Episode] = []
        self.task_config: Dict = {}
        
        self.position_threshold = position_threshold
        self.height_threshold_relative = height_threshold_relative
        
        self.enable_logging = enable_logging
        self.logger = PromptLogger() if (enable_logging and log_format == "json") else None
        
        self.intent_parser = PydanticOutputParser(pydantic_object=ExpertIntent)
        self.intent_prompt = self._create_intent_prompt()
        self.position_tolerance = position_tolerance
        self.height_tolerance_relative = height_tolerance_relative
    
    def _create_intent_prompt(self) -> ChatPromptTemplate:
        """Universal prompt: extract expected peak change from ANY comment."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an epidemiologist. Analyze the expert comment and determine the EXPECTED CHANGE in the INFECTED (I) PEAK.

The infected peak has exactly two properties:
1. position: when the peak occurs → "later", "earlier", or "unchanged"
2. height: maximum number of infected → "higher", "lower", or "unchanged"

**YOUR TASK:**
For ANY expert comment, infer how the peak should change. If the comment implies no change to a property, set it to "unchanged".

**INFERENCE RULES:**
- Comments about restrictive measures (mask, lockdown, distancing) → peak LATER and LOWER
- Comments about relaxing measures (reopen, lift) → peak EARLIER and HIGHER
- Comments about vaccination → peak LOWER, position UNCHANGED
- Comments about new variants (more transmissible) → peak EARLIER and HIGHER
- Comments about prolonged/dragging epidemic → peak LATER, height UNCHANGED
- Comments about mortality/treatment → BOTH UNCHANGED (doesn't affect I-peak)
- If comment explicitly mentions direction, use that

**OUTPUT FORMAT:**
{format_instructions}

Return ONLY valid JSON."""),
            ("human", "Expert comment: {expert_comment}")
        ])
        
        return template.partial(format_instructions=self.intent_parser.get_format_instructions())
    
    # ========== УНИВЕРСАЛЬНЫЕ ПРОВЕРКИ ==========
    
    def _check_position(self, baseline: float, new: float, expected: str) -> dict:
        """
        Check if peak position changed as expected.
        expected ∈ {"later", "earlier", "unchanged"}
        """
        change = new - baseline
        is_significant = abs(change) >= self.position_threshold if self.position_threshold > 0 else abs(change) > 1e-6
        
        if expected == "later":
            ok = change > 0 and is_significant
            desc = f"moved later: {baseline:.1f} → {new:.1f}" if ok else f"expected later, but moved {new:.1f}"
        elif expected == "earlier":
            ok = change < 0 and is_significant
            desc = f"moved earlier: {baseline:.1f} → {new:.1f}" if ok else f"expected earlier, but moved {new:.1f}"
        else:  # unchanged
            # ✅ Толерантность: ±position_tolerance дней считается "unchanged"
            within_tolerance = abs(change) <= self.position_tolerance
            ok = within_tolerance
            if within_tolerance:
                desc = f"remained stable: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} within ±{self.position_tolerance})"
            else:
                desc = f"changed unexpectedly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} > ±{self.position_tolerance})"
        
        return {"ok": ok, "description": desc, "change": change, "is_significant": is_significant}
    
    def _check_height(self, baseline: float, new: float, expected: str) -> dict:
        """
        Check if peak height changed as expected.
        expected ∈ {"higher", "lower", "unchanged"}
        """
        change = new - baseline
        relative_change = abs(change) / baseline if baseline > 0 else 0
        is_significant = relative_change >= self.height_threshold_relative if self.height_threshold_relative > 0 else abs(change) > 1e-6
        
        if expected == "higher":
            ok = change > 0 and is_significant
            desc = f"increased: {baseline:.0f} → {new:.0f}" if ok else f"expected higher, but got {new:.0f}"
        elif expected == "lower":
            ok = change < 0 and is_significant
            desc = f"decreased: {baseline:.0f} → {new:.0f}" if ok else f"expected lower, but got {new:.0f}"
        else:  # unchanged
            # ✅ Толерантность: ±height_tolerance_relative считается "unchanged"
            within_tolerance = relative_change <= self.height_tolerance_relative
            ok = within_tolerance
            if within_tolerance:
                desc = f"remained stable: {baseline:.0f} → {new:.0f} ({relative_change*100:.1f}% within ±{self.height_tolerance_relative*100:.0f}%)"
            else:
                desc = f"changed unexpectedly: {baseline:.0f} → {new:.0f} ({relative_change*100:.1f}% > ±{self.height_tolerance_relative*100:.0f}%)"
        
        return {"ok": ok, "description": desc, "change": change, "is_significant": is_significant}
    
    def _make_decision(self, pos_check: dict, height_check: dict) -> dict:
        """
        Simple decision: accept ONLY if ALL expectations are met.
        """
        if pos_check["ok"] and height_check["ok"]:
            decision = "accept"
            reasoning = f"Position: {pos_check['description']}. Height: {height_check['description']}"
        else:
            decision = "reject"
            reasons = []
            if not pos_check["ok"]:
                reasons.append(f"Position: {pos_check['description']}")
            if not height_check["ok"]:
                reasons.append(f"Height: {height_check['description']}")
            reasoning = "; ".join(reasons)
        
        return {"decision": decision, "reasoning": reasoning}
    
    # ========== ПАРСИНГ ==========
    
    def _parse_intent(self, comment: str) -> dict:
        """Parse expert intent with LLM, fallback to keyword matching."""
        if not comment:
            return {"position_expected": "unchanged", "height_expected": "unchanged"}
        
        try:
            chain = self.intent_prompt | self.llm
            result = chain.invoke({"expert_comment": comment})
            text = result.content if hasattr(result, 'content') else str(result)
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "position_expected": parsed.get("position_direction", "unchanged"),
                    "height_expected": parsed.get("height_direction", "unchanged")
                }
            return json.loads(text)
            
        except Exception as e:
            print(f"   ⚠️ LLM failed, using fallback: {e}")
            return self._fallback_parse(comment)
    
    def _fallback_parse(self, comment: str) -> dict:
        """Keyword-based fallback."""
        low = comment.lower()
        
        # Position expectations
        if any(w in low for w in ["later", "delay", "prolong", "extend"]):
            pos_exp = "later"
        elif any(w in low for w in ["earlier", "sooner", "accelerate"]):
            pos_exp = "earlier"
        else:
            pos_exp = "unchanged"
        
        # Height expectations
        if any(w in low for w in ["higher", "increase", "surge", "more cases"]):
            height_exp = "higher"
        elif any(w in low for w in ["lower", "decrease", "reduce", "fewer", "flatten"]):
            height_exp = "lower"
        else:
            height_exp = "unchanged"
        
        # Корректировка для мер
        mitigation = ["mask", "lockdown", "quarantine", "distance", "restrict"]
        if any(w in low for w in mitigation):
            pos_exp = "later"
            height_exp = "lower"
        
        return {"position_expected": pos_exp, "height_expected": height_exp}
    
    # ========== ОСНОВНЫЕ МЕТОДЫ ==========
    
    def set_task_config(self, config: Dict):
        self.task_config = config
        print(f"✅ Critic configured")
    
    def add_to_history(self, episode: Episode):
        self.history.append(episode)
    
    def critique(
        self,
        baseline_episode: Episode,
        new_params: Dict,
        new_results: Dict,
        expert_comment: str,
        position_expected: str = None,    
        height_expected: str = None       
    ) -> Episode:
        print("=" * 60)
        print("🔍 CRITIC AGENT")
        print("=" * 60)
        
        baseline_peak = baseline_episode.peak_position or 0
        baseline_height = baseline_episode.peak_height or 0
        new_peak = new_results.get('peak_position', 0)
        new_height = new_results.get('peak_height', 0)
        
        print(f"📊 Baseline: peak={baseline_peak:.1f}, height={baseline_height:.0f}")
        print(f"📊 New:      peak={new_peak:.1f}, height={new_height:.0f}")
        print(f"💬 Expert:   {expert_comment}")
        
        # ✅ Используем переданные ожидания, если есть; иначе парсим
        if position_expected is not None and height_expected is not None:
            pos_exp = position_expected
            height_exp = height_expected
            print(f"\n📍 Using pre-parsed expectations (from IntentParser):")
        else:
            print("\n🔍 Analyzing expected peak change...")
            intent = self._parse_intent(expert_comment)
            pos_exp = intent["position_expected"]
            height_exp = intent["height_expected"]
        
        print(f"   Expected position: {pos_exp}")
        print(f"   Expected height:   {height_exp}")
        
        # Проверяем
        print("\n⚙️ Checking...")
        pos_check = self._check_position(baseline_peak, new_peak, pos_exp)
        height_check = self._check_height(baseline_height, new_height, height_exp)
        print(f"   Position: {'✓' if pos_check['ok'] else '✗'} {pos_check['description']}")
        print(f"   Height:   {'✓' if height_check['ok'] else '✗'} {height_check['description']}")
        
        # Решение
        print("\n⚖️ Decision...")
        decision = self._make_decision(pos_check, height_check)
        
        print(f"✅ {decision['decision'].upper()}")
        print(f"💭 {decision['reasoning']}")
        
        episode = Episode(
            beta=new_params['beta'],
            gamma=new_params['gamma'],
            mu=new_params['mu'],
            peak_position=new_peak,
            peak_height=new_height,
            total_deaths=new_results.get('total_deaths'),
            expert_comment=expert_comment,
            accepted=(decision['decision'] == 'accept'),
            iteration=len(self.history) + 1,
            reasoning=decision['reasoning']
        )
        
        self.add_to_history(episode)
        return episode
    
    def __call__(self, state: PipelineState) -> PipelineState:
        history = state.get('history', [])
        baseline_episode = None
        
        for ep in history:
            if ep.iteration == 0:
                baseline_episode = ep
                break
        
        if baseline_episode is None:
            baseline_episode = state.get('current_episode')
            if baseline_episode is None:
                raise ValueError("No baseline episode")
        
        # ✅ Берем готовые ожидания из state (уже распарсены IntentParser)
        position_expected = state.get('expected_position', 'unchanged')
        height_expected = state.get('expected_height', 'unchanged')
        
        # ✅ Передаем их в critique
        episode = self.critique(
            baseline_episode=baseline_episode,
            new_params=state.get('generated_params'),
            new_results=state.get('surrogate_results'),
            expert_comment=state.get('expert_comment'),
            position_expected=position_expected,    # ✅ новый параметр
            height_expected=height_expected         # ✅ новый параметр
        )
        
        state['critic_decision'] = 'accept' if episode.accepted else 'reject'
        state['critic_reasoning'] = episode.reasoning
        state['final_episode'] = episode
        
        return state