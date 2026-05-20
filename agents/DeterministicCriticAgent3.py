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
    Simplified critic agent - LLM parses expert comment, deterministic tools check changes.
    No LangChain agents - direct and reliable.
    """
    
    def __init__(
        self,
        llm,
        max_retries=3,
        enable_logging: bool = True,
        log_format: str = "json",
        # Пороги значимости (можно переопределить)
        position_threshold: float = 0.0,      # дней
        height_threshold_relative: float = 0.0  # 5% от baseline
    ):
        self.llm_client = llm if isinstance(llm, BaseLLMClient) else None
        self.llm = llm
        self.max_retries = max_retries
        self.history: List[Episode] = []
        self.task_config: Dict = {}
        
        # Пороги для определения значимого изменения
        self.position_threshold = position_threshold
        self.height_threshold_relative = height_threshold_relative
        
        self.enable_logging = enable_logging
        if enable_logging and log_format == "json":
            self.logger = PromptLogger()
        else:
            self.logger = None
        
        self.intent_parser = PydanticOutputParser(pydantic_object=ExpertIntent)
        self.intent_prompt = self._create_intent_prompt()
    
    def _create_intent_prompt(self) -> ChatPromptTemplate:
        """Create prompt for parsing expert intent"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """Parse the expert comment to extract requirements.

    Rules:
    - "later"/"earlier"/"shift" → cares about peak position
    - "higher"/"lower"/"increase"/"decrease" → cares about peak height
    - If both mentioned → mark both as True
    - If unclear → mark both as True with direction "any"

    {format_instructions}

    Return ONLY valid JSON."""),
            ("human", "Expert comment: {expert_comment}")
        ])
        
        return template.partial(format_instructions=self.intent_parser.get_format_instructions())
    
    # ========== ДЕТЕРМИНИРОВАННЫЕ ИНСТРУМЕНТЫ С ПОРОГОМ ЗНАЧИМОСТИ ==========
    
    def _check_position(self, baseline: float, new: float, direction: str) -> dict:
        """
        Deterministic position check with significance threshold.
        
        Returns:
            dict with keys: ok, description, change, is_significant
        """
        change = new - baseline
        is_significant = abs(change) >= self.position_threshold
        
        if direction == "later":
            ok = change > 0 and is_significant
            if not is_significant and change > 0:
                desc = f"moved later but insignificantly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} < {self.position_threshold} days)"
            elif ok:
                desc = f"moved later significantly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} days)"
            elif change <= 0:
                desc = f"moved earlier instead of later: {baseline:.1f} → {new:.1f}"
            else:
                desc = f"no significant change: {baseline:.1f} → {new:.1f}"
                
        elif direction == "earlier":
            ok = change < 0 and is_significant
            if not is_significant and change < 0:
                desc = f"moved earlier but insignificantly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} < {self.position_threshold} days)"
            elif ok:
                desc = f"moved earlier significantly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} days)"
            elif change >= 0:
                desc = f"moved later instead of earlier: {baseline:.1f} → {new:.1f}"
            else:
                desc = f"no significant change: {baseline:.1f} → {new:.1f}"
        else:
            # direction == "any"
            ok = is_significant
            if ok:
                desc = f"changed significantly: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} days)"
            else:
                desc = f"no significant change: {baseline:.1f} → {new:.1f} (Δ={change:+.1f} < {self.position_threshold} days)"
        
        return {
            "ok": ok, 
            "description": desc, 
            "change": change,
            "is_significant": is_significant
        }
    
    def _check_height(self, baseline: float, new: float, direction: str) -> dict:
        """
        Deterministic height check with significance threshold.
        
        Returns:
            dict with keys: ok, description, change, is_significant
        """
        change = new - baseline
        relative_change = abs(change) / baseline if baseline > 0 else 0
        is_significant = relative_change >= self.height_threshold_relative
        
        if direction == "higher":
            ok = change > 0 and is_significant
            if not is_significant and change > 0:
                desc = f"increased but insignificantly: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, {relative_change*100:.1f}% < {self.height_threshold_relative*100:.0f}%)"
            elif ok:
                desc = f"increased significantly: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, +{relative_change*100:.1f}%)"
            elif change <= 0:
                desc = f"decreased instead of increased: {baseline:.0f} → {new:.0f}"
            else:
                desc = f"no significant change: {baseline:.0f} → {new:.0f}"
                
        elif direction == "lower":
            ok = change < 0 and is_significant
            if not is_significant and change < 0:
                desc = f"decreased but insignificantly: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, {relative_change*100:.1f}% < {self.height_threshold_relative*100:.0f}%)"
            elif ok:
                desc = f"decreased significantly: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, -{relative_change*100:.1f}%)"
            elif change >= 0:
                desc = f"increased instead of decreased: {baseline:.0f} → {new:.0f}"
            else:
                desc = f"no significant change: {baseline:.0f} → {new:.0f}"
        else:
            # direction == "any"
            ok = is_significant
            if ok:
                desc = f"changed significantly: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, {relative_change*100:.1f}%)"
            else:
                desc = f"no significant change: {baseline:.0f} → {new:.0f} (Δ={change:+.0f}, {relative_change*100:.1f}% < {self.height_threshold_relative*100:.0f}%)"
        
        return {
            "ok": ok, 
            "description": desc, 
            "change": change,
            "is_significant": is_significant,
            "relative_change": relative_change
        }
    
    def _make_decision(self, pos_check: dict, height_check: dict, 
                      care_pos: bool, care_height: bool) -> dict:
        """Deterministic decision based on significance checks"""
        pos_ok = not care_pos or pos_check["ok"]
        height_ok = not care_height or height_check["ok"]
        
        # Проверяем, было ли вообще значимое изменение в том, о чём заботится эксперт
        pos_significant = pos_check.get("is_significant", False)
        height_significant = height_check.get("is_significant", False)
        
        if pos_ok and height_ok:
            # Дополнительно проверяем, было ли реально значимое изменение
            if (care_pos and not pos_significant) or (care_height and not height_significant):
                decision = "reject"
                reasoning = "Change was in expected direction but not significant enough"
            else:
                decision = "accept"
                reasoning = "Parameters changed significantly in expected direction"
        elif not pos_ok and not height_ok:
            decision = "reject"
            reasoning = "Parameters changed in wrong direction or insignificantly"
        else:
            decision = "adjust"
            reasoning = "Partial success - some metrics need adjustment"
        
        # Добавляем детали
        details = []
        if care_pos:
            details.append(f"Position: {pos_check['description']}")
        if care_height:
            details.append(f"Height: {height_check['description']}")
        
        if details:
            reasoning += ". " + "; ".join(details)
        
        return {"decision": decision, "reasoning": reasoning}
    
    # ========== ОСНОВНЫЕ МЕТОДЫ ==========
    
    def _parse_intent(self, comment: str) -> dict:
        """Parse expert intent with LLM or fallback"""
        if not comment:
            return {"cares_about_position": True, "position_direction": "any",
                   "cares_about_height": True, "height_direction": "any"}
        
        try:
            chain = self.intent_prompt | self.llm
            result = chain.invoke({"expert_comment": comment})
            text = result.content if hasattr(result, 'content') else str(result)
            
            # Извлекаем JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                print(f"   [LLM parsed intent successfully]")
                return parsed
            return json.loads(text)
        except Exception as e:
            print(f"   ⚠️ LLM intent parsing failed, using fallback: {e}")
            # Fallback: keyword matching
            low = comment.lower()
            care_pos = any(w in low for w in ["later", "earlier", "shift", "peak", "position"])
            care_height = any(w in low for w in ["higher", "lower", "increase", "decrease", "height"])
            
            pos_dir = "any"
            if "later" in low: pos_dir = "later"
            elif "earlier" in low: pos_dir = "earlier"
            
            height_dir = "any"
            if "higher" in low or "increase" in low: height_dir = "higher"
            elif "lower" in low or "decrease" in low: height_dir = "lower"
            
            return {
                "cares_about_position": care_pos or not (care_pos or care_height),
                "position_direction": pos_dir,
                "cares_about_height": care_height or not (care_pos or care_height),
                "height_direction": height_dir
            }
    
    def set_task_config(self, config: Dict):
        self.task_config = config
        # Можно переопределить пороги из конфига
        if 'position_threshold' in config:
            self.position_threshold = config['position_threshold']
        if 'height_threshold_relative' in config:
            self.height_threshold_relative = config['height_threshold_relative']
        print(f"✅ Critic task configured: {config.get('description', 'No description')}")
        print(f"   Thresholds: position={self.position_threshold} days, height={self.height_threshold_relative*100:.0f}%")
    
    def add_to_history(self, episode: Episode):
        self.history.append(episode)
        print(f"📝 Added episode {episode.iteration} to history")
    
    def critique(
        self,
        baseline_episode: Episode,
        new_params: Dict,
        new_results: Dict,
        expert_comment: str
    ) -> Episode:
        """Main evaluation method"""
        print("=" * 60)
        print("🔍 CRITIC AGENT (Deterministic Tools with Significance Thresholds)")
        print("=" * 60)
        
        baseline_peak = baseline_episode.peak_position or 0
        baseline_height = baseline_episode.peak_height or 0
        new_peak = new_results.get('peak_position', 0)
        new_height = new_results.get('peak_height', 0)
        
        print(f"📊 Baseline: peak={baseline_peak:.1f}, height={baseline_height:.0f}")
        print(f"📊 New:      peak={new_peak:.1f}, height={new_height:.0f}")
        print(f"📏 Thresholds: position={self.position_threshold} days, height={self.height_threshold_relative*100:.0f}%")
        print(f"💬 Expert:   {expert_comment}")
        
        # 1. Парсим намерения
        print("\n🔍 Parsing expert intent...")
        intent = self._parse_intent(expert_comment)
        print(f"   Position: {intent['cares_about_position']} ({intent['position_direction']})")
        print(f"   Height:   {intent['cares_about_height']} ({intent['height_direction']})")
        
        # 2. Детерминированные проверки с порогами
        print("\n⚙️ Running significance checks...")
        pos_check = self._check_position(baseline_peak, new_peak, intent['position_direction'])
        height_check = self._check_height(baseline_height, new_height, intent['height_direction'])
        print(f"   Position: {'✓' if pos_check['ok'] else '✗'} {pos_check['description']}")
        print(f"   Height:   {'✓' if height_check['ok'] else '✗'} {height_check['description']}")
        
        # 3. Принимаем решение
        print("\n⚖️ Making decision...")
        decision = self._make_decision(pos_check, height_check, 
                                       intent['cares_about_position'],
                                       intent['cares_about_height'])
        
        print(f"✅ Decision: {decision['decision'].upper()}")
        print(f"💭 {decision['reasoning']}")
        
        # Создаем эпизод
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
        """
        Вызов агента из LangGraph.
        Baseline ищется по флагу is_baseline в Episode.
        """
        history = state.get('history', [])
        baseline_episode = None
        
        # Ищем baseline по явному флагу is_baseline
        for ep in history:
            if getattr(ep, 'is_baseline', False) or ep.iteration == 0:  # обратная совместимость
                baseline_episode = ep
                break
        
        # Если не нашли — используем current_episode (но логируем предупреждение)
        if baseline_episode is None:
            print("⚠️ WARNING: Baseline episode not found in history, using current_episode as fallback")
            baseline_episode = state.get('current_episode')
            
            # Если и current_episode нет — критическая ошибка
            if baseline_episode is None:
                raise ValueError("CRITICAL: No baseline episode available for comparison")
        
        episode = self.critique(
            baseline_episode=baseline_episode,
            new_params=state.get('generated_params'),
            new_results=state.get('surrogate_results'),
            expert_comment=state.get('expert_comment')
        )
        
        state['critic_decision'] = 'accept' if episode.accepted else 'reject'
        state['critic_reasoning'] = episode.reasoning
        state['final_episode'] = episode
        
        return state