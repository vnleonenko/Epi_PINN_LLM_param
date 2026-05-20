# agents/ReActCriticAgent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from formats.data_formats import PipelineState, Episode, ExpertIntent
from utils.PromptLogger import PromptLogger
from typing import Dict, List

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
        log_format: str = "json"
    ):
        self.llm_client = llm if isinstance(llm, BaseLLMClient) else None
        self.llm = llm
        self.max_retries = max_retries
        self.history: List[Episode] = []
        self.task_config: Dict = {}
        
        self.enable_logging = enable_logging
        if enable_logging and log_format == "json":
            self.logger = PromptLogger()
        else:
            self.logger = None
        
        
        
        self.intent_parser = PydanticOutputParser(pydantic_object=ExpertIntent)
        
        # ПОТОМ создаем промпт, который использует парсер
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
        
        # Частичное заполнение переменных
        return template.partial(format_instructions=self.intent_parser.get_format_instructions())
    
    # ========== ДЕТЕРМИНИРОВАННЫЕ ИНСТРУМЕНТЫ ==========
    
    def _check_position(self, baseline: float, new: float, direction: str) -> dict:
        """Deterministic position check"""
        if direction == "later":
            ok = new > baseline
            desc = f"moved later: {baseline:.1f} → {new:.1f}"
        elif direction == "earlier":
            ok = new < baseline
            desc = f"moved earlier: {baseline:.1f} → {new:.1f}"
        else:
            ok = True
            desc = f"changed: {baseline:.1f} → {new:.1f}"
        
        return {"ok": ok, "description": desc, "change": new - baseline}
    
    def _check_height(self, baseline: float, new: float, direction: str) -> dict:
        """Deterministic height check"""
        if direction == "higher":
            ok = new > baseline
            desc = f"increased: {baseline:.0f} → {new:.0f}"
        elif direction == "lower":
            ok = new < baseline
            desc = f"decreased: {baseline:.0f} → {new:.0f}"
        else:
            ok = True
            desc = f"changed: {baseline:.0f} → {new:.0f}"
        
        return {"ok": ok, "description": desc, "change": new - baseline}
    
    def _make_decision(self, pos_check: dict, height_check: dict, 
                      care_pos: bool, care_height: bool) -> dict:
        """Deterministic decision"""
        pos_ok = not care_pos or pos_check["ok"]
        height_ok = not care_height or height_check["ok"]
        
        if pos_ok and height_ok:
            decision = "accept"
            reasoning = "Parameters changed in expected direction"
        elif not pos_ok and not height_ok:
            decision = "reject"
            reasoning = "Parameters changed in wrong direction"
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
                return json.loads(json_match.group())
            return json.loads(text)
        except:
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
        print(f"✅ Critic task configured: {config.get('description', 'No description')}")
    
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
        print("🔍 CRITIC AGENT (Deterministic Tools)")
        print("=" * 60)
        
        baseline_peak = baseline_episode.peak_position or 0
        baseline_height = baseline_episode.peak_height or 0
        new_peak = new_results.get('peak_position', 0)
        new_height = new_results.get('peak_height', 0)
        
        print(f"📊 Baseline: peak={baseline_peak:.1f}, height={baseline_height:.0f}")
        print(f"📊 New:      peak={new_peak:.1f}, height={new_height:.0f}")
        print(f"💬 Expert:   {expert_comment}")
        
        # 1. Парсим намерения
        print("\n🔍 Parsing expert intent...")
        intent = self._parse_intent(expert_comment)
        print(f"   Position: {intent['cares_about_position']} ({intent['position_direction']})")
        print(f"   Height:   {intent['cares_about_height']} ({intent['height_direction']})")
        
        # 2. Детерминированные проверки
        print("\n⚙️ Running checks...")
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
        history = state.get('history', [])
        baseline_episode = None
        for ep in history:
            if ep.iteration == 0:
                baseline_episode = ep
                break
        
        if baseline_episode is None:
            baseline_episode = state.get('current_episode')
        
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