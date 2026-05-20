# agents/EpiParamGeneratorAgent.py

from typing import Dict, List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.exceptions import OutputParserException
from utils.RetryParser import RetryParser

from formats.data_formats import EpiParameters, Episode, PipelineState
from utils.PromptLogger import PromptLogger
from agents.BaseLLMClient import BaseLLMClient, LLMResponse



class LLMEpiParamGenerator:
    """
    LLM agent that generates epidemiological parameters for SIRD model
    """
    
    def __init__(
        self,
        llm,
        output_class=EpiParameters,
        enable_logging: bool = True,
        log_format: str = "json",
        max_retries: int = 3,
        retry_temperature: float = 0.3 
    ):
        """
        Initialize the generator agent
        
        Args:
            llm: LangChain LLM instance (ChatHuggingFace)
            output_class: Pydantic class for output validation
            enable_logging: Whether to enable prompt logging
            log_format: Log format - "json" or "text"
        """
        # Сохраняем llm_client
        self.llm_client = llm if isinstance(llm, BaseLLMClient) else None
        self.llm = llm  # для обратной совместимости

        self.parser = PydanticOutputParser(pydantic_object=output_class)
        self.prompt = self._create_prompt()
        self.history: List[Episode] = []
        self.task_config: Dict = {}
        
        # Initialize logger
        self.enable_logging = enable_logging
        if enable_logging:
            if log_format == "json":
                self.logger = PromptLogger()
        else:
            self.logger = None

        self.max_retries = max_retries
        self.retry_temperature = retry_temperature
        self.retry_parser = None
        if self.retry_parser is None:
            self.retry_parser = RetryParser(
                llm=self._get_llm_for_retry(),
                parser=self.parser,
                max_retries=self.max_retries,
                delay=1,
                retry_temperature=self.retry_temperature
            )
    

    def _get_llm_for_retry(self):
        """Получить LLM объект для retry parser"""
        if self.llm_client:
            # Для BaseLLMClient нужно адаптировать под интерфейс langchain
            # Создаем обертку или используем напрямую
            return self._create_langchain_compatible_llm()
        return self.llm
    
    def _create_langchain_compatible_llm(self):
        """Создать обертку для BaseLLMClient для совместимости с langchain"""
        from langchain_core.language_models.llms import LLM
        from typing import Any, List, Mapping, Optional
        
        class LLMClientWrapper(LLM):
            client: BaseLLMClient
            
            @property
            def _llm_type(self) -> str:
                return "base_llm_client_wrapper"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                response = self.client.invoke(prompt)
                return response.content
            
            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"model": self.client.model_name}
        
        return LLMClientWrapper(client=self.llm_client)
    
    def _call_llm(self, prompt: str) -> str:
        """Вызов LLM через унифицированный интерфейс"""
        if self.llm_client:
            response = self.llm_client.invoke(prompt)
            return response.content
        elif hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            raise ValueError("No valid LLM client available")

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for parameter generation"""
        
        prompt_template = ChatPromptTemplate(messages=[
            ("system", """You are an expert epidemiologist specialized in SIRD model parameter optimization. Your task is to select optimal parameters (β, γ, μ) to achieve target epidemic scenarios.

**Your role:**
- Analyze the current epidemic state and history of previous attempts
- Consider expert feedback to improve parameter selection
- Apply epidemiological knowledge about parameter relationships
- Balance exploration and exploitation in the parameter space

**Key epidemiological relationships:**
- **R0 = β/(γ+μ) > 1.0** (epidemic will grow, not die out)
             
**🚨 SIMPLE RULE FOR PEAK HEIGHT (FOLLOW EXACTLY):**
- If expected_height = "lower" → you MUST DECREASE β and/or INCREASE γ
- If expected_height = "higher" → you MUST INCREASE β and/or DECREASE γ

**Optimization strategy:**
1. Start with reasonable parameter ranges
2. Adjust based on expert feedback
3. Learn from successful and failed attempts
4. Consider trade-offs between peak timing, peak height           

**FLEXIBLE PARAMETER UPDATES:**
- In the FIRST 2-3 iterations, change ONLY ONE parameter at a time
- This creates a clear cause-effect relationship for the critic to learn from
- You do NOT need to change all parameters (β, γ, μ) in every iteration
- Changing 1 or 2 parameters is perfectly acceptable
- The only requirement is that at least ONE parameter changes from the previous iteration
- Keeping some parameters stable helps isolate the effect of changes
             

{format_instructions}

Return ONLY valid JSON in the exact format specified. Do not include any additional text, explanations, or markdown formatting."""),

            ("human", """Please generate new epidemiological parameters for the SIRD model based on the following information:

## 🎯 TARGET PEAK DIRECTION (from expert comment)
Expected position change: **{expected_position}**
Expected height change: **{expected_height}**


## 0. Parameter Sensitivity Map — YOUR CHEAT SHEET
{sensitivity_map}

**How to use the Sensitivity Map:**
- This map shows EXACTLY how changing each parameter affects the peak.
- It was computed SPECIFICALLY for your current baseline parameters.
- Use it to make INFORMED, PRECISE adjustments — NOT random guesses.

**Example of using the map:**
- If expected_position = "later" → look at the map: which parameter change gives POSITIVE Δ day?
- If expected_height = "higher" → look at the map: which parameter change gives POSITIVE Δ height?
- Combine adjustments proportionally to achieve both goals.


## 1. Task Configuration
{task_config}
             
**🚨 CRITICAL: PARAMETER BOUNDS (READ FIRST)**

You MUST generate parameters within these EXACT bounds:
- β (infection rate): between {beta_min:.5f} and {beta_max:.5f}
- γ (recovery rate): between {gamma_min:.5f} and {gamma_max:.5f}
- μ (mortality rate): between {mu_min:.6f} and {mu_max:.6f}

**DO NOT suggest values outside these ranges! They will be rejected.**

## 2. Current Episode (Previous Attempt)
{current_episode}

## 3. History of Previous Attempts
{history}

## 4. Expert Comment (Goal to Achieve)
{expert_comment}

## 5. Statistical Summary of Previous Attempts
{stats_summary}

## 6. Target Metrics
{target_metrics}
                    
## 7. BALANCE ANALYSIS (CRITICAL!)
Based on the history above, check if you are FOCUSING ONLY ON ONE METRIC:





Generate new parameters that will bring us closer to the target epidemic scenario described in the Expert Comment.
Use the history of previous attempts to learn what worked and what didn't.
Return only the valid JSON object without any additional text.""")
        ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()})
        
        return prompt_template
    
    def _format_history(self, history: List, max_episodes: int = 5) -> str:
        """Format history episodes for prompt - handles Episode objects only"""
        if not history:
            return "No previous attempts available."
        
        recent_history = history[-max_episodes:]
        
        formatted = ""
        for episode in recent_history:
            if hasattr(episode, 'to_prompt_format'):
                formatted += episode.to_prompt_format() + "\n"
            else:
                status = "✅ ACCEPTED" if getattr(episode, 'accepted', False) else "❌ REJECTED"
                formatted += f"""
    Episode {getattr(episode, 'iteration', 'N/A')}:
    - Parameters: β={getattr(episode, 'beta', 0):.4f}, γ={getattr(episode, 'gamma', 0):.4f}, μ={getattr(episode, 'mu', 0):.5f}
    - Results: peak at day {getattr(episode, 'peak_position', 0):.1f}, height {getattr(episode, 'peak_height', 0):.0f}, deaths {getattr(episode, 'total_deaths', 0):.0f}
    - Status: {status}
    - Reasoning: {getattr(episode, 'reasoning', 'No reasoning')[:100]}...
    """
        
        return formatted
    
    def _format_stats_summary(self, history: List) -> str:
        """Generate statistical summary from history - handles Episode objects"""
        if not history:
            return "No statistics available yet."
        
        betas = [ep.beta for ep in history if hasattr(ep, 'beta')]
        gammas = [ep.gamma for ep in history if hasattr(ep, 'gamma')]
        mus = [ep.mu for ep in history if hasattr(ep, 'mu')]
        peaks = [ep.peak_position for ep in history if hasattr(ep, 'peak_position') and ep.peak_position is not None]
        deaths = [ep.total_deaths for ep in history if hasattr(ep, 'total_deaths') and ep.total_deaths is not None]
        
        if not betas:
            return "No valid parameter data in history."
        
        target_peak = self.task_config.get('target_peak', 30)
        
        best_episode = None
        best_error = float('inf')
        
        for ep in history:
            if hasattr(ep, 'peak_position') and ep.peak_position:
                error = abs(ep.peak_position - target_peak)
                if error < best_error:
                    best_error = error
                    best_episode = ep
        
        summary = f"""
    **Parameter ranges explored:**
    - β: {min(betas):.4f} – {max(betas):.4f}
    - γ: {min(gammas):.4f} – {max(gammas):.4f}
    - μ: {min(mus):.5f} – {max(mus):.5f}

    **Results achieved:**
    - Peak position: {min(peaks):.1f} – {max(peaks):.1f} days (target: {target_peak})
    - Total deaths: {min(deaths):.0f} – {max(deaths):.0f}

    **Best attempt so far:**
    """
        if best_episode:
            summary += f"""
    - β={best_episode.beta:.4f}, γ={best_episode.gamma:.4f}, μ={best_episode.mu:.5f}
    - Peak at day {best_episode.peak_position:.1f} with {best_episode.peak_height:.0f} infected
    - Total deaths: {best_episode.total_deaths:.0f}
    """
        
        return summary
    
    def _format_current_episode(self, episode) -> str:
        """Format current episode for prompt - handles Episode object"""
        if not episode:
            return "No current episode available."
        
        if hasattr(episode, 'beta'):
            return f"""
        - β = {episode.beta:.4f}
        - γ = {episode.gamma:.4f}
        - μ = {episode.mu:.5f}
        - Peak at day {getattr(episode, 'peak_position', 0):.1f}
        - Peak height: {getattr(episode, 'peak_height', 0):.0f} infected
        - Total deaths: {getattr(episode, 'total_deaths', 0):.0f}
        """
        else:
            return f"Unknown episode format: {type(episode)}"
    
    def _format_task_config(self) -> str:
        """Format task configuration for prompt"""
        if not self.task_config:
            return "No task configuration provided."
        
        config_str = f"""
- Description: {self.task_config.get('description', 'Not specified')}
- Population: {self.task_config.get('population', '1,000,000'):,}
- Initial infected: {self.task_config.get('I0', 100)}
- Target peak: {self.task_config.get('target_peak', 'Not specified')} days
"""
        return config_str
    
    def _format_target_metrics(self) -> str:
        """Format target metrics for prompt"""
        if not self.task_config:
            return "No target metrics specified."
        
        metrics = f"""
- Desired peak position: {self.task_config.get('target_peak', 'Not specified')} days
"""
        if self.task_config.get('target_height'):
            metrics += f"- Desired peak height: {self.task_config.get('target_height'):,.0f} infected\n"
        if self.task_config.get('target_deaths'):
            metrics += f"- Desired total deaths: {self.task_config.get('target_deaths'):,.0f}\n"
        
        return metrics
    
    def _format_sensitivity_map(self, sensitivity_map: dict) -> str:
        """Форматирует карту чувствительности для вставки в промпт"""
        
        if not sensitivity_map:
            return "No sensitivity map available."
        
        baseline = sensitivity_map['baseline']
        
        text = f"""
    **PARAMETER SENSITIVITY MAP (computed around your baseline):**

    Baseline: β={baseline['beta']:.4f}, γ={baseline['gamma']:.4f}, μ={baseline['mu']:.5f}
    Baseline peak: position={baseline['peak_position']:.1f} days, height={baseline['peak_height']:.0f} infected

    **How parameters affect the peak:**

    ┌──────────┬─────────────┬──────────────────────┬─────────────────────┐
    │ Parameter│ Change      │ Peak position Δ      │ Peak height Δ       │
    ├──────────┼─────────────┼──────────────────────┼─────────────────────┤
    """
        
        # Beta
        for i, var in enumerate(sensitivity_map['beta']['variations']):
            res = sensitivity_map['beta']['results'][i]
            text += f"│ β        │ {var:+.0%}        │ {res['position_delta']:+.1f} days             │ {res['height_delta']:+.0f} infected          │\n"
        
        text += "├──────────┼─────────────┼──────────────────────┼─────────────────────┤\n"
        
        # Gamma
        for i, var in enumerate(sensitivity_map['gamma']['variations']):
            res = sensitivity_map['gamma']['results'][i]
            text += f"│ γ        │ {var:+.0%}        │ {res['position_delta']:+.1f} days             │ {res['height_delta']:+.0f} infected          │\n"
        
        text += "├──────────┼─────────────┼──────────────────────┼─────────────────────┤\n"
        
        # Mu
        for i, var in enumerate(sensitivity_map['mu']['variations']):
            res = sensitivity_map['mu']['results'][i]
            text += f"│ μ        │ {var:+.0%}        │ {res['position_delta']:+.1f} days             │ {res['height_delta']:+.0f} infected          │\n"
        
        text += """└──────────┴─────────────┴──────────────────────┴─────────────────────┘

    

    **How to use this map:**
    To achieve desired peak changes, combine parameter adjustments proportionally.
    """
        
        return text
    
    def set_task_config(self, config: Dict):
        """Set the task configuration"""
        self.task_config = config
        print(f"✅ Task configured: {config.get('description', 'No description')}")
    
    def update_history(self, history: List[Episode]):
        """Update history from critic agent"""
        self.history = history
        print(f"📚 Updated generator history with {len(history)} episodes")
    
    def generate(self, state: PipelineState) -> PipelineState:
        """
        Generate new parameters based on current state
        """
        print("=" * 60)
        print("🎯 LLM-EPIPARAM GENERATOR AGENT")
        print("=" * 60)
        
        current_iteration = state.get('iteration', 0)
        print(f"📍 Current iteration from state: {current_iteration}")
        
        current_episode = state.get('current_episode')
        
        # ✅ Экспертный комментарий всегда есть в state
        expert_comment = state.get('expert_comment')

        
        # Проверяем наличие экспертного комментария
        if expert_comment is None or expert_comment.strip() == "":
            raise ValueError("❌ Expert comment is required but not provided in state!")
        
        expected_position = state.get('expected_position', 'unchanged')
        expected_height = state.get('expected_height', 'unchanged')
        direction_hint = state.get('direction_hint', '')
        
        print(f"📝 Expert comment: {expert_comment}")
        print(f"🎯 Expected: position={expected_position}, height={expected_height}")
        if direction_hint:
            print(f"💡 Hint: {direction_hint}")
        
        if not current_episode:
            print("❌ No current episode in state")
            return state
        
        history_from_state = state.get('history', [])

        if current_episode and hasattr(current_episode, 'beta'):
            current_beta = current_episode.beta
            current_gamma = current_episode.gamma
            current_mu = current_episode.mu

        # current_beta = current_episode.beta
        # current_gamma = current_episode.gamma
        # current_mu = current_episode.mu
        
        beta_min = current_beta - 0.01
        beta_max = current_beta + 0.01 # 0.002
        gamma_min = current_gamma - 0.0005
        gamma_max = current_gamma + 0.0005
        mu_min = current_mu - 0.00005
        mu_max = current_mu + 0.00005

        sensitivity_map = state.get('sensitivity_map', {})
        sensitivity_text = self._format_sensitivity_map(sensitivity_map)
        
        # Prepare prompt inputs
        prompt_inputs = {
            "task_config": self._format_task_config(),
            "current_episode": self._format_current_episode(current_episode),
            "history": self._format_history(history_from_state),
            "expert_comment": expert_comment,
            "stats_summary": self._format_stats_summary(history_from_state),
            "target_metrics": self._format_target_metrics(),
            "current_beta": current_beta,
            "current_gamma": current_gamma,
            "current_mu": current_mu,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "gamma_min": gamma_min,
            "gamma_max": gamma_max,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "sensitivity_map": sensitivity_text,
            "expected_position": expected_position,   
            "expected_height": expected_height,        
            "direction_hint": direction_hint,         
        }
        
        # Create chain
        chain = self.prompt | RunnableParallel(
            response=self.llm, 
            prompt=RunnablePassthrough()
        )
        
        try:
            print("🚀 Generating new parameters...")
            result = chain.invoke(prompt_inputs)
            
            # Get prompt text and response
            prompt_text = result.get('prompt', {})
            if hasattr(prompt_text, 'to_string'):
                prompt_text_str = prompt_text.to_string()
            else:
                prompt_text_str = str(prompt_text)
            
            raw_response = result.get('response', {}).content if hasattr(result.get('response', {}), 'content') else str(result)
            
            # Parse the response
            parsed_output = self.retry_parser.parse(raw_response, prompt_text_str)
            
            print(f"✅ Generated: β={parsed_output.beta:.4f}, γ={parsed_output.gamma:.4f}, μ={parsed_output.mu:.5f}")
            print(f"💭 Reasoning: {parsed_output.reasoning}")
            
            # Log prompt and response if logging is enabled
            if self.enable_logging and self.logger:
                context = {
                    'task_config': self.task_config,
                    'current_episode_beta': current_episode.beta if hasattr(current_episode, 'beta') else None,
                    'current_episode_gamma': current_episode.gamma if hasattr(current_episode, 'gamma') else None,
                    'current_episode_mu': current_episode.mu if hasattr(current_episode, 'mu') else None,
                    'current_episode_accepted': getattr(current_episode, 'accepted', False),
                    'history_length': len(self.history),
                    'expert_comment': expert_comment
                }
                
                metadata = {
                    'iteration': current_iteration,
                    'model': getattr(self.llm, 'model_id', 'unknown')
                }
                
                self.logger.log_generator_prompt(
                                        prompt_text=prompt_text_str,
                                        response_text=raw_response,
                                        parsed_output=parsed_output,
                                        context=context,
                                        iteration=current_iteration,
                                        metadata=metadata,
                                        llm=self.llm 
                                    )
            
            # Store generated parameters in state
            state['generated_params'] = {
                'reasoning': parsed_output.reasoning,
                'beta': parsed_output.beta,
                'gamma': parsed_output.gamma,
                'mu': parsed_output.mu,
                'confidence': parsed_output.confidence
            }

            state['iteration'] = state['iteration'] + 1
            
            return state
            
        except OutputParserException as e:
            print(f"❌ All retry attempts failed: {e}")
            state['generated_params'] = None
            state['generation_error'] = f"Parser error after {self.max_retries} retries: {str(e)}"
            return state
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Call the agent"""
        return self.generate(state)