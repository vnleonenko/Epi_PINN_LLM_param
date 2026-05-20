# agents/ParameterCriticAgent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from formats.data_formats import PipelineState, CriticOutput, Episode
from utils.PromptLogger import PromptLogger
from utils.RetryParser import RetryParser
from typing import Dict, List, Optional
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.exceptions import OutputParserException

from agents.BaseLLMClient import BaseLLMClient

# ============================================
# Parameter Critic LLM-Agent
# ============================================

class ParameterCriticAgent:
    """
    LLM agent that evaluates epidemiological parameters
    Determines if parameters will change forecast according to expert comment
    """
    
    def __init__(
        self,
        llm,
        max_retries=3,
        retry_temperature=0.3,
        enable_logging: bool = True,
        log_format: str = "json"  # "json" 
    ):
        """
        Initialize the critic agent
        
        Args:
            llm: LangChain LLM instance (ChatHuggingFace)
            max_retries: Maximum number of retries for parsing
            retry_temperature: Temperature for retry attempts
            enable_logging: Whether to enable prompt logging
            log_format: Log format - "json" or "text"
        """
        # Сохраняем клиент
        self.llm_client = llm if isinstance(llm, BaseLLMClient) else None
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=CriticOutput)
        self.max_retries = max_retries
        self.retry_temperature = retry_temperature
        self.retry_parser = None
        self.prompt = self._create_prompt()
        self.history: List[Episode] = []
        self.task_config: Dict = {}

        
        
        # Initialize logger
        self.enable_logging = enable_logging
        if enable_logging:
            if log_format == "json":
                self.logger = PromptLogger()
            # else:
            #     self.logger = SimplePromptLogger()
        else:
            self.logger = None
        
        self.retry_parser = RetryParser(
            llm=self._get_llm_for_retry(),
            parser=self.parser,
            max_retries=self.max_retries,
            delay=1,
            retry_temperature=self.retry_temperature
        )
    
    def _get_llm_for_retry(self):
        """Получить LLM для retry parser"""
        if self.llm_client:
            return self._create_langchain_compatible_llm()
        return self.llm
    
    def _create_langchain_compatible_llm(self):
        """Создать обертку для BaseLLMClient"""
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
        """Create the prompt template for parameter critique"""
        
        prompt_template = ChatPromptTemplate(messages=[
            ("system", """You are an expert epidemiologist evaluating SIRD model parameters.

    **YOUR TASK:** Determine if new parameters achieve the expert's intended direction.

    **VALID PARAMETER RANGES:**
    - β (infection rate): 0.0-1.0
    - γ (recovery rate): 0.0-1.0  
    - μ (mortality rate): 0.0-0.1

    **CRITICAL RULE - FOLLOW EXACTLY:**

    You MUST look at the expert comment and determine what SINGLE METRIC the expert wants to change.

    Then check ONLY that metric. DO NOT mention or evaluate any other metric in your reasoning or decision.

    **Mapping expert words to metrics:**
    | Expert says | Check ONLY this |
    |-------------|-----------------|
    | "later", "earlier", "move peak", "shift peak", "peak position" | Peak position |
    | "higher", "lower", "peak should be", "increase peak", "decrease peak" | Peak height |

    **Your decision logic:**
    - ACCEPT → The metric expert asked for moved in the correct direction
    - REJECT → The metric expert asked for moved in the WRONG direction
    - ADJUST → Correct direction but wrong magnitude

    **STRICT PROHIBITION:**
    - NEVER mention peak height if expert asked about position
    - NEVER mention peak position if expert asked about height
    - NEVER reject because an unmentioned metric changed

    **Examples of CORRECT behavior:**
    - Expert: "Move peak later" + Peak: 140→145, Height: 165→150 → ACCEPT (height not mentioned, so ignore it)
    - Expert: "Peak should be higher" + Height: 165→170, Peak: 140→135 → ACCEPT (position not mentioned, so ignore it)
    - Expert: "Need later peak" + Peak: 140→145 → ACCEPT
    - Expert: "Need later peak" + Peak: 140→135 → REJECT

    {format_instructions}

    Return ONLY valid JSON. No additional text."""),
            
            ("human", """## Expert Comment
    {expert_comment}

    ## Baseline
    - Peak day: {baseline_peak:.1f}
    - Peak height: {baseline_height:.0f} infected

    ## New Parameters
    - Peak day: {new_peak:.1f}
    - Peak height: {new_height:.0f} infected

    ## Change
    - Peak position: {peak_change:+.1f} days ({peak_direction})
    - Peak height: {height_change:+.0f} infected ({height_direction})

    ## Your Analysis

    **Step 1: What metric does the expert want to change?**
    Look at "{expert_comment}". The expert is asking to change: ___________ (write ONLY "peak position" or "peak height")

    **Step 2: Check ONLY that metric**
    - If expert wants peak position: Is {new_peak} > {baseline_peak}? (for "later") or < (for "earlier")?
    - If expert wants peak height: Is {new_height} > {baseline_height}? (for "higher") or < (for "lower")?

    **Step 3: Decision**
    - If answer to Step 2 is YES → ACCEPT
    - If answer to Step 2 is NO → REJECT

    **Step 4: Output JSON**

    Return JSON only.""")
        ],
        partial_variables={"format_instructions": self.parser.get_format_instructions()})
        
        return prompt_template

    def _format_changes(self, current: Dict, new: Dict) -> Dict:
        """Calculate and format changes between iterations"""
        peak_change = new.get('peak_position', 0) - current.get('peak_position', 0)
        height_change = new.get('peak_height', 0) - current.get('peak_height', 0)
        deaths_change = new.get('total_deaths', 0) - current.get('total_deaths', 0)
        
        return {
            'peak_change': peak_change,
            'peak_direction': "earlier" if peak_change < 0 else "later",
            'height_change': height_change,
            'height_direction': "lower" if height_change < 0 else "higher",
            'deaths_change': deaths_change,
            'deaths_direction': "fewer" if deaths_change < 0 else "more"
        }
    
    def set_task_config(self, config: Dict):
        """Set the task configuration"""
        self.task_config = config
        print(f"✅ Critic task configured: {config.get('description', 'No description')}")
    
    def add_to_history(self, episode: Episode):
        """Add episode to history"""
        self.history.append(episode)
        print(f"📝 Added episode {episode.iteration} to critic history")
    
    def critique(
        self,
        baseline_episode: Episode,
        new_params: Dict,
        new_results: Dict,
        expert_comment: str
    ) -> Episode:
        """
        Evaluate new parameters and return decision
        
        Args:
            current_episode: Current episode (previous iteration)
            new_params: New parameters to evaluate (beta, gamma, mu)
            new_results: Results from surrogate model for new parameters
            expert_comment: Expert feedback
            
        Returns:
            Episode with decision and reasoning
        """
        print("=" * 60)
        print("🔍 PARAMETER CRITIC AGENT")
        print("=" * 60)
        
        # Prepare current results from episode
        current_results = {
            'peak_position': baseline_episode.peak_position or 0,
            'peak_height': baseline_episode.peak_height or 0,
            'total_deaths': baseline_episode.total_deaths or 0
        }
        
        # Calculate changes
        changes = self._format_changes(current_results, new_results)
        
        # Prepare prompt inputs
        target_peak = self.task_config.get('target_peak', 'Not specified')
        target_desc = f"peak at day {target_peak}" if target_peak != 'Not specified' else self.task_config.get('target_metric', 'optimize epidemic parameters')
        
        prompt_inputs = {
            # Baseline parameters (как ожидает промпт)
            "baseline_beta": baseline_episode.beta,
            "baseline_gamma": baseline_episode.gamma,
            "baseline_mu": baseline_episode.mu,
            "baseline_peak": current_results['peak_position'],
            "baseline_height": current_results['peak_height'],
            "baseline_deaths": current_results['total_deaths'],
            
            # New parameters
            "new_beta": new_params.get('beta', 0),
            "new_gamma": new_params.get('gamma', 0),
            "new_mu": new_params.get('mu', 0),
            "new_peak": new_results.get('peak_position', 0),
            "new_height": new_results.get('peak_height', 0),
            "new_deaths": new_results.get('total_deaths', 0),
            
            # Changes
            "peak_change": changes['peak_change'],
            "peak_direction": changes['peak_direction'],
            "height_change": changes['height_change'],
            "height_direction": changes['height_direction'],
            "deaths_change": changes['deaths_change'],
            "deaths_direction": changes['deaths_direction'],
            
            # Expert comment
            "expert_comment": expert_comment if expert_comment else "No expert comment provided."
        }
        
        try:
            print("🚀 Sending request to LLM...")
            
            # Create chain
            chain = self.prompt | RunnableParallel(
                response=self.llm,
                prompt=RunnablePassthrough()
            )
            
            # Invoke chain
            result = chain.invoke(prompt_inputs)
            
            # Extract prompt text and response
            prompt_text = result["prompt"]
            if hasattr(prompt_text, 'to_string'):
                prompt_text_str = prompt_text.to_string()
            else:
                prompt_text_str = str(prompt_text)
            
            response_text = result["response"].content if hasattr(result["response"], 'content') else str(result["response"])
            
            if self.retry_parser is None:
                self.retry_parser = RetryParser(
                    llm=self.llm,
                    parser=self.parser,
                    max_retries=self.max_retries,
                    delay=1,
                    retry_temperature=self.retry_temperature
                )
            
            # Parse with retry
            parsed_output = self.retry_parser.parse(response_text, prompt_text_str)
            
            print(f"✅ Decision: {parsed_output.decision.upper()}")
            print(f"💭 Reasoning: {parsed_output.reasoning}")
            print(f"📊 Confidence: {parsed_output.confidence}")
            
            if parsed_output.issues:
                print(f"⚠️ Issues: {', '.join(parsed_output.issues)}")
            
            # Log prompt and response if logging is enabled
            if self.enable_logging and self.logger:
                iteration = len(self.history) + 1
                
                context = {
                    'task_config': self.task_config,
                    'current_params': {
                        'beta': baseline_episode.beta,
                        'gamma': baseline_episode.gamma,
                        'mu': baseline_episode.mu
                    },
                    'current_results': current_results,
                    'new_params': new_params,
                    'new_results': new_results,
                    'changes': changes,
                    'expert_comment': expert_comment
                }
                
                metadata = {
                    'decision': parsed_output.decision,
                    'confidence': parsed_output.confidence
                }

                self.logger.log_critic_prompt(
                    prompt_text=prompt_text_str,
                    response_text=response_text,
                    parsed_output=parsed_output,
                    context=context,
                    iteration=iteration,
                    metadata=metadata,  # ← добавить
                    llm=self.llm 
                )
            
            # Create episode with decision
            episode = Episode(
                beta=new_params['beta'],
                gamma=new_params['gamma'],
                mu=new_params['mu'],
                peak_position=new_results.get('peak_position'),
                peak_height=new_results.get('peak_height'),
                total_deaths=new_results.get('total_deaths'),
                expert_comment=expert_comment,
                accepted=(parsed_output.decision == 'accept'),
                iteration=len(self.history) + 1,
                reasoning=parsed_output.reasoning
            )
            
            # Add to history
            self.add_to_history(episode)
            
            # If decision is adjust and suggestions provided, create adjusted episode
            if parsed_output.decision == 'adjust':
                # suggested_beta = parsed_output.suggested_beta or new_params['beta']
                # suggested_gamma = parsed_output.suggested_gamma or new_params['gamma']
                # suggested_mu = parsed_output.suggested_mu or new_params['mu']
                
                # print(f"🔄 Suggested adjustments: β={suggested_beta:.4f}, γ={suggested_gamma:.4f}, μ={suggested_mu:.5f}")
                
                # # Store suggested params in episode for pipeline use
                # episode.suggested_beta = suggested_beta
                # episode.suggested_gamma = suggested_gamma
                # episode.suggested_mu = suggested_mu

                print(f"🔄 Parameters need adjustment according to critic")
            
            return episode
            
        except OutputParserException as e:
            print(f"❌ All retry attempts failed: {e}")
            
            # Log the error if logging is enabled
            if self.enable_logging and self.logger:
                try:
                    error_context = {
                        'error_type': 'OutputParserException',
                        'error_message': str(e),
                        'prompt_inputs': {k: str(v)[:500] for k, v in prompt_inputs.items()}
                    }
                    # Create a dummy parsed output for logging
                    dummy_output = CriticOutput(
                        decision='reject',
                        reasoning=f"Parser error: {str(e)}",
                        confidence=0.0,
                        issues=['Parsing failed']
                    )
                    self.logger.log_critic_prompt(
                        prompt_text=prompt_text_str if 'prompt_text_str' in locals() else "Error getting prompt",
                        response_text=response_text if 'response_text' in locals() else str(e),
                        parsed_output=dummy_output,
                        context=error_context,
                        iteration=len(self.history) + 1,
                        metadata={'error': True, 'error_type': 'OutputParserException'}
                    )
                except Exception as log_error:
                    print(f"⚠️ Could not log error: {log_error}")
            
            # Return rejected episode on failure
            return Episode(
                beta=new_params['beta'],
                gamma=new_params['gamma'],
                mu=new_params['mu'],
                peak_position=new_results.get('peak_position'),
                peak_height=new_results.get('peak_height'),
                total_deaths=new_results.get('total_deaths'),
                expert_comment=expert_comment,
                accepted=False,
                iteration=len(self.history) + 1,
                reasoning=f"Critic failed to parse response: {str(e)}"
            )
            
        except Exception as e:
            print(f"❌ Critique error: {e}")
            import traceback
            traceback.print_exc()
            
            # Log the error if logging is enabled
            if self.enable_logging and self.logger:
                try:
                    error_context = {
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': traceback.format_exc()[:1000]
                    }
                    dummy_output = CriticOutput(
                        decision='reject',
                        reasoning=f"Critic error: {str(e)}",
                        confidence=0.0,
                        issues=['Critical error during evaluation']
                    )
                    self.logger.log_critic_prompt(
                        prompt_text="Error occurred during critique",
                        response_text=str(e),
                        parsed_output=dummy_output,
                        context=error_context,
                        iteration=len(self.history) + 1,
                        metadata={'error': True, 'error_type': type(e).__name__}
                    )
                except Exception as log_error:
                    print(f"⚠️ Could not log error: {log_error}")
            
            return Episode(
                beta=new_params['beta'],
                gamma=new_params['gamma'],
                mu=new_params['mu'],
                peak_position=new_results.get('peak_position'),
                peak_height=new_results.get('peak_height'),
                total_deaths=new_results.get('total_deaths'),
                expert_comment=expert_comment,
                accepted=False,
                iteration=len(self.history) + 1,
                reasoning=f"Critic error: {str(e)}"
            )
    
    def __call__(self, state: PipelineState) -> PipelineState:
        # baseline_episode - это episode с iteration=0 (начальные параметры)
        history = state.get('history', [])
        baseline_episode = None
        for ep in history:
            if ep.iteration == 0:
                baseline_episode = ep
                break
        
        if baseline_episode is None:
            # Если нет истории, используем current_episode как baseline
            baseline_episode = state.get('current_episode')
        
        episode = self.critique(
            baseline_episode=baseline_episode,  # ← новое
            new_params=state.get('generated_params'),
            new_results=state.get('surrogate_results'),
            expert_comment=state.get('expert_comment')
        )
        
        state['critic_decision'] = 'accept' if episode.accepted else 'reject'
        state['critic_reasoning'] = episode.reasoning
        state['final_episode'] = episode
        
        return state