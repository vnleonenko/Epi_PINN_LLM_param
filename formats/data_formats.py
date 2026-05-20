import re
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Literal
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, field_validator, validator

# ============================================
# Pydantic models for output parsing
# ============================================

class EpiParameters(BaseModel):
    """Pydantic model for LLM output validation"""
    
    beta: float = Field(description="Infection rate (0.0 - 1.0)")
    gamma: float = Field(description="Recovery rate (0.00 - 1.0)")
    mu: float = Field(description="Mortality rate (0.000 - 0.1)")
    reasoning: str = Field(description="Detailed justification for parameter choices")
    # expected_peak_position: float = Field(description="Expected peak position in days")
    # expected_peak_height: float = Field(description="Expected peak height (number infected)")
    # expected_total_deaths: float = Field(description="Expected total deaths")
    confidence: str = Field(description="Confidence level: high/medium/low")
    
    @validator('beta')
    def beta_validator(cls, v):
        if not (0.0 <= v <= 3.0):
            raise ValueError(f'beta must be between 0.0 and 1.0, got {v}')
        return v
    
    @validator('gamma')
    def gamma_validator(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'gamma must be between 0.00 and 1.0, got {v}')
        return v
    
    @validator('mu')
    def mu_validator(cls, v):
        if not (0.00 <= v <= 0.1):
            raise ValueError(f'mu must be between 0.000 and 0.1, got {v}')
        return v
    
    @validator('confidence')
    def confidence_validator(cls, v):
        if v not in ['high', 'medium', 'low']:
            raise ValueError(f'confidence must be high/medium/low, got {v}')
        return v


@dataclass
class Episode:
    """Data class for storing parameter episodes"""
     
    beta: float
    gamma: float
    mu: float
    reasoning: Optional[str]
    peak_position: float = None
    peak_height: float = None
    total_deaths: float = None
    timestamp: str = None
    expert_comment: str = None
    accepted: bool = False
    iteration: int = None
    
    
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_prompt_format(self) -> str:
        """Format episode for prompt display"""
        status = "✓ ACCEPTED" if self.accepted else "✗ REJECTED"
        result = (
            f"**Iteration {self.iteration}:**\n"
            f"- Parameters: β={self.beta:.4f}, γ={self.gamma:.4f}, μ={self.mu:.5f}\n"
            f"- Results: peak at day {self.peak_position:.1f}, height {self.peak_height:.0f}, deaths {self.total_deaths:.0f}\n"
            f"- Status: {status}\n"
            f"- Expert comment: {self.expert_comment if self.expert_comment else 'None'}\n"
        )
        
        # Add reasoning if available
        if self.reasoning:
            result += f"- Reasoning: {self.reasoning}\n"
        
        return result
    

class CriticOutput(BaseModel):
    """Pydantic model for critic agent output"""
    reasoning: str = Field(description="Detailed reasoning for the decision")
    decision: str = Field(description="Decision: accept, reject, or adjust. Write with a lowercase letter")
    # suggested_beta: Optional[float] = Field(default=None)
    # suggested_gamma: Optional[float] = Field(default=None)
    # suggested_mu: Optional[float] = Field(default=None)
    confidence: str = Field(default="medium", description="Confidence level: high, medium, low")
    issues: List[str] = Field(default_factory=list)
    
    @field_validator('decision', mode='before')
    @classmethod
    def decision_validator(cls, v):
        """Convert decision to lowercase and validate"""
        if isinstance(v, str):
            v = v.lower().strip()
        if v not in ['accept', 'reject', 'adjust']:
            raise ValueError(f'decision must be accept/reject/adjust, got {v}')
        return v

class ExpertIntent(BaseModel):
            cares_about_position: bool = Field(description="Whether expert cares about peak position")
            position_direction: str = Field(description="Expected direction: 'later', 'earlier', or 'any'")
            cares_about_height: bool = Field(description="Whether expert cares about peak height")
            height_direction: str = Field(description="Expected direction: 'higher', 'lower', or 'any'")
            primary_metric: Literal["position", "height", "both"] = "both"
            reasoning: str = Field(description="Brief parsing reasoning")

# class GraphState:
#     """State for LangGraph workflow"""
#     def __init__(self, data: Dict = None):
#         self.data = data or {}
    
#     def get(self, key, default=None):
#         return self.data.get(key, default)
    
#     def set(self, key, value):
#         self.data[key] = value
    
#     def to_dict(self):
#         return self.data
    

# ============================================
# Graph State
# ============================================

class PipelineState(TypedDict):
    """State for the LangGraph pipeline"""
    # Task configuration
    task_config: Dict[str, Any]
    
    # Current state
    current_episode: Optional[Episode] 
    expert_comment: Optional[str]

    expected_position: Optional[str]   # "later", "earlier", "unchanged"
    expected_height: Optional[str]     # "higher", "lower", "unchanged"
    
    # History
    history: List[Episode]
    
    # Generation stage
    generated_params: Optional[Dict]
    
    # Surrogate stage
    surrogate_results: Optional[Dict]
    
    # Critic stage
    critic_decision: Optional[str]
    critic_reasoning: Optional[str]
    # suggested_params: Optional[Dict]
    
    # Final output
    final_episode: Optional[Episode] 

    pinn_results: Optional[Dict]
    
    # Control
    iteration: int
    max_iterations: int
    should_continue: bool
