"""
Utilities for logging prompts sent to LLM and their responses
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import config


class PromptLogger:
    """
    Logger for saving prompts and responses from LLM interactions
    """

    def __init__(self, log_dir: str = "logs/prompts"):
        """
        Initialize prompt logger

        Args:
            log_dir: Directory to save logs (default: "logs/prompts")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.generator_dir = self.log_dir / "generator"
        self.critic_dir = self.log_dir / "critic"

        self.generator_dir.mkdir(parents=True, exist_ok=True)
        self.critic_dir.mkdir(parents=True, exist_ok=True)

        print(f"📝 PromptLogger initialized. Logs: {self.log_dir.absolute()}")

    # ============================================
    # Utility Methods
    # ============================================

    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _extract_llm_metadata(self, llm=None) -> Dict[str, Any]:
        """
        Extract LLM metadata directly from config
        """
        metadata = {
            "provider": config.LLM_PROVIDER,
            "model": config.DEFAULT_MODEL_NAME,
            "temperature": config.DEFAULT_TEMPERATURE,
            "max_tokens": config.DEFAULT_MAX_TOKENS,
        }
        
        # Добавляем специфичные параметры в зависимости от провайдера
        if config.LLM_PROVIDER == "huggingface":
            metadata["use_api"] = config.HF_USE_API
            metadata["device"] = config.HF_DEVICE
            metadata["model"] = config.HF_MODEL
            metadata["temperature"] = config.HF_TEMPERATURE
            metadata["max_tokens"] = config.HF_MAX_TOKENS
            
        elif config.LLM_PROVIDER == "openai":
            metadata["model"] = config.OPENAI_MODEL
            
        elif config.LLM_PROVIDER == "vllm":
            metadata["model"] = config.VLLM_MODEL
            metadata["temperature"] = config.VLLM_TEMPERATURE
            metadata["max_tokens"] = config.VLLM_MAX_TOKENS
            metadata["tensor_parallel_size"] = config.VLLM_TENSOR_PARALLEL
            
        elif config.LLM_PROVIDER == "lmstudio":
            metadata["model"] = config.LMSTUDIO_MODEL
            metadata["temperature"] = config.LMSTUDIO_TEMPERATURE
            metadata["max_tokens"] = config.LMSTUDIO_MAX_TOKENS
            metadata["base_url"] = config.LMSTUDIO_BASE_URL
        
        return metadata

    def _save_log(self, data: Dict[str, Any], log_type: str, subdir: Path) -> str:
        """
        Save log to file
        """
        timestamp = self._get_timestamp()
        filename = f"{log_type}_{timestamp}.json"
        filepath = subdir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)

    # ============================================
    # Generator Logging
    # ============================================

    def log_generator_prompt(
        self,
        prompt_text: str,
        response_text: str,
        parsed_output: Any,
        context: Dict[str, Any],
        iteration: int = 0,
        metadata: Optional[Dict] = None,
        llm=None
    ) -> str:
        """
        Log generator prompt
        """

        llm_metadata = self._extract_llm_metadata(llm) if llm else {}

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'generator',
            'iteration': iteration,
            'prompt': prompt_text,
            'response_raw': response_text,
            'parsed_output': (
                parsed_output if isinstance(parsed_output, dict)
                else parsed_output.dict()
            ),
            'context': context,
            'metadata': {
                **(metadata or {}),
                **llm_metadata
            }
        }

        filepath = self._save_log(
            log_data,
            f"generator_iter_{iteration:03d}",
            self.generator_dir
        )

        print(f"💾 Generator log saved: {filepath}")
        return filepath

    # ============================================
    # Critic Logging
    # ============================================

    def log_critic_prompt(
        self,
        prompt_text: str,
        response_text: str,
        parsed_output: Any,
        context: Dict[str, Any],
        iteration: int = 0,
        metadata: Optional[Dict] = None,
        llm=None
    ) -> str:
        """
        Log critic prompt
        """

        llm_metadata = self._extract_llm_metadata(llm) if llm else {}

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'critic',
            'iteration': iteration,
            'prompt': prompt_text,
            'response_raw': response_text,
            'parsed_output': (
                parsed_output if isinstance(parsed_output, dict)
                else parsed_output.dict()
            ),
            'context': context,
            'metadata': {
                **(metadata or {}),
                **llm_metadata
            }
        }

        filepath = self._save_log(
            log_data,
            f"critic_iter_{iteration:03d}",
            self.critic_dir
        )

        print(f"💾 Critic log saved: {filepath}")
        return filepath

    # ============================================
    # Generation Attempt Logging
    # ============================================

    def log_generation_attempt(
        self,
        iteration: int,
        params: Dict[str, float],
        reasoning: str,
        decision: Optional[str] = None,
        metadata: Optional[Dict] = None,
        llm=None
    ) -> str:
        """
        Log generation attempt
        """

        llm_metadata = self._extract_llm_metadata(llm) if llm else {}

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'generator',
            'iteration': iteration,
            'type': 'attempt',
            'params': params,
            'reasoning': reasoning,
            'decision': decision,
            'metadata': {
                **(metadata or {}),
                **llm_metadata
            }
        }

        filepath = self._save_log(
            log_data,
            f"attempt_iter_{iteration:03d}",
            self.generator_dir
        )

        print(f"💾 Attempt log saved: {filepath}")
        return filepath