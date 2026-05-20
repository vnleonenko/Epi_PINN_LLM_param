from agents.BaseLLMClient import BaseLLMClient
from agents.LLMClients import OpenAIClient, HuggingFaceClient, LocalVLLMClient, LMStudioClient
from typing import Optional
import os

class LLMFactory:
    """Фабрика для создания LLM клиентов"""
    
    @staticmethod
    def create_client(
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> BaseLLMClient:
        """
        Создать LLM клиент
        
        Args:
            provider: "openai", "huggingface", "vllm", "lmstudio"
            model_name: имя модели
            temperature: температура
            max_tokens: макс. токенов
            **kwargs: дополнительные параметры (api_key, device, use_api и т.д.)
        """
        
        if provider == "openai":
            return OpenAIClient(
                model_name=model_name or "gpt-4",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
            )
        
        elif provider == "huggingface":
            return HuggingFaceClient(
                model_name=model_name or "microsoft/phi-2",
                temperature=temperature,
                max_tokens=max_tokens,
                device=kwargs.get("device", "cpu"),
                use_api=kwargs.get("use_api", False),
                api_token=kwargs.get("api_token"),  
                load_in_8bit=kwargs.get("load_in_8bit", False),
                load_in_4bit=kwargs.get("load_in_4bit", False)
            )
        
        elif provider == "vllm":
            return LocalVLLMClient(
                model_name=model_name or "meta-llama/Llama-2-7b-chat-hf",
                temperature=temperature,
                max_tokens=max_tokens,
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1)
            )
        elif provider == "lmstudio":  
            return LMStudioClient(
                model_name=model_name or "local-model",
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=kwargs.get("base_url", "http://127.0.0.1:1234/v1"))
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, huggingface, vllm")
    
    @staticmethod
    def from_config(config: dict) -> BaseLLMClient:
        """Создать клиент из конфигурации"""
        return LLMFactory.create_client(
            provider=config.get("provider", "huggingface"),
            model_name=config.get("model_name"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
            **config.get("kwargs", {})
        )