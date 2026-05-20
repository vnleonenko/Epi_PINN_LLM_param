from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Унифицированный ответ от LLM"""
    content: str
    raw_response: Any = None
    model_name: str
    usage: Dict[str, int] = {}  # tokens usage если доступно

class BaseLLMClient(ABC):
    """Абстрактный клиент для работы с различными LLM"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Вызов LLM с промптом"""
        pass
    
    @abstractmethod
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Вызов LLM с списком сообщений (chat format)"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Стриминг ответа (опционально)"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": self.__class__.__name__
        }