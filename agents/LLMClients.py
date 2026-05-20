from agents.BaseLLMClient import BaseLLMClient, LLMResponse
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Клиент для OpenAI API"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, 
                 max_tokens: int = 1000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Вызов с промптом"""
        try:
            response = self.llm.invoke(prompt)
            return LLMResponse(
                content=response.content,
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage_metadata or {}
            )
        except Exception as e:
            logger.error(f"OpenAI invoke error: {e}")
            raise
    
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Вызов с сообщениями"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            response = self.llm.invoke(langchain_messages)
            return LLMResponse(
                content=response.content,
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage_metadata or {}
            )
        except Exception as e:
            logger.error(f"OpenAI messages error: {e}")
            raise
    
    def stream(self, prompt: str, **kwargs):
        """Стриминг"""
        return self.llm.stream(prompt)


class HuggingFaceClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float = 0.7, 
                 max_tokens: int = 1000, device: str = "cpu", 
                 use_api: bool = False, api_token: Optional[str] = None,
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        super().__init__(model_name, temperature, max_tokens)
        self.use_api = use_api
        self.api_token = api_token
        
        if use_api:
            # ✅ Используем рабочий вариант с HuggingFaceEndpoint + ChatHuggingFace
            try:
                from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
                
                self.llm_endpoint = HuggingFaceEndpoint(
                    repo_id=model_name,
                    huggingfacehub_api_token=api_token,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                self.chat = ChatHuggingFace(llm=self.llm_endpoint)
                print(f"✅ Using HuggingFace API with model: {model_name}")
            except ImportError:
                raise ImportError("Please install langchain-huggingface: pip install langchain-huggingface")
        else:
            # Локальная загрузка модели
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=api_token
                )
                
                if load_in_8bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        token=api_token
                    )
                elif load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        token=api_token
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        token=api_token
                    )
                
                print(f"✅ Using local model on {device}: {model_name}")
                
            except ImportError as e:
                raise ImportError(f"Please install required packages: {e}")
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Вызов с промптом"""
        if self.use_api:
            # ✅ Используем ChatHuggingFace для вызова
            try:
                response = self.chat.invoke(prompt)
                return LLMResponse(
                    content=response.content,
                    raw_response=response,
                    model_name=self.model_name
                )
            except Exception as e:
                print(f"HuggingFace invoke error: {e}")
                raise
        else:
            # Локальный инференс
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Убираем промпт из ответа
            response = response[len(prompt):].strip()
            return LLMResponse(
                content=response,
                raw_response=response,
                model_name=self.model_name
            )
    
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Вызов с сообщениями"""
        if self.use_api:
            # ✅ Используем ChatHuggingFace для messages
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            response = self.chat.invoke(langchain_messages)
            return LLMResponse(
                content=response.content,
                raw_response=response,
                model_name=self.model_name
            )
        else:
            # Локальный инференс
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
            return self.invoke(prompt, **kwargs)
        
    def stream(self, prompt: str, **kwargs):
        """Стриминг ответа"""
        raise NotImplementedError("Streaming not implemented for HuggingFaceClient")


class LocalVLLMClient(BaseLLMClient):
    """Клиент для локальных моделей через vLLM (быстрый инференс)"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, 
                 max_tokens: int = 1000, tensor_parallel_size: int = 1):
        super().__init__(model_name, temperature, max_tokens)
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Вызов с промптом"""
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text
            return LLMResponse(
                content=response,
                raw_response=response,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"vLLM invoke error: {e}")
            raise
    
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Конвертируем messages в промпт"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return self.invoke(prompt, **kwargs)
    
    def stream(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming not supported in vLLM client")
    
class LMStudioClient(BaseLLMClient):
    """Клиент для LM Studio (локальный сервер с OpenAI-совместимым API)"""
    
    def __init__(self, model_name: str = "local-model", temperature: float = 0.7,
                 max_tokens: int = 1000, base_url: str = "http://127.0.0.1:1234/v1"):
        super().__init__(model_name, temperature, max_tokens)
        self.base_url = base_url
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url,
                api_key="not-needed"
            )
            print(f"✅ Connected to LM Studio at {base_url}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Вызов с промптом"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"LM Studio invoke error: {e}")
            raise
    
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Вызов с сообщениями"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"LM Studio messages error: {e}")
            raise
    
    def stream(self, prompt: str, **kwargs):
        """Стриминг ответа"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LM Studio stream error: {e}")
            raise