import os
from dotenv import load_dotenv, find_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv(find_dotenv(usecwd=True))

# Берем параметры модели из переменных окружения
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME_HF", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE_HF", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

# ============================================
# LLM Configuration для унифицированного интерфейса
# ============================================

# Выбор провайдера: "openai", "huggingface", "vllm"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

# Настройки для HuggingFace (используем существующие переменные)
HF_MODEL = os.getenv("MODEL_NAME_HF", DEFAULT_MODEL_NAME)
HF_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE_HF", DEFAULT_TEMPERATURE))
HF_MAX_TOKENS = int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS))
HF_USE_API = os.getenv("HF_USE_API", "True").lower() == "true"  # True = через API, False = локально
HF_DEVICE = os.getenv("HF_DEVICE", "cpu")

# Настройки для OpenAI (если понадобится)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Настройки для vLLM (если понадобится)
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", DEFAULT_TEMPERATURE))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
VLLM_TENSOR_PARALLEL = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))

# LM Studio настройки
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")
LMSTUDIO_TEMPERATURE = float(os.getenv("LMSTUDIO_TEMPERATURE", DEFAULT_TEMPERATURE))
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", DEFAULT_MAX_TOKENS))

# Общая конфигурация LLM
LLM_CONFIG = {
    "provider": LLM_PROVIDER,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "kwargs": {}
}


if LLM_PROVIDER == "openai":
    LLM_CONFIG["model_name"] = OPENAI_MODEL
    LLM_CONFIG["kwargs"]["api_key"] = OPENAI_API_KEY

elif LLM_PROVIDER == "huggingface":
    LLM_CONFIG["model_name"] = HF_MODEL
    LLM_CONFIG["temperature"] = HF_TEMPERATURE
    LLM_CONFIG["max_tokens"] = HF_MAX_TOKENS
    LLM_CONFIG["kwargs"]["use_api"] = HF_USE_API
    LLM_CONFIG["kwargs"]["device"] = HF_DEVICE
    LLM_CONFIG["kwargs"]["api_token"] = HUGGINGFACE_TOKEN

elif LLM_PROVIDER == "vllm":
    LLM_CONFIG["model_name"] = VLLM_MODEL
    LLM_CONFIG["temperature"] = VLLM_TEMPERATURE
    LLM_CONFIG["max_tokens"] = VLLM_MAX_TOKENS
    LLM_CONFIG["kwargs"]["tensor_parallel_size"] = VLLM_TENSOR_PARALLEL

elif LLM_PROVIDER == "lmstudio":
    LLM_CONFIG["model_name"] = LMSTUDIO_MODEL
    LLM_CONFIG["temperature"] = LMSTUDIO_TEMPERATURE
    LLM_CONFIG["max_tokens"] = LMSTUDIO_MAX_TOKENS
    LLM_CONFIG["kwargs"]["base_url"] = LMSTUDIO_BASE_URL