# ============================================
# Retry Mechanism
# ============================================

import time
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

class RetryParser:
    """Simple retry mechanism for parsing LLM responses"""
    
    def __init__(self, llm, parser, max_retries=3, delay=1, retry_temperature=0.3):
        self.llm = llm
        self._retry_llm = llm.bind(temperature=retry_temperature)
        self._parser = parser
        self.max_retries = max_retries
        self.delay = delay
    
    def parse(self, response_text: str, prompt_text: str = None) -> BaseModel:
        """Parse response with retry logic"""
        attempts = 0
        current_response = response_text
        
        while attempts < self.max_retries:
            try:
                return self._parser.parse(current_response)
            except Exception as e:
                attempts += 1
                print(f"⚠️ Critic retry {attempts}/{self.max_retries}: {e}")
                
                if attempts < self.max_retries and prompt_text:
                    wait_time = self.delay * (2 ** (attempts - 1))
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                    # Retry with lower temperature
                    retry_response = self._retry_llm.invoke(prompt_text)
                    current_response = retry_response.content if hasattr(retry_response, 'content') else str(retry_response)
                else:
                    raise OutputParserException(f"Failed to parse after {self.max_retries} attempts: {e}")
        
        raise OutputParserException("Failed to parse response")