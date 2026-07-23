from google import genai
from google.genai import types
from typing import Iterator, Type, TypeVar
from pydantic import BaseModel
import time

from llm_playground_cli.config import get_api_key, GenerationConfig

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = genai.Client(api_key=get_api_key())

    def _build_generate_config(self, override_temp: float | None=None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=override_temp if 
            override_temp is not None else 
            self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p
        )
    
    def count_tokens(self, prompt: str) -> int:
        response=self.client.models.count_tokens(
            model=self.config.model,
            contents=prompt
        )
        return response.total_tokens

    def stream(
        self,
        prompt: str,
        system: str = "",
        override_temp: float | None = None,
    ) -> Iterator[str]:

        cfg = self._build_generate_config(override_temp)
        contents = []
        if system:
            cfg.system_instruction = system
        contents.append(prompt)

        for chunk in self.client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=cfg
        ):
            if chunk.text:
                yield chunk.text
    
    def generate(
        self,
        prompt: str = "",
        system: str = "",
        override_temp: float | None = None,
    ) -> tuple[str, int, int]:

        cfg = self._build_generate_config(override_temp)
        if system:
            cfg.system_instruction = system
        start = time.time()
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[prompt],
            config=cfg,
        )
        latency = round(time.time() - start, 2)
        text = response.text
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        return text, input_tokens, output_tokens, latency

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str = "",
        override_temp: float | None = None,
    ) -> tuple[T, int, int, float]:

        cfg = self._build_generate_config(override_temp)
        cfg.response_mime_type = "application/json"
        cfg.response_schema = schema
        if system:
            cfg.system_instruction = system
        
        start = time.time()
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[prompt],
            config=cfg
        )
        latency = round(time.time() - start, 2)
        
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0

        instance = schema.model_validate_json(response.text)
        return instance, input_tokens, output_tokens, latency