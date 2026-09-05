from ast import List
from google import genai
from google.genai import types
from typing import Iterator, Type, TypeVar, Callable
from pydantic import BaseModel
from llm_playground_cli.tools.base import ToolResult
import time

from llm_playground_cli.config import get_api_key, GenerationConfig, get_rag_config

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
    
    def get_embedding(self, text: str) -> List[float]:
        """Convert a single text string into a vector."""
        response = self.client.models.embed_content(
            model=get_rag_config().embedding_model,
            contents=text
        )
        return response.embeddings[0].values
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Batch convert multiple texts into vectors (faster for ingestion)."""
        response = self.client.models.embed_content(
            model=get_rag_config().embedding_model,
            contents=texts
        )
        return [emb.values for emb in response.embeddings]

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
    
    def agent_loop(
        self,
        prompt: str,
        system: str,
        tools: list,
        on_tool_call: Callable | None = None, 
        max_iterations: int = 3,
    ) -> tuple[str, list[dict]]:
        # ── Build Gemini tool declarations ────────────────────────
        function_declarations = []
        tool_map = {}
        for tool in tools:
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
            tool_map[tool.name] = tool
            cfg = self._build_generate_config()
            cfg.tools = [types.Tool(function_declarations=function_declarations)]
            if system:
                cfg.system_instruction = system
            
        # ── Build initial conversation ────────────────────────────
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        tool_call_log = []

        for iteration in range(max_iterations):
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=cfg,
            )
            model_content = response.candidates[0].content
            contents.append(model_content)
            # Check if the LLM wants to call tools
            function_calls = [
                part.function_call
                for part in model_content.parts
                if part.function_call
            ]
            if not function_calls:
                # No tool calls — LLM gave a text answer. We're done.
                text_parts = [
                    part.text for part in model_content.parts if part.text
                ]
                return "".join(text_parts), tool_call_log
            # ── Execute each tool call ────────────────────────────
            response_parts = []
            for fc in function_calls:
                name = fc.name
                args = dict(fc.args) if fc.args else {}
                tool = tool_map.get(name)

                # Guardrail: Check confirmation via on_tool_call callback
                allowed = True
                if on_tool_call:
                    # If callback returns explicit False, the user denied execution
                    callback_result = on_tool_call(name, args)
                    if callback_result is False:
                        allowed = False

                # Look up and execute
                if not allowed:
                    result_data = f"Action cancelled: Execution of dangerous tool '{name}' was denied by user."
                    result = ToolResult(success=False, data="", error="Denied by user")
                elif tool:
                    result = tool.execute(**args)
                    result_data = result.data if result.success else f"Error: {result.error}"
                else:
                    result_data = f"Error: Unknown tool '{name}'"
                    result = ToolResult(success=False, data="", error=f"Unknown tool: {name}")

                tool_call_log.append({
                    "name": name,
                    "args": args,
                    "success": result.success if (tool and allowed) else False,
                    "result_preview": result_data[:200],
                })
                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result_data},
                    )
                )
            # ── Feed tool results back to the LLM ────────────────
            contents.append(
                types.Content(parts=response_parts)
            )
        return "Error: Maximum tool call iterations reached.", tool_call_log


        