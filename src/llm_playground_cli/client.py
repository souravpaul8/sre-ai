from ast import List
import json
from concurrent.futures import ThreadPoolExecutor
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
        max_iterations: int = 10,
        initial_contents: list | None = None,
        initial_tool_log: list | None = None,
        initial_seen_hashes: set | None = None,
        on_checkpoint: Callable[[list, list, set], None] | None = None,
    ) -> tuple[str, list[dict], list, set]:
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
            
        # ── Build initial conversation (incorporating history if resuming) ──
        if initial_contents:
            contents = list(initial_contents)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            )
        else:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

        tool_call_log = list(initial_tool_log) if initial_tool_log else []
        seen_call_hashes = set(initial_seen_hashes) if initial_seen_hashes else set()
        total_input_tokens = 0
        total_output_tokens = 0

        # Initial checkpoint
        if on_checkpoint:
            on_checkpoint(contents, tool_call_log, seen_call_hashes)

        for iteration in range(max_iterations):
            iter_start = time.time()
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=cfg,
            )
            # Track token usage per iteration
            if response.usage_metadata:
                iter_in = response.usage_metadata.prompt_token_count or 0
                iter_out = response.usage_metadata.candidates_token_count or 0
                total_input_tokens += iter_in
                total_output_tokens += iter_out
            else:
                iter_in = 0
                iter_out = 0

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
                final_answer = "".join(text_parts)
                if on_checkpoint:
                    on_checkpoint(contents, tool_call_log, seen_call_hashes)
                return final_answer, tool_call_log, contents, seen_call_hashes

            # ── Helper for executing a single tool call ─────────────
            def execute_single_tool(fc, step_num):
                tool_start = time.time()
                name = fc.name
                args = dict(fc.args) if fc.args else {}
                tool = tool_map.get(name)

                # 3. Cycle Detection: Check for identical repeated tool calls
                call_signature = json.dumps({"name": name, "args": args}, sort_keys=True)
                if call_signature in seen_call_hashes:
                    result_data = (
                        f"SYSTEM ERROR: You are stuck in a loop calling '{name}' with identical arguments "
                        f"{json.dumps(args)}. Stop and reconsider your strategy."
                    )
                    result = ToolResult(success=False, data="", error="Cycle detected")
                    elapsed = round(time.time() - tool_start, 3)
                    return name, args, result_data, result, False, elapsed, step_num

                seen_call_hashes.add(call_signature)

                # Guardrail: Check confirmation via on_tool_call callback
                allowed = True
                if on_tool_call:
                    callback_result = on_tool_call(name, args)
                    if callback_result is False:
                        allowed = False

                if not allowed:
                    result_data = f"Action cancelled: Execution of dangerous tool '{name}' was denied by user."
                    result = ToolResult(success=False, data="", error="Denied by user")
                elif tool:
                    try:
                        result = tool.execute(**args)
                        if result.success:
                            result_data = result.data
                        else:
                            # 7. System Reflection Injection (Self-Healing)
                            result_data = (
                                f"SYSTEM ERROR: Tool '{name}' failed with error: {result.error}. "
                                "Please inspect your arguments and try an alternative strategy or correct the missing inputs."
                            )
                    except TypeError as te:
                        # 7. Self-Healing for missing or invalid arguments
                        result_data = (
                            f"SYSTEM ERROR: Invalid arguments for '{name}': {te}. "
                            "Please adjust your parameters to match the required schema and try again."
                        )
                        result = ToolResult(success=False, data="", error=str(te))
                    except Exception as ex:
                        result_data = (
                            f"SYSTEM ERROR: Exception encountered in tool '{name}': {ex}. "
                            "Please correct your inputs and try again."
                        )
                        result = ToolResult(success=False, data="", error=str(ex))
                else:
                    available = list(tool_map.keys())
                    result_data = f"SYSTEM ERROR: Unknown tool '{name}'. Available tools are: {available}. Please call one of the available tools."
                    result = ToolResult(success=False, data="", error=f"Unknown tool: {name}")

                elapsed = round(time.time() - tool_start, 3)

                # 5. Indirect Prompt Injection Defense (Security XML tag wrapping)
                safe_wrapped_data = (
                    f'<tool_observation tool_name="{name}">\n'
                    f"[RAW UNTRUSTED DATA START]\n"
                    f"{result_data}\n"
                    f"[RAW UNTRUSTED DATA END]\n"
                    f"</tool_observation>"
                )

                return name, args, safe_wrapped_data, result, (tool is not None and allowed and result.success), elapsed, step_num

            # ── 4. Parallel Tool Execution ─────────────────────────
            current_base_step = len(tool_call_log) + 1
            if len(function_calls) > 1:
                with ThreadPoolExecutor(max_workers=min(len(function_calls), 8)) as executor:
                    tasks = [
                        (fc, current_base_step + i)
                        for i, fc in enumerate(function_calls)
                    ]
                    results = list(executor.map(lambda t: execute_single_tool(t[0], t[1]), tasks))
            else:
                results = [execute_single_tool(function_calls[0], current_base_step)]

            response_parts = []
            for name, args, safe_data, result, is_success, elapsed, step_num in results:
                # 6. Trajectory Tracing Log Entry
                tool_call_log.append({
                    "step": step_num,
                    "iteration": iteration + 1,
                    "name": name,
                    "args": args,
                    "success": is_success,
                    "execution_time_s": elapsed,
                    "input_tokens": iter_in,
                    "output_tokens": iter_out,
                    "result_preview": str(safe_data)[:200],
                })
                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": safe_data},
                    )
                )

            # ── Feed tool results back to the LLM ────────────────
            contents.append(
                types.Content(parts=response_parts)
            )

            # Checkpoint after tools execution step
            if on_checkpoint:
                on_checkpoint(contents, tool_call_log, seen_call_hashes)

        return "Error: Maximum tool call iterations reached.", tool_call_log, contents, seen_call_hashes


        