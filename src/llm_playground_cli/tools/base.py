from os import error
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ToolResult:
        success: bool
        data: str
        error: str | None = None

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable[..., ToolResult],
        is_dangerous: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.is_dangerous = is_dangerous

    def execute(self, **kwargs) -> ToolResult:
        try:
            return self.function(**kwargs)
        except TypeError as e:
            return ToolResult(
                success=False,
                data="",
                error=f"Invalid arguments for '{self.name}': {e}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data="",
                error=f"Tool '{self.name}' failed: {e}",
            )


    def to_gemini_declaration(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }