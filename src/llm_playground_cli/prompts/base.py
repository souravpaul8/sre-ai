import hashlib
from dataclasses import dataclass, field
from string import Template
from typing import Any

@dataclass
class PromptTemplate:
    name: str
    version: str
    system_template: str
    user_template: str
    required_variables: list[str]
    temperature: float = 0.2
    max_tokens: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **variables) -> dict:
        missing = set(self.required_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Template '{self.name}' missing variables: {missing}")
        
        return {
            "system": Template(self.system_template).safe_substitute(**variables),
            "user": Template(self.user_template).safe_substitute(**variables),
            "temperature": self.temperature,
        }
    
    @property
    def fingerprint(self) -> str:
        content = f"{self.system_template}{self.user_template}{self.version}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]