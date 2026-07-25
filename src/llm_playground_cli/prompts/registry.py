import time
import logging
from llm_playground_cli.prompts.base import PromptTemplate

logger = logging.getLogger(__name__)

class PromptLibrary:
    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._usage_log: list[dict] = []
    
    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template
        logger.debug(f"Registered: {template.name} v{template.version} [{template.fingerprint}]")

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            available = list(self._templates.keys())
            raise KeyError(f"Template '{name}' not found. Available: {available}")
        return self._templates[name]

    def render(self, name: str, **variables) -> dict:
        """Get a template, render it, and log the usage."""
        template = self.get(name)
        rendered = template.render(**variables)
        self._usage_log.append({
            "template": name,
            "version": template.version,
            "timestamp": time.time(),
        })
        return rendered

    def list_templates(self) -> list[dict]:
        """Returns metadata for all registered templates — used in the menu."""
        return [
            {
                "name": t.name,
                "version": t.version,
                "description": t.metadata.get("description", ""),
            }
            for t in self._templates.values()
        ]