from typing import Literal
from pydantic import BaseModel, Field
from llm_playground_cli.client import LLMClient
from llm_playground_cli.config import get_router_config
from llm_playground_cli.prompts.loader import library 

CONFIDENCE_THRESHOLD = 0.6

class RoutingDecision(BaseModel):
    domain: str = Field(                   
        description="The classified domain name"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="One sentence explaining why this domain was chosen"
    )

def _build_router_prompt() -> str:
    """
    Builds the router system prompt dynamically from registered templates.
    When you add a new YAML file, the router automatically knows about it.
    """
    templates = library.list_templates()
    lines = [
        "You are a query classifier for a technical assistant.",
        "",
        "Given a user's question, classify it into exactly one domain:",
    ]
    for t in templates:
        keywords = t.get("description", t["name"])
        lines.append(f'- "{t["name"]}": {keywords}')
    lines.append('- "unknown": question doesn\'t fit any domain above')
    lines.append("")
    lines.append("Return your confidence (0.0–1.0) and a one-sentence reasoning.")
    return "\n".join(lines)

    

class QueryRouter:
    def __init__(self):
        # Router always uses the cheap model from env
        self.client = LLMClient(get_router_config())
        self._valid_domains = {t["name"] for t in library.list_templates()}
        self._system_prompt = _build_router_prompt()


    def classify(self, question: str) -> RoutingDecision | None:
        """
        Classify the question. Returns None if domain is 'unknown'
        or confidence is too low — caller should handle fallback.
        """
        prompt = f"Classify this question:\n\n{question}"
        result, _, _, _ = self.client.generate_structured(
            prompt=prompt,
            schema=RoutingDecision,
            system=self._system_prompt,
        )
        if result.domain not in self._valid_domains or result.confidence < CONFIDENCE_THRESHOLD:
            return None
        return result