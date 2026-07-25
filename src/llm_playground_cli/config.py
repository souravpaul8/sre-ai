from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class GenerationConfig:
    model: str
    temperature: float
    max_output_tokens: int
    top_p: float

def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example → .env and add your key from aistudio.google.com"
        )
    return key
    
def get_default_config() -> GenerationConfig:
    return GenerationConfig(
        model=os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite"),
        temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.3")),
        max_output_tokens=int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "2048")),
        top_p=float(os.getenv("DEFAULT_TOP_P", "0.95"))
    )

def get_router_config() -> GenerationConfig:
    return GenerationConfig(
        model=os.getenv("ROUTER_MODEL", "gemini-3.5-flash-lite"),
        temperature=0.0,     
        max_output_tokens=256,
        top_p=1.0,
    )