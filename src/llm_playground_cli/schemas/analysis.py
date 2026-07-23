from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

# ── 1. For the `explain` command (YAML/config file analysis) ──────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FileAnalysis(BaseModel):
    summary: str = Field(
        description="One-sentence summary of what this file does"
    )
    issues: list[str] = Field(
        description="List of problems, anti-patterns, or missing best practices found"
    )
    recommendations: list[str] = Field(
        description="Specific, actionable fixes for each issue"
    )
    risk_level: RiskLevel = Field(
        description="Overall risk level: low, medium, or high"
    )
    file_type: str = Field(
        description="Type of file, e.g. 'Kubernetes Deployment', 'Docker Compose', 'Shell script'"
    )

# ── 2. For the `convert` command (shell commands → documented YAML/Markdown) ──

class CommandDoc(BaseModel):
    command: str = Field(
        description="The original shell command"
        )
    description: str = Field(
        description="Plain-English explanation of what this command does"
        )
    flags: list[str] = Field(
        description="Explanation of each flag used, e.g. '-l: long listing format'"
        )
    example_output: str = Field(
        description="Example of what the command typically outputs"
        )
    category: str = Field(
        description="Category like 'file management', 'networking', 'process control'"
        )


class ShellDocumentation(BaseModel):
    title: str = Field(
        description="A short title for this collection of commands"
        )
    commands: list[CommandDoc] = Field(
        description="List of documented commands"
        )
    prerequisites: list[str] = Field(
        description="Tools or permissions required to run these commands"
        )
