from pathlib import Path
import yaml
from llm_playground_cli.prompts.base import PromptTemplate
from llm_playground_cli.prompts.registry import PromptLibrary


def load_templates_from_dir(templates_dir: Path) -> PromptLibrary:

    library = PromptLibrary()
    for yaml_file in sorted(templates_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        template = PromptTemplate(
            name=data["name"],
            version=data["version"],
            system_template=data["system_template"].strip(),
            user_template=data["user_template"].strip(),
            required_variables=data.get("required_variables", []),
            temperature=data.get("temperature", 0.2),
            max_tokens=data.get("max_tokens", 2048),
            metadata=data.get("metadata", {}),
        )
        library.register(template)
    return library
    
# Singleton — built once when this module is first imported
TEMPLATES_DIR = Path(__file__).parent / "templates"
library = load_templates_from_dir(TEMPLATES_DIR)
