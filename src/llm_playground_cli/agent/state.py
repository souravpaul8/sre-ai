import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from google.genai import types

STATE_FILE = os.path.expanduser("~/.sre-agent-state.json")

def serialize_content(content: types.Content) -> Dict[str, Any]:
    """Convert types.Content to a JSON-serializable dictionary."""
    parts_data = []
    for part in content.parts:
        if part.text is not None:
            parts_data.append({"type": "text", "text": part.text})
        elif part.function_call is not None:
            fc = part.function_call
            parts_data.append({
                "type": "function_call",
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {}
            })
        elif part.function_response is not None:
            fr = part.function_response
            parts_data.append({
                "type": "function_response",
                "name": fr.name,
                "response": dict(fr.response) if fr.response else {}
            })
    return {
        "role": content.role,
        "parts": parts_data
    }

def deserialize_content(data: Dict[str, Any]) -> types.Content:
    """Convert a JSON-serializable dictionary back to types.Content."""
    parts = []
    for p in data.get("parts", []):
        p_type = p.get("type")
        if p_type == "text":
            parts.append(types.Part.from_text(text=p.get("text", "")))
        elif p_type == "function_call":
            parts.append(types.Part.from_function_call(
                name=p.get("name", ""),
                args=p.get("args", {})
            ))
        elif p_type == "function_response":
            parts.append(types.Part.from_function_response(
                name=p.get("name", ""),
                response=p.get("response", {})
            ))
    return types.Content(
        role=data.get("role", "user"),
        parts=parts
    )

class StateManager:
    def __init__(self, filepath: str = STATE_FILE):
        self.filepath = filepath

    def exists(self) -> bool:
        return os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0

    def save(
        self,
        contents: List[types.Content],
        tool_call_log: List[Dict[str, Any]],
        seen_call_hashes: set
    ) -> None:
        """Atomically persist agent state to disk."""
        data = {
            "version": 1,
            "saved_at": time.time(),
            "contents": [serialize_content(c) for c in contents],
            "tool_call_log": tool_call_log,
            "seen_call_hashes": list(seen_call_hashes)
        }
        tmp_file = f"{self.filepath}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, self.filepath)
        except Exception:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def load(self) -> Optional[Tuple[List[types.Content], List[Dict[str, Any]], set]]:
        """Load state from disk. Returns (contents, tool_call_log, seen_call_hashes)."""
        if not self.exists():
            return None
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            contents = [deserialize_content(c) for c in data.get("contents", [])]
            tool_call_log = data.get("tool_call_log", [])
            seen_call_hashes = set(data.get("seen_call_hashes", []))
            return contents, tool_call_log, seen_call_hashes
        except Exception:
            return None

    def clear(self) -> None:
        """Remove state checkpoint file."""
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception:
                pass
