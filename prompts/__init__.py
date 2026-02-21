"""Load prompt text from files in this folder."""
from pathlib import Path


def get_prompts_dir() -> Path:
    """Return the prompts directory (same as this package)."""
    return Path(__file__).resolve().parent


def load_prompt(filename: str) -> str:
    """Load and return prompt text from prompts/<filename>. Raises FileNotFoundError if missing."""
    path = get_prompts_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
