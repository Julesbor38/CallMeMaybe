import json

from pathlib import Path
from typing import Any

from .models import FunctionDefinition, Prompt


def load_json(path: str | Path) -> Any:
    """
    Load and parse a JSON file.

    """

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as e:
        raise RuntimeError(f"'{path}' not found.") from e

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in file: {path}") from e

    except OSError as e:
        raise RuntimeError(f"Unable to read file: {path}") from e


def load_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load the list of available function definitions."""
    data = load_json(path)

    if not isinstance(data, list):
        raise RuntimeError("functions_definition.json"
                           " must contain a JSON array.")

    return [FunctionDefinition.model_validate(obj) for obj in data]


def load_prompts(path: str | Path) -> list[Prompt]:
    """Load the list of prompts."""
    data = load_json(path)

    if not isinstance(data, list):
        raise RuntimeError("function_calling_tests.json"
                           " must contain a JSON array.")

    return [Prompt.model_validate(obj) for obj in data]
