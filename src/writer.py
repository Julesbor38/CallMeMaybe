from pathlib import Path
import json
from typing import Any
from .models import FunctionCall


def write_function_calls(
    path: str | Path,
    function_calls: list[FunctionCall],
) -> None:
    """Write function calls to a JSON file."""

    path = Path(path)

    # Create the output directory if needed
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data: list[dict[str, Any]] = [
        {
            "prompt": call.prompt,
            "name": call.function,
            "parameter": call.arguments,
        }
        for call in function_calls
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
