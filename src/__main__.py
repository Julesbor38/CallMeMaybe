import argparse
from pathlib import Path
from typing import Any
from .generator import FunctionGenerator
from .loader import load_function_definitions, load_prompts
from .prompt_builder import build_function_selection_prompt
from .writer import write_function_calls


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"))

    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"))

    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/function_calls.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompts(args.input)
    generator = FunctionGenerator()
    function_calls: list[Any] = []

    for prompt in prompts:
        selection_prompt = build_function_selection_prompt(
            prompt.prompt, functions)

        function_call = generator.generate_function_call(
            user_prompt=prompt.prompt,
            selection_prompt=selection_prompt, functions=functions)
        function_calls.append(function_call)
    write_function_calls(args.output, function_calls)


if __name__ == "__main__":
    main()
