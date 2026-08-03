import argparse
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .decoder import ConstrainedDecoder
from .loader import load_function_definitions, load_prompts
from .prompt_builder import (
    build_function_selection_prompt,
    build_arguments_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    functions = load_function_definitions(
        args.functions_definition,
    )

    prompts = load_prompts(
        args.input,
    )

    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(llm)

    for prompt in prompts:

        print("=" * 80)
        print(f"PROMPT : {prompt.prompt}")
        print("=" * 80)

        selection_prompt = build_function_selection_prompt(
            prompt.prompt,
            functions,
        )

        function = decoder.select_function(
            selection_prompt,
            functions,
        )

        print(f"Function : {function.name}")

        arguments_prompt = build_arguments_prompt(
            prompt.prompt,
            function,
        )

        try:
            arguments = decoder.decode_json(
                arguments_prompt,
                function,
            )

            print(f"Arguments : {arguments}")

        except Exception as e:
            print(f"JSON decoding failed : {e}")

        print()


if __name__ == "__main__":
    main()
