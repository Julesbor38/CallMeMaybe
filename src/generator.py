from typing import Any

from llm_sdk import Small_LLM_Model

from .decoder import ConstrainedDecoder
from .models import FunctionCall, FunctionDefinition
from .prompt_builder import build_arguments_prompt


class FunctionGenerator:

    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        self.decoder = ConstrainedDecoder(self.llm)

    def select_function(self, prompt: str,
                        functions: list[FunctionDefinition]
                        ) -> FunctionDefinition:
        """Select the function using constrained decoding."""

        return self.decoder.select_function(prompt, functions)

    def generate_arguments(self,
                           user_prompt: str,
                           function: FunctionDefinition) -> dict[str, Any]:
        """Generate the function arguments."""

        prompt = build_arguments_prompt(user_prompt, function)
        return self.decoder.generate_arguments(prompt, function)

    def generate_function_call(self, user_prompt: str,
                               functions: list[FunctionDefinition],
                               selection_prompt: str) -> FunctionCall:
        """Generate a complete function call."""

        function = self.decoder.select_function(
            selection_prompt, functions)
        arguments = self.decoder.generate_arguments(
            build_arguments_prompt(user_prompt, function), function)
        return FunctionCall(prompt=user_prompt,
                            function=function.name, arguments=arguments)
