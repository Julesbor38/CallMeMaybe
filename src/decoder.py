from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from typing import Any


class ConstrainedDecoder:
    """Utilities used for constrained decoding."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm = llm

    def _encode(self, text: str) -> list[int]:
        """Encode a string."""
        return [int(token) for token in self.llm.encode(text)[0]]

    def _decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Decode token ids into a string."""
        return self.llm.decode(token_ids)

    def _next_logits(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return next-token logits."""
        return self.llm.get_logits_from_input_ids(input_ids)

    def allowed_tokens(
        self,
        candidates: list[list[int]],
        generated: list[int],
    ) -> set[int]:
        """Return every valid next token."""

        allowed: set[int] = set()

        for candidate in candidates:
            if candidate[: len(generated)] != generated:
                continue

            if len(candidate) > len(generated):
                allowed.add(candidate[len(generated)])

        return allowed

    def filter_candidates(
        self,
        candidates: list[list[int]],
        generated: list[int],
    ) -> list[list[int]]:
        """Keep only candidates matching the generated prefix."""

        return [
            candidate
            for candidate in candidates
            if candidate[: len(generated)] == generated
        ]

    def best_allowed_token(
        self,
        logits: list[float],
        allowed: set[int],
    ) -> int:
        """Return the highest-scoring allowed token."""

        if not allowed:
            raise RuntimeError("No allowed token.")

        return max(
            allowed,
            key=lambda token: logits[token],
        )

    def decode_choice(
        self,
        prompt: str,
        choices: list[str],
    ) -> str:
        """Decode one choice using constrained decoding."""

        encoded_choices = [
            self._encode(choice)
            for choice in choices
        ]

        generated: list[int] = []

        prompt_ids = self._encode(prompt)

        while True:
            # 1. Quels tokens sont encore possibles ?
            allowed = self.allowed_tokens(
                encoded_choices,
                generated,
            )

            if not allowed:
                raise RuntimeError("No valid continuation.")

            # 2. Logits du prochain token
            context = prompt_ids + generated
            logits = self._next_logits(context)

            # 3. Meilleur token parmi ceux autorisés
            next_token = self.best_allowed_token(
                logits,
                allowed,
            )

            generated.append(next_token)

            # 4. On élimine les candidats incompatibles
            encoded_choices = self.filter_candidates(
                encoded_choices,
                generated,
            )

            # 5. S'il ne reste qu'un seul candidat entièrement généré
            if (
                len(encoded_choices) == 1
                and generated == encoded_choices[0]
            ):
                return self._decode(generated)

    def select_function(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionDefinition:

        selected = self.decode_choice(
            prompt,
            [function.name for function in functions],
        )

        for function in functions:
            if function.name == selected:
                return function

        raise RuntimeError("Function not found.")

    def expected_parameters(
        self,
        function: FunctionDefinition,
    ) -> list[tuple[str, str]]:
        """
        Return the ordered list of expected parameters.

        Example:
            [("a", "number"), ("b", "number")]
        """

        return [
            (name, parameter.type)
            for name, parameter in function.parameters.items()
        ]

    def generate_arguments(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """
        Generate the arguments for a function.
        """

        return self.decode_json(
            prompt,
            function,
        )

    def build_json_template(self,
                            function: FunctionDefinition, ) -> dict[str, Any]:
        """
        Build an empty JSON object matching the function signature.
        """

        result: dict[str, Any] = {}

        for name, parameter in function.parameters.items():
            match parameter.type:
                case "number":
                    result[name] = None
                case "string":
                    result[name] = None
                case "boolean":
                    result[name] = None
                case _:
                    raise RuntimeError(f"Unsupported type: {parameter.type}")

        return result

    def decode_number(self, context: list[int],) -> tuple[float, list[int]]:
        """
        Decode a number and return both the decoded value and its tokens.
        """

        generated: list[int] = []

        digit_tokens = {
            self._encode(str(i))[0]
            for i in range(10)
        }

        allowed = digit_tokens | {
            self._encode("-")[0],
            self._encode(".")[0],
        }

        while True:

            logits = self._next_logits(
                context + generated,
            )

            next_token = max(
                allowed,
                key=lambda token: logits[token],
            )

            generated.append(next_token)

            logits = self._next_logits(
                context + generated,
            )

            best = max(
                range(len(logits)),
                key=lambda token: logits[token],
            )

            if best not in allowed:
                break

        value = float(self._decode(generated))

        return value, generated

    def decode_string(self, context: list[int],) -> tuple[str, list[int]]:
        """
        Decode a string and return both the value and its tokens.
        """

        generated: list[int] = []

        quote = self._encode('"')[0]

        generated.append(quote)

        while True:

            logits = self._next_logits(
                context + generated,
            )

            next_token = max(
                range(len(logits)),
                key=lambda token: logits[token],
            )

            generated.append(next_token)

            if next_token == quote:
                break

        value = self._decode(generated)[1:-1]

        return value, generated

    def decode_boolean(
        self,
        prompt: str,
    ) -> tuple[bool, list[int]]:
        """
        Decode a boolean and return both the value and its tokens.
        """

        value = self.decode_choice(
            prompt,
            [
                "true",
                "false",
            ],
        )

        return (
            value == "true",
            self._encode(value),
        )


    def decode_json(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """
        Decode a JSON object matching the selected function.
        """

        context = self._encode(prompt)

        result: dict[str, Any] = {}

        context += self._encode("{")

        parameters = list(function.parameters.items())

        for index, (name, parameter) in enumerate(parameters):

            context += self._encode(f'"{name}":')

            match parameter.type:

                case "number":
                    value, tokens = self.decode_number(context)

                case "string":
                    value, tokens = self.decode_string(context)

                case "boolean":
                    value, tokens = self.decode_boolean(prompt)

                case _:
                    raise RuntimeError(
                        f"Unsupported type: {parameter.type}"
                    )

            result[name] = value

            context += tokens

            if index < len(parameters) - 1:
                context += self._encode(",")

        context += self._encode("}")

        return result
