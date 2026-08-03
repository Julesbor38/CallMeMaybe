from .models import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt used to select the most appropriate function."""

    lines = [
        "You are a function selection assistant.",
        "",
        "Available functions:",
        "",
    ]

    for function in functions:
        lines.append(f"Name: {function.name}")
        lines.append(f"Description: {function.description}")
        lines.append("Parameters:")

        for name, parameter in function.parameters.items():
            lines.append(f"- {name}: {parameter.type}")

        lines.append("")

    lines.extend(
        [
            "User request:",
            user_prompt,
            "",
            "Respond ONLY with one of the following function names:",
        ]
    )
    for function in functions:
        lines.append(f"- {function.name}")
    lines.extend(["", "Do not output anything else.",])

    return "\n".join(lines)


def build_arguments_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build the prompt used to generate the function arguments."""

    lines = [
        "You are an AI assistant that extracts function arguments.",
        "",
        f"Function: {function.name}",
        "",
        f"Description: {function.description}",
        "",
        "Parameters:",
    ]

    for name, parameter in function.parameters.items():
        lines.append(f"- {name} ({parameter.type})")

    lines.extend(
        [
            "",
            "Expected JSON format:",
            "{",
        ]
    )

    parameters = list(function.parameters.items())

    for index, (name, parameter) in enumerate(parameters):
        comma = "," if index < len(parameters) - 1 else ""
        lines.append(f'    "{name}": <{parameter.type}>{comma}')

    lines.extend(
        [
            "}",
            "",
            "User request:",
            user_prompt,
            "",
            "Return ONLY a valid JSON object.",
            "Do not output any text before or after the JSON.",
        ]
    )

    return "\n".join(lines)
