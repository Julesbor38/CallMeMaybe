from pydantic import BaseModel
from typing import Literal, Any


class Parameter(BaseModel):
    type: Literal["string", "number", "boolean"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    function: str
    arguments: dict[str, Any]
