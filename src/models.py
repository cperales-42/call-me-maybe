from typing import Any

from pydantic import BaseModel


class TypeSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class TestCase(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
