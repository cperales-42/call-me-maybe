"""Pydantic models shared by the function-calling pipeline."""

from __future__ import annotations

from math import isfinite
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


SUPPORTED_PARAMETER_TYPES = frozenset(
    {"number", "integer", "string", "boolean"}
)


class TypeSpec(BaseModel):
    """Type descriptor used by function definitions."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that a type name is not empty.

        Args:
            value: Declared type name.

        Returns:
            The validated type name.

        Raises:
            ValueError: If the type name is empty.
        """
        if not value.strip():
            raise ValueError("type name must not be empty")
        return value


class FunctionDefinition(BaseModel):
    """One function that the model may call."""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate that the function name is not empty.

        Args:
            value: Declared function name.

        Returns:
            The validated function name.

        Raises:
            ValueError: If the function name is empty.
        """
        if not value.strip():
            raise ValueError("function name must not be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, TypeSpec],
    ) -> dict[str, TypeSpec]:
        """Validate parameter names and supported parameter types.

        Args:
            value: Declared parameter mapping.

        Returns:
            The validated parameter mapping.

        Raises:
            ValueError: If a parameter name is empty or its type is
                unsupported.
        """
        for name, specification in value.items():
            if not name:
                raise ValueError("parameter names must not be empty")
            if specification.type not in SUPPORTED_PARAMETER_TYPES:
                raise ValueError(
                    f"unsupported parameter type {specification.type!r} "
                    f"for parameter {name!r}; supported types are "
                    f"{sorted(SUPPORTED_PARAMETER_TYPES)}"
                )
        return value


class TestCase(BaseModel):
    """One natural-language prompt to translate into a function call."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate that a prompt contains non-whitespace text.

        Args:
            value: Natural-language prompt.

        Returns:
            The validated prompt.

        Raises:
            ValueError: If the prompt is empty or whitespace-only.
        """
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class FunctionCallResult(BaseModel):
    """Generated function call returned for one prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_result(self) -> FunctionCallResult:
        """Validate names and scalar parameter values.

        Returns:
            The validated result.

        Raises:
            ValueError: If the function name is empty, a parameter value is
                missing, non-finite, or not a JSON scalar.
        """
        if not self.name:
            raise ValueError("result name must not be empty")
        for parameter_name, value in self.parameters.items():
            if not parameter_name:
                raise ValueError("parameter names must not be empty")
            if value is None:
                raise ValueError(
                    f"parameter {parameter_name!r} has no value"
                )
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(
                    f"parameter {parameter_name!r} is not finite"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"parameter {parameter_name!r} has an unsupported "
                    f"value type: {type(value).__name__}"
                )
        return self


class ModelProtocol(BaseModel):
    """Public model interface required by the generation pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Compute the next-token logits for an input sequence.

        Args:
            input_ids: Token identifiers to use as model input.

        Returns:
            One logit for each token in the model vocabulary.
        """
        raise NotImplementedError

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the model vocabulary file.

        Returns:
            The vocabulary file path.
        """
        raise NotImplementedError

    def get_path_to_merges_file(self) -> str:
        """Return the path to the model byte-level BPE merges file.

        Returns:
            The merges file path.
        """
        raise NotImplementedError
