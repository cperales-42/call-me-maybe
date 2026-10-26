"""JSON input loading and output writing with explicit error handling."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import FunctionCallResult, FunctionDefinition, TestCase


def _read_json(path: str, label: str) -> Any:
    """Read a JSON file and translate filesystem errors into ValueError.

    Args:
        path: File to read.
        label: Human-readable file description used in error messages.

    Returns:
        The decoded JSON value.

    Raises:
        ValueError: If the file is missing, inaccessible, or invalid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"{label} file not found: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} file contains invalid JSON: {error}"
        ) from error
    except PermissionError as error:
        raise ValueError(
            f"permission denied for {label} file: {error}"
        ) from error
    except OSError as error:
        raise ValueError(f"could not read {label} file: {error}") from error


class IOUtils(BaseModel):
    """Paths and operations for the input and output JSON files."""

    functions_definition_path: str
    test_cases_path: str
    output_path: str

    def load_functions_definition(self) -> list[FunctionDefinition]:
        """Load and validate function definitions.

        Returns:
            The validated function definitions.

        Raises:
            ValueError: If the file cannot be read, is not a non-empty list,
                or contains invalid definitions.
        """
        data = _read_json(
            self.functions_definition_path,
            "function definitions",
        )
        if not isinstance(data, list):
            raise ValueError(
                "function definitions must be a JSON array"
            )
        if not data:
            raise ValueError(
                "function definitions must contain at least one function"
            )

        definitions: list[FunctionDefinition] = []
        seen_names: set[str] = set()
        for index, entry in enumerate(data):
            try:
                definition = FunctionDefinition.model_validate(entry)
            except ValidationError as error:
                raise ValueError(
                    f"invalid function definition at index {index}: {error}"
                ) from error
            if definition.name in seen_names:
                raise ValueError(
                    f"duplicate function name: {definition.name!r}"
                )
            seen_names.add(definition.name)
            definitions.append(definition)
        return definitions

    def load_test_cases(self) -> list[TestCase]:
        """Load and validate test cases.

        Returns:
            The validated test cases in input order.

        Raises:
            ValueError: If the file cannot be read, is not a list, or contains
                invalid prompts.
        """
        data = _read_json(self.test_cases_path, "test cases")
        if not isinstance(data, list):
            raise ValueError("test cases must be a JSON array")

        cases: list[TestCase] = []
        for index, entry in enumerate(data):
            try:
                cases.append(TestCase.model_validate(entry))
            except ValidationError as error:
                raise ValueError(
                    f"invalid test case at index {index}: {error}"
                ) from error
        return cases

    def write_output(self, results: list[FunctionCallResult]) -> None:
        """Write function-call results as an atomic JSON file.

        Args:
            results: Function-call results to serialize.

        Raises:
            ValueError: If the output directory cannot be created or the file
                cannot be written.
        """
        output = [
            result.model_dump(mode="json")
            for result in results
        ]
        parent = os.path.dirname(self.output_path)
        temporary_path = f"{self.output_path}.tmp"
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(output, file, indent=4, ensure_ascii=False)
                file.write("\n")
            os.replace(temporary_path, self.output_path)
        except PermissionError as error:
            raise ValueError(
                f"permission denied for output file: {error}"
            ) from error
        except OSError as error:
            raise ValueError(
                f"could not write output file: {error}"
            ) from error
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"could not serialize output: {error}"
            ) from error
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
