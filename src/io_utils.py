from __future__ import annotations

import json

from pydantic import BaseModel

from .models import FunctionCallResult, FunctionDefinition, TestCase


class IOUtils(BaseModel):
    functions_definition_path: str
    test_cases_path: str
    output_path: str

    def load_functions_definition(self) -> list[FunctionDefinition]:
        """Load function definitions from a JSON file.

        Returns:
            The validated function definitions.

        Raises:
            ValueError: If the file cannot be read, contains invalid JSON, or
                does not contain a list of objects.
        """
        try:
            with open(
                self.functions_definition_path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(f"file not found: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"broken json: {error}")
        except PermissionError as error:
            raise ValueError(f"Permission denied for file: {error}")
        except OSError as error:
            raise ValueError(f"File error: {error}")
        if not isinstance(data, list):
            raise ValueError(
                "functions definition must be a list of objects"
            )
        return [FunctionDefinition(**entry) for entry in data]

    def load_test_cases(self) -> list[TestCase]:
        """Load test cases from a JSON file.

        Returns:
            The validated function-calling test cases.

        Raises:
            ValueError: If the file cannot be read, contains invalid JSON, or
                does not contain a list of objects.
        """
        try:
            with open(
                self.test_cases_path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(f"file not found: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"broken json: {error}")
        except PermissionError as error:
            raise ValueError(f"Permission denied for file: {error}")
        except OSError as error:
            raise ValueError(f"File error: {error}")
        if not isinstance(data, list):
            raise ValueError("test cases must be a list of objects")
        return [TestCase(**entry) for entry in data]

    def write_output(self, results: list[FunctionCallResult]) -> None:
        """Write function-call results to the configured JSON file.

        Args:
            results: Function-call results to serialize.

        Raises:
            ValueError: If the output file cannot be written.
        """
        output = [result.model_dump() for result in results]
        try:
            with open(self.output_path, "w", encoding="utf-8") as file:
                json.dump(output, file, indent=4)
        except FileNotFoundError as error:
            raise ValueError(f"file not found: {error}")
        except PermissionError as error:
            raise ValueError(f"Permission denied for file: {error}")
        except OSError as error:
            raise ValueError(f"File error: {error}")
