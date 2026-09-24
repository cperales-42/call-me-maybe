from __future__ import annotations
import json

from .models import FunctionDefinition, TestCase, FunctionCallResult


class IOUtils(BaseModel):
    functions_definition_path: str
    test_cases_path: str
    output_path: str

    def load_functions_definition(self) -> list[FunctionDefinition]:
        try:
            with open(self.functions_definition_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as fnf:
            raise ValueError(f"file not found: {fnf}")
        except json.JSONDecodeError as je:
            raise ValueError(f"broken json: {je}")
        except PermissionError as pe:
            raise ValueError(f"Permission denied for file: {pe}")
        except OSError as e:
            raise ValueError(f"File error: {e}")
        if not isinstance(data, list):
            raise ValueError("functions definition must be a list of objects")
        return [FunctionDefinition(**entry) for entry in data]

    def load_test_cases(self) -> list[TestCase]:
        try:
            with open(self.test_cases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as fnf:
            raise ValueError(f"file not found: {fnf}")
        except json.JSONDecodeError as je:
            raise ValueError(f"broken json: {je}")
        except PermissionError as pe:
            raise ValueError(f"Permission denied for file: {pe}")
        except OSError as e:
            raise ValueError(f"File error: {e}")
        if not isinstance(data, list):
            raise ValueError("test cases must be a list of objects")
        return [TestCase(**entry) for entry in data]

    def write_output(self, results: list[FunctionCallResult]) -> None:
        output = [result.model_dump() for result in results]
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4)
        except FileNotFoundError as fnf:
            raise ValueError(f"file not found: {fnf}")
        except PermissionError as pe:
            raise ValueError(f"Permission denied for file: {pe}")
        except OSError as e:
            raise ValueError(f"File error: {e}")