"""Tests for request-level batching and input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.batch import BatchProcessor
from src.io_utils import IOUtils
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    TestCase as PromptCase,
    TypeSpec,
)


def _definition(name: str = "fn") -> dict[str, object]:
    """Build a raw function definition.

    Args:
        name: Function name.

    Returns:
        A JSON-compatible definition.
    """
    return {
        "name": name,
        "description": "A test function.",
        "parameters": {"a": {"type": "number"}},
        "returns": {"type": "number"},
    }


def test_chunk_preserves_order() -> None:
    processor = BatchProcessor(batch_size=2)
    assert processor.chunk([1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]
    assert processor.chunk([]) == []


def test_process_preserves_order() -> None:
    processor = BatchProcessor(batch_size=3)
    assert processor.process([1, 2, 3, 4], lambda item: item * 10) == [
        10,
        20,
        30,
        40,
    ]


def test_process_propagates_errors() -> None:
    processor = BatchProcessor(batch_size=1)

    def fail(item: int) -> int:
        raise RuntimeError(f"bad item {item}")

    with pytest.raises(RuntimeError, match="bad item 1"):
        processor.process([1, 2], fail)


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size(size: int) -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        BatchProcessor(batch_size=size)


def test_load_valid_input_files(tmp_path: Path) -> None:
    functions_path = tmp_path / "functions.json"
    tests_path = tmp_path / "tests.json"
    functions_path.write_text(json.dumps([_definition()]), encoding="utf-8")
    tests_path.write_text(json.dumps([{"prompt": "hello"}]), encoding="utf-8")
    utils = IOUtils(
        functions_definition_path=str(functions_path),
        test_cases_path=str(tests_path),
        output_path=str(tmp_path / "out.json"),
    )
    functions = utils.load_functions_definition()
    cases = utils.load_test_cases()
    assert functions[0].name == "fn"
    assert cases[0].prompt == "hello"


@pytest.mark.parametrize(
    "payload,message",
    [
        ("{}", "must be a JSON array"),
        ("[]", "at least one function"),
        ("[{}]", "invalid function definition"),
    ],
)
def test_invalid_function_files(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    functions_path = tmp_path / "functions.json"
    tests_path = tmp_path / "tests.json"
    functions_path.write_text(payload, encoding="utf-8")
    tests_path.write_text("[]", encoding="utf-8")
    utils = IOUtils(
        functions_definition_path=str(functions_path),
        test_cases_path=str(tests_path),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match=message):
        utils.load_functions_definition()


def test_duplicate_function_names_are_rejected(tmp_path: Path) -> None:
    functions_path = tmp_path / "functions.json"
    tests_path = tmp_path / "tests.json"
    functions_path.write_text(
        json.dumps([_definition(), _definition()]),
        encoding="utf-8",
    )
    tests_path.write_text("[]", encoding="utf-8")
    utils = IOUtils(
        functions_definition_path=str(functions_path),
        test_cases_path=str(tests_path),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match="duplicate function name"):
        utils.load_functions_definition()


@pytest.mark.parametrize(
    "payload,message",
    [
        ("{}", "must be a JSON array"),
        ("[{}]", "invalid test case"),
        ('[{"prompt": ""}]', "invalid test case"),
    ],
)
def test_invalid_test_case_files(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    functions_path = tmp_path / "functions.json"
    tests_path = tmp_path / "tests.json"
    functions_path.write_text(json.dumps([_definition()]), encoding="utf-8")
    tests_path.write_text(payload, encoding="utf-8")
    utils = IOUtils(
        functions_definition_path=str(functions_path),
        test_cases_path=str(tests_path),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match=message):
        utils.load_test_cases()


def test_missing_input_file_is_reported(tmp_path: Path) -> None:
    utils = IOUtils(
        functions_definition_path=str(tmp_path / "missing.json"),
        test_cases_path=str(tmp_path / "tests.json"),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match="not found"):
        utils.load_functions_definition()


def test_malformed_input_json_is_reported(tmp_path: Path) -> None:
    functions_path = tmp_path / "functions.json"
    functions_path.write_text("{", encoding="utf-8")
    utils = IOUtils(
        functions_definition_path=str(functions_path),
        test_cases_path=str(tmp_path / "tests.json"),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        utils.load_functions_definition()


def test_write_output_creates_directory_and_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "out.json"
    utils = IOUtils(
        functions_definition_path="unused",
        test_cases_path="unused",
        output_path=str(output_path),
    )
    result = FunctionCallResult(
        prompt="p",
        name="fn",
        parameters={"a": 2.0},
    )
    utils.write_output([result])
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == [
        {"prompt": "p", "name": "fn", "parameters": {"a": 2.0}}
    ]
    assert not (tmp_path / "nested" / "out.json.tmp").exists()


def test_write_output_reports_permission_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    utils = IOUtils(
        functions_definition_path="unused",
        test_cases_path="unused",
        output_path=str(blocker / "out.json"),
    )
    with pytest.raises(ValueError, match="could not write output"):
        utils.write_output([])


def test_definition_validation() -> None:
    with pytest.raises(ValidationError, match="function name"):
        FunctionDefinition.model_validate(
            {
                "name": " ",
                "description": "d",
                "parameters": {},
                "returns": {"type": "string"},
            }
        )
    with pytest.raises(ValidationError, match="unsupported parameter type"):
        FunctionDefinition.model_validate(
            {
                "name": "fn",
                "description": "d",
                "parameters": {"a": {"type": "object"}},
                "returns": {"type": "string"},
            }
        )


def test_type_spec_and_test_case_validation() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        TypeSpec(type=" ")
    with pytest.raises(ValidationError, match="prompt must not be empty"):
        PromptCase(prompt="   ")


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"a": None}, "has no value"),
        ({"a": float("nan")}, "not finite"),
        ({"a": [1]}, "unsupported value type"),
    ],
)
def test_result_validation(
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        FunctionCallResult(prompt="p", name="fn", parameters=parameters)


def test_result_rejects_empty_name() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        FunctionCallResult(prompt="p", name="", parameters={})
