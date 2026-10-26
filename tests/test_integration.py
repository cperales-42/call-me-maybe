"""End-to-end tests for the orchestrating entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import run

from .helpers import TargetModel, build_target_model


FUNCTIONS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
]


def _write_json(path: Path, payload: object) -> str:
    """Write a JSON payload and return its path.

    Args:
        path: Destination file.
        payload: JSON-serializable value.

    Returns:
        The stringified path.
    """
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _paths(tmp_path: Path) -> tuple[str, str, str]:
    """Create input and output paths for an integration run.

    Args:
        tmp_path: Temporary test directory.

    Returns:
        Function-definition, test-case, and output paths.
    """
    functions_path = _write_json(tmp_path / "functions.json", FUNCTIONS)
    cases_path = _write_json(
        tmp_path / "cases.json",
        [{"prompt": "What is the sum of 265 and 345?"}],
    )
    output_path = str(tmp_path / "out.json")
    return functions_path, cases_path, output_path


def _model(tmp_path: Path, target: str) -> TargetModel:
    """Build a deterministic model whose vocabulary covers the prompt.

    Args:
        tmp_path: Temporary test directory.
        target: JSON text the model should generate.

    Returns:
        The deterministic model.
    """
    return build_target_model(tmp_path, target)


def test_run_writes_valid_output(tmp_path: Path) -> None:
    functions_path, cases_path, output_path = _paths(tmp_path)
    model = _model(
        tmp_path,
        '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    )
    results = run(
        functions_path,
        cases_path,
        output_path,
        model=model,
        tokenizer=model.tokenizer,
    )
    assert len(results) == 1
    assert results[0].name == "fn_add_numbers"
    assert results[0].parameters == {"a": 265.0, "b": 345.0}
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert data == [
        {
            "prompt": "What is the sum of 265 and 345?",
            "name": "fn_add_numbers",
            "parameters": {"a": 265.0, "b": 345.0},
        }
    ]


def test_run_can_use_visualization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions_path, cases_path, output_path = _paths(tmp_path)
    model = _model(
        tmp_path,
        '{"name":"fn_greet","parameters":{"name":"shrek"}}',
    )
    cases_path = _write_json(
        tmp_path / "greet.json",
        [{"prompt": "Greet shrek"}],
    )
    run(
        functions_path,
        cases_path,
        output_path,
        visualize=True,
        model=model,
        tokenizer=model.tokenizer,
    )
    captured = capsys.readouterr()
    assert "[visualize] prompt: Greet shrek" in captured.out
    assert "[visualize] result:" in captured.out
    assert '"name": "fn_greet"' in captured.out


def test_cache_avoids_regeneration(tmp_path: Path) -> None:
    functions_path, cases_path, output_path = _paths(tmp_path)
    model = _model(
        tmp_path,
        '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    )
    cache_path = str(tmp_path / "cache.json")
    run(
        functions_path,
        cases_path,
        output_path,
        use_cache=True,
        model=model,
        tokenizer=model.tokenizer,
        cache_path=cache_path,
    )
    generated_calls = model.calls
    assert generated_calls > 1

    model.calls = 0
    run(
        functions_path,
        cases_path,
        output_path,
        use_cache=True,
        model=model,
        tokenizer=model.tokenizer,
        cache_path=cache_path,
    )
    # Only the one-token logits-size probe reaches the model.
    assert model.calls == 1


class _FailingModel(TargetModel):
    """Target model that fails after the logits-size probe."""

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Fail on every call except the initialization probe.

        Args:
            input_ids: Model input identifiers.

        Returns:
            Neutral logits for the first call.

        Raises:
            RuntimeError: On every call after the first.
        """
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("synthetic inference failure")
        return [0.0] * self.logits_size


def test_run_wraps_generation_errors(tmp_path: Path) -> None:
    functions_path, cases_path, output_path = _paths(tmp_path)
    model = _model(
        tmp_path,
        '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    )
    failing = _FailingModel(
        target=model.target,
        tokenizer=model.tokenizer,
        logits_size=model.logits_size,
        pieces=model.pieces,
    )
    with pytest.raises(RuntimeError, match="failed to generate a call"):
        run(
            functions_path,
            cases_path,
            output_path,
            model=failing,
            tokenizer=failing.tokenizer,
        )


def test_run_rejects_invalid_configuration(tmp_path: Path) -> None:
    functions_path, cases_path, output_path = _paths(tmp_path)
    model = _model(
        tmp_path,
        '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    )
    with pytest.raises(ValueError, match="max_iter"):
        run(
            functions_path,
            cases_path,
            output_path,
            model=model,
            tokenizer=model.tokenizer,
            max_iter=0,
        )
    with pytest.raises(ValueError, match="batch_size"):
        run(
            functions_path,
            cases_path,
            output_path,
            model=model,
            tokenizer=model.tokenizer,
            batch_size=0,
        )
