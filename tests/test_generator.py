"""Tests for the constrained generator and its fast path."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.generator import GenerationEvent, generate_prompt
from src.grammar import Grammar
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    ModelProtocol,
)
from src.tokenizer import SimpleTokenizer
from src.vocabulary import Vocabulary

from .helpers import (
    TargetModel,
    ascii_pieces,
    base_pieces,
    build_target_model,
    definition,
    make_vocabulary,
)


TWO_NUMBER_TARGET = (
    '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}'
)


def _run_target(
    tmp_path: Path,
    target: str,
    function: FunctionDefinition,
    pieces: list[str] | None = None,
) -> tuple[
    FunctionCallResult,
    TargetModel,
    list[GenerationEvent],
]:
    """Generate a target call and collect observer events.

    Args:
        tmp_path: Directory for the temporary vocabulary.
        target: JSON text the fake model should follow.
        function: Function definition to expose to the grammar.
        pieces: Optional vocabulary pieces.

    Returns:
        The result, the model, and the emitted events.
    """
    model = build_target_model(tmp_path, target, pieces)
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    grammar = Grammar(vocab=vocab, functions=[function])
    events: list[GenerationEvent] = []
    result = generate_prompt(
        model,
        grammar,
        "prompt",
        "prompt",
        512,
        tokenizer=model.tokenizer,
        observer=events.append,
    )
    return result, model, events


def test_generation_follows_target_and_preserves_numbers(
    tmp_path: Path,
) -> None:
    result, model, events = _run_target(
        tmp_path,
        TWO_NUMBER_TARGET,
        definition("fn_add_numbers", {"a": "number", "b": "number"}),
    )
    assert result.name == "fn_add_numbers"
    assert result.parameters == {"a": 265.0, "b": 345.0}
    assert result.prompt == "prompt"
    assert events
    assert any(event.source == "forced" for event in events)
    model_events = [event for event in events if event.source == "model"]
    assert len(model_events) == model.calls
    assert all(event.score is not None for event in model_events)
    assert all(event.allowed_tokens > 1 for event in model_events)
    assert all(
        event.allowed_tokens == 1
        for event in events
        if event.source == "forced"
    )


def test_forced_structural_tokens_do_not_query_model(
    tmp_path: Path,
) -> None:
    target = '{"name":"fn","parameters":{}}'
    function = definition("fn")
    pieces = ["{", "}", ":", ",", '"', "name", "parameters", "fn"]
    model = build_target_model(tmp_path, target, pieces)
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    grammar = Grammar(vocab, [function])
    events: list[GenerationEvent] = []
    result = generate_prompt(
        model,
        grammar,
        "prompt",
        "fn",
        512,
        tokenizer=model.tokenizer,
        observer=events.append,
    )
    assert result.parameters == {}
    assert model.calls == 0
    assert all(event.source == "forced" for event in events)
    assert result.name == "fn"


def test_singleton_mask_skips_model_call(tmp_path: Path) -> None:
    pieces = [
        "{",
        "}",
        ":",
        ",",
        '"',
        "name",
        "parameters",
        "fn",
        "a",
        "1",
        "2",
        "x",
    ]
    model = build_target_model(
        tmp_path,
        '{"name":"fn","parameters":{"a":1}}',
        pieces,
    )
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    grammar = Grammar(vocab, [definition("fn", {"a": "number"})])
    result = generate_prompt(
        model,
        grammar,
        "prompt",
        "x",
        512,
        tokenizer=model.tokenizer,
    )
    assert result.parameters == {"a": 1.0}
    # The model only chooses the digit and the closing delimiter.
    assert model.calls == 2


def test_integer_and_boolean_values(tmp_path: Path) -> None:
    target = '{"name":"fn","parameters":{"a":42,"b":true}}'
    result, _, _ = _run_target(
        tmp_path,
        target,
        definition("fn", {"a": "integer", "b": "boolean"}),
        ascii_pieces(),
    )
    assert result.parameters == {"a": 42, "b": True}


def test_string_value(tmp_path: Path) -> None:
    target = '{"name":"fn","parameters":{"a":"hello"}}'
    result, _, _ = _run_target(
        tmp_path,
        target,
        definition("fn", {"a": "string"}),
    )
    assert result.parameters == {"a": "hello"}


def test_generator_consumes_quote_spanning_token(tmp_path: Path) -> None:
    target = '{"name":"fn","parameters":{"a":"hello"}}'
    pieces = base_pieces() + ['"}}', "prompt"]
    result, _, events = _run_target(
        tmp_path,
        target,
        definition("fn", {"a": "string"}),
        pieces,
    )
    assert result.parameters == {"a": "hello"}
    assert events[-1].piece == '"}}'
    assert events[-1].source == "model"


def test_observer_event_reconstructs_target(tmp_path: Path) -> None:
    _, _, events = _run_target(
        tmp_path,
        TWO_NUMBER_TARGET,
        definition("fn_add_numbers", {"a": "number", "b": "number"}),
    )
    generated = "".join(event.piece for event in events)
    assert generated == TWO_NUMBER_TARGET


def test_generation_errors(tmp_path: Path) -> None:
    model = build_target_model(tmp_path, '{"name":"fn","parameters":{}}')
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    with pytest.raises(ValueError, match="max_iter must be positive"):
        generate_prompt(
            model,
            Grammar(vocab, [definition("fn")]),
            "prompt",
            "prompt",
            0,
        )
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_prompt(
            model,
            Grammar(vocab, [definition("fn")]),
            "",
            "prompt",
            10,
        )
    with pytest.raises(ValueError, match="no tokens"):
        generate_prompt(
            model,
            Grammar(vocab, [definition("fn")]),
            "prompt",
            "",
            10,
        )


def test_generation_rejects_completed_grammar(tmp_path: Path) -> None:
    model = build_target_model(tmp_path, '{"name":"fn","parameters":{}}')
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    grammar = Grammar(vocab, [definition("fn")])
    for piece in [
        "{",
        '"',
        "name",
        '"',
        ":",
        '"',
        "fn",
        '"',
        ",",
        '"',
        "parameters",
        '"',
        ":",
        "{",
        "}",
        "}",
    ]:
        grammar.advance(piece)
    with pytest.raises(ValueError, match="already completed"):
        generate_prompt(
            model,
            grammar,
            "prompt",
            "prompt",
            10,
        )


def test_generation_hits_token_limit(tmp_path: Path) -> None:
    model = build_target_model(tmp_path, '{"name":"fn","parameters":{}}')
    vocab = Vocabulary(
        vocab_path=model.tokenizer.vocab_path,
        logits_size=model.logits_size,
    )
    with pytest.raises(ValueError, match="did not finish"):
        generate_prompt(
            model,
            Grammar(vocab, [definition("fn")]),
            "prompt",
            "prompt",
            1,
            tokenizer=model.tokenizer,
        )


class _BadModel(ModelProtocol):
    """Model double that returns an invalid logit vector."""

    vocab_path: str
    size: int
    nan_only: bool = False

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return malformed logits for generator error tests.

        Args:
            input_ids: Model input identifiers.

        Returns:
            A short or NaN-filled logit vector.
        """
        if self.nan_only:
            return [float("nan")] * self.size
        return [0.0] * (self.size - 1)

    def get_path_to_vocab_file(self) -> str:
        """Return the vocabulary path.

        Returns:
            The configured vocabulary path.
        """
        return self.vocab_path

    def get_path_to_merges_file(self) -> str:
        """Return an empty merges path.

        Returns:
            An empty string.
        """
        return ""


def _bad_model(
    tmp_path: Path,
    *,
    nan_only: bool,
) -> tuple[_BadModel, Grammar, SimpleTokenizer]:
    """Build a bad model and a grammar with two allowed numeric tokens.

    Args:
        tmp_path: Directory for the temporary vocabulary.
        nan_only: Whether the model should return all NaN logits.

    Returns:
        The model and a grammar positioned at a numeric value.
    """
    vocab = make_vocabulary(
        tmp_path,
        ["{", "}", ":", ",", '"', "name", "parameters", "fn", "a", "1", "2"],
    )
    model = _BadModel(
        vocab_path=vocab.vocab_path,
        size=vocab.get_vocab_size(),
        nan_only=nan_only,
    )
    tokenizer = SimpleTokenizer(vocab.vocab_path)
    grammar = Grammar(vocab, [definition("fn", {"a": "number"})])
    for piece in [
        "{",
        '"',
        "name",
        '"',
        ":",
        '"',
        "fn",
        '"',
        ",",
        '"',
        "parameters",
        '"',
        ":",
        "{",
        '"',
        "a",
        '"',
        ":",
    ]:
        grammar.advance(piece)
    return model, grammar, tokenizer


def test_logits_length_mismatch_is_rejected(tmp_path: Path) -> None:
    model, grammar, tokenizer = _bad_model(tmp_path, nan_only=False)
    with pytest.raises(ValueError, match="logits length"):
        generate_prompt(
            model,
            grammar,
            "prompt",
            "1",
            10,
            tokenizer=tokenizer,
        )


def test_all_nan_logits_are_rejected(tmp_path: Path) -> None:
    model, grammar, tokenizer = _bad_model(tmp_path, nan_only=True)
    with pytest.raises(ValueError, match="no finite logit"):
        generate_prompt(
            model,
            grammar,
            "prompt",
            "1",
            10,
            tokenizer=tokenizer,
        )
