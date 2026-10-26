"""Tests for the schema-aware finite-state grammar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.grammar import Grammar
from src.models import FunctionDefinition

from .helpers import base_pieces, definition, make_vocabulary


def _grammar(tmp_path: Path, functions: list[FunctionDefinition]) -> Grammar:
    """Build a grammar from the shared test vocabulary.

    Args:
        tmp_path: Directory for the temporary vocabulary.
        functions: Function definitions available to the grammar.

    Returns:
        A grammar in its initial state.
    """
    return Grammar(make_vocabulary(tmp_path, base_pieces()), functions)


def _advance_all(grammar: Grammar, pieces: list[str]) -> None:
    """Advance a grammar through every supplied piece.

    Args:
        grammar: Grammar to advance.
        pieces: Decoded token pieces in generation order.
    """
    for piece in pieces:
        grammar.advance(piece)


def _prefix(name: str, first_parameter: str) -> list[str]:
    """Return the JSON pieces before the first parameter value.

    Args:
        name: Selected function name.
        first_parameter: First parameter key.

    Returns:
        Pieces that open the call and reach the value state.
    """
    return [
        "{",
        '"',
        "name",
        '"',
        ":",
        '"',
        name,
        '"',
        ",",
        '"',
        "parameters",
        '"',
        ":",
        "{",
        '"',
        first_parameter,
        '"',
        ":",
    ]


def test_zero_parameter_function_closes_both_objects(
    tmp_path: Path,
) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    pieces = [
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
    ]
    _advance_all(grammar, pieces)
    assert grammar.is_done()
    assert grammar.name() == "fn"
    assert grammar.parameters() == {}


def test_multi_token_number_is_not_closed_early(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(
        grammar,
        _prefix("fn", "a") + ["2", "6", "5", "}", "}"],
    )
    assert grammar.is_done()
    assert grammar.parameters() == {"a": 265.0}


def test_number_can_continue_after_it_is_syntactically_complete(
    tmp_path: Path,
) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(grammar, _prefix("fn", "a"))
    grammar.advance("2")
    assert grammar.forced_next_token_id() is None
    mask = grammar.mask_for_next()
    delimiter = grammar.vocab.id_of("}")
    assert delimiter is not None and mask[delimiter]
    for digit in "3456789":
        digit_id = grammar.vocab.id_of(digit)
        assert digit_id is not None and mask[digit_id]
    grammar.advance("6")
    grammar.advance("5")
    grammar.advance("}")
    grammar.advance("}")
    assert grammar.parameters() == {"a": 265.0}


def test_integer_parameter_returns_python_int(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "integer"})])
    _advance_all(grammar, _prefix("fn", "a") + ["4", "2", "}", "}"])
    value = grammar.parameters()["a"]
    assert value == 42
    assert isinstance(value, int)


def test_negative_decimal_exponent_number(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(
        grammar,
        _prefix("fn", "a") + ["-", "1", ".", "5", "e", "2", "}", "}"],
    )
    assert grammar.parameters() == {"a": -150.0}


def test_boolean_parameters(tmp_path: Path) -> None:
    grammar = _grammar(
        tmp_path,
        [definition("fn", {"flag": "boolean"})],
    )
    _advance_all(grammar, _prefix("fn", "flag") + ["true", "}", "}"])
    assert grammar.parameters() == {"flag": True}

    false_grammar = _grammar(
        tmp_path,
        [definition("fn", {"flag": "boolean"})],
    )
    _advance_all(false_grammar, _prefix("fn", "flag") + ["false", "}", "}"])
    assert false_grammar.parameters() == {"flag": False}


def test_parameter_named_name_is_not_treated_as_top_level_key(
    tmp_path: Path,
) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"name": "string"})])
    _advance_all(
        grammar,
        _prefix("fn", "name") + ['"', "hello", '"', "}", "}"],
    )
    assert grammar.parameters() == {"name": "hello"}


def test_string_value_with_multiple_pieces(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "string"})])
    _advance_all(
        grammar,
        _prefix("fn", "a") + ['"', "hello", " ", "hello", '"', "}", "}"],
    )
    assert grammar.parameters() == {"a": "hello hello"}


def test_two_parameters_are_separated_by_comma(tmp_path: Path) -> None:
    grammar = _grammar(
        tmp_path,
        [definition("fn", {"a": "number", "b": "number"})],
    )
    _advance_all(
        grammar,
        _prefix("fn", "a")
        + ["2", ",", '"', "b", '"', ":", "3", "}", "}"],
    )
    assert grammar.parameters() == {"a": 2.0, "b": 3.0}


def test_string_mask_allows_quote_spanning_token(tmp_path: Path) -> None:
    pieces = base_pieces() + [']",', '"}']
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "string", "b": "string"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ['"', "hello"])
    mask = grammar.mask_for_next()
    spanning_id = grammar.vocab.id_of(']",')
    premature_close_id = grammar.vocab.id_of('"}')
    assert spanning_id is not None and mask[spanning_id]
    assert premature_close_id is not None and not mask[premature_close_id]
    grammar.advance(']",')
    assert grammar.get_state() == "key"
    assert grammar.forced_next_piece() == '"'


def test_token_can_close_string_and_both_objects(tmp_path: Path) -> None:
    pieces = base_pieces() + ['"}} \n']
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "string"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ['"', "hello"])
    mask = grammar.mask_for_next()
    closing_id = grammar.vocab.id_of('"}} \n')
    assert closing_id is not None and mask[closing_id]
    grammar.advance('"}} \n')
    assert grammar.is_done()
    assert grammar.parameters() == {"a": "hello"}


def test_multi_character_token_is_atomic_on_failure(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "string"})])
    _advance_all(grammar, _prefix("fn", "a") + ['"', "hello"])
    with pytest.raises(ValueError, match="after parsing completed"):
        grammar.advance('"}}x')
    assert grammar.get_state() == "value_str"
    assert grammar.parameters() == {}
    grammar.advance('"}}')
    assert grammar.is_done()
    assert grammar.parameters() == {"a": "hello"}


def test_number_and_delimiters_can_share_one_token(tmp_path: Path) -> None:
    grammar = _grammar(
        tmp_path,
        [definition("fn", {"a": "number", "b": "number"})],
    )
    _advance_all(grammar, _prefix("fn", "a"))
    grammar.advance("265,")
    assert grammar.get_state() == "key"
    grammar.advance('"b":345}}')
    assert grammar.is_done()
    assert grammar.parameters() == {"a": 265.0, "b": 345.0}


def test_forced_tokens_follow_json_structure(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    assert grammar.forced_next_piece() == "{"
    grammar.advance("{")
    assert grammar.forced_next_piece() == '"'
    grammar.advance('"')
    assert grammar.forced_next_token_id() is None
    grammar.advance("name")
    assert grammar.forced_next_piece() == '"'
    grammar.advance('"')
    assert grammar.forced_next_piece() == ":"
    grammar.advance(":")
    assert grammar.forced_next_piece() == '"'
    grammar.advance('"')
    grammar.advance("fn")
    assert grammar.forced_next_piece() == '"'
    grammar.advance('"')
    assert grammar.forced_next_piece() == ","
    grammar.advance(",")
    assert grammar.forced_next_piece() == '"'
    grammar.advance('"')
    grammar.advance("parameters")
    grammar.advance('"')
    assert grammar.forced_next_piece() == ":"
    grammar.advance(":")
    assert grammar.forced_next_piece() == "{"
    grammar.advance("{")
    grammar.advance('"')
    grammar.advance("a")
    grammar.advance('"')
    grammar.advance(":")
    assert grammar.forced_next_token_id() is None
    grammar.advance("1")
    assert grammar.forced_next_token_id() is None
    grammar.advance("}")
    assert grammar.forced_next_piece() == "}"
    grammar.advance("}")
    assert grammar.is_done()
    assert grammar.forced_next_token_id() is None


def test_forced_closing_quote_for_single_function_name(
    tmp_path: Path,
) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    _advance_all(grammar, ["{", '"', "name", '"', ":", '"'])
    grammar.advance("fn")
    assert grammar.forced_next_piece() == '"'


def test_ambiguous_prefix_does_not_force_name(tmp_path: Path) -> None:
    pieces = base_pieces() + ["_long"]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn"), definition("fn_long")],
    )
    _advance_all(grammar, ["{", '"', "name", '"', ":", '"'])
    grammar.advance("fn")
    assert grammar.forced_next_token_id() is None
    mask = grammar.mask_for_next()
    quote_id = grammar.vocab.id_of('"')
    long_id = grammar.vocab.id_of("_long")
    assert quote_id is not None and mask[quote_id]
    assert long_id is not None and mask[long_id]
    grammar.advance("_long")
    assert grammar.forced_next_piece() == '"'


def test_name_prefix_tokens_are_allowed(tmp_path: Path) -> None:
    pieces = base_pieces() + ["f", "n", "_", "l", "o", "g", "_long"]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn"), definition("fn_long")],
    )
    _advance_all(grammar, ["{", '"', "name", '"', ":", '"'])
    grammar.advance("f")
    mask = grammar.mask_for_next()
    initial_id = grammar.vocab.id_of("n")
    assert initial_id is not None and mask[initial_id]
    grammar.advance("n")
    mask = grammar.mask_for_next()
    for piece in ["_long", '"']:
        token_id = grammar.vocab.id_of(piece)
        assert token_id is not None and mask[token_id]


def test_string_mask_excludes_quotes_backslashes_and_controls(
    tmp_path: Path,
) -> None:
    pieces = base_pieces() + ["safe", "line\nbreak"]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "string"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ['"'])
    mask = grammar.mask_for_next()
    safe_id = grammar.vocab.id_of("safe")
    quote_id = grammar.vocab.id_of('"')
    newline_id = grammar.vocab.id_of("line\nbreak")
    assert safe_id is not None and mask[safe_id]
    assert quote_id is not None and mask[quote_id]
    assert newline_id is not None and not mask[newline_id]


def test_string_length_limit_forces_closing_quote(
    tmp_path: Path,
) -> None:
    long_piece = "z" * 256
    pieces = base_pieces() + [long_piece]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "string"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ['"'])
    grammar.advance(long_piece)
    assert grammar.forced_next_piece() == '"'
    with pytest.raises(ValueError, match="maximum length"):
        grammar.advance("x")


def test_number_digit_budget_is_enforced(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(grammar, _prefix("fn", "a"))
    for _ in range(15):
        grammar.advance("1")
    with pytest.raises(ValueError, match="maximum digit count"):
        grammar.advance("1")


def test_non_finite_number_is_rejected(tmp_path: Path) -> None:
    pieces = base_pieces() + ["999"]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "number"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ["1", "e"])
    with pytest.raises(ValueError, match="not finite"):
        grammar.advance("999")


def test_integer_rejects_fraction_and_exponent(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "integer"})])
    _advance_all(grammar, _prefix("fn", "a") + ["1"])
    for piece in [".", "e", "E"]:
        with pytest.raises(ValueError, match="invalid numeric"):
            grammar.advance(piece)


def test_invalid_numeric_continuation(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(grammar, _prefix("fn", "a") + ["0"])
    with pytest.raises(ValueError, match="invalid numeric"):
        grammar.advance("1")


def test_incomplete_number_cannot_close(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(grammar, _prefix("fn", "a") + ["-"])
    with pytest.raises(ValueError, match="incomplete number"):
        grammar.advance("}")


def test_parameter_cannot_be_closed_before_all_values_exist(
    tmp_path: Path,
) -> None:
    grammar = _grammar(
        tmp_path,
        [definition("fn", {"a": "boolean", "b": "boolean"})],
    )
    _advance_all(grammar, _prefix("fn", "a") + ["true"])
    with pytest.raises(ValueError, match="before all values exist"):
        grammar.advance("}")


def test_comma_after_final_parameter_is_rejected(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "number"})])
    _advance_all(grammar, _prefix("fn", "a") + ["1"])
    with pytest.raises(ValueError, match="after the final parameter"):
        grammar.advance(",")


def test_early_key_close_is_rejected(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    _advance_all(grammar, ["{", '"'])
    with pytest.raises(ValueError, match="does not continue key"):
        grammar.advance('"')


def test_wrong_key_piece_is_rejected(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    _advance_all(grammar, ["{"])
    with pytest.raises(ValueError, match="opening quote"):
        grammar.advance("x")
    grammar.advance('"')
    with pytest.raises(ValueError, match="does not continue key"):
        grammar.advance("x")


def test_wrong_function_name_is_rejected(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn_long")])
    _advance_all(grammar, ["{", '"', "name", '"', ":", '"'])
    with pytest.raises(ValueError, match="does not continue any function"):
        grammar.advance("x")
    grammar.advance("fn")
    with pytest.raises(ValueError, match="unknown function"):
        grammar.advance('"')


def test_function_name_must_start_with_quote(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    _advance_all(grammar, ["{", '"', "name", '"', ":"])
    with pytest.raises(ValueError, match="opening quote for function"):
        grammar.advance("fn")


def test_invalid_string_content_is_rejected(tmp_path: Path) -> None:
    pieces = base_pieces() + ["bad\\value", "bad\"value"]
    grammar = Grammar(
        make_vocabulary(tmp_path, pieces),
        [definition("fn", {"a": "string"})],
    )
    _advance_all(grammar, _prefix("fn", "a"))
    with pytest.raises(ValueError, match="must start with a quote"):
        grammar.advance("x")
    grammar.advance('"')
    with pytest.raises(ValueError, match="invalid string piece"):
        grammar.advance("bad\\value")


def test_invalid_boolean_continuation(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "boolean"})])
    _advance_all(grammar, _prefix("fn", "a") + ["t"])
    with pytest.raises(ValueError, match="invalid boolean"):
        grammar.advance("x")


def test_done_grammar_rejects_further_operations(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    _advance_all(
        grammar,
        [
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
        ],
    )
    with pytest.raises(ValueError, match="after parsing completed"):
        grammar.advance("}")
    with pytest.raises(ValueError, match="already done"):
        grammar.mask_for_next()


def test_get_name_before_completion_is_rejected(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    with pytest.raises(ValueError, match="before parsing completed"):
        grammar.name()


def test_mask_is_boolean_and_sized(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    mask = grammar.mask_for_next()
    assert mask.shape == (grammar.vocab.get_vocab_size(),)
    assert mask.dtype == bool
    assert np.count_nonzero(mask) == 1


@pytest.mark.parametrize(
    "functions,message",
    [
        ([], "at least one function"),
        (
            [definition("fn"), definition("fn")],
            "duplicate function definition",
        ),
    ],
)
def test_invalid_function_lists(
    tmp_path: Path,
    functions: list[FunctionDefinition],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _grammar(tmp_path, functions)


def test_unsupported_parameter_type_is_rejected(tmp_path: Path) -> None:
    function = definition("fn")
    function.parameters["a"] = function.returns.model_copy(
        update={"type": "object"}
    )
    with pytest.raises(ValueError, match="unsupported parameter type"):
        _grammar(tmp_path, [function])


def test_quoted_function_name_is_rejected(tmp_path: Path) -> None:
    function = definition('fn"quote')
    with pytest.raises(ValueError, match="quotes or backslashes"):
        _grammar(tmp_path, [function])


def test_empty_parameter_name_is_rejected(tmp_path: Path) -> None:
    function = definition("fn")
    function.parameters[""] = function.returns.model_copy()
    with pytest.raises(ValueError, match="empty parameter name"):
        _grammar(tmp_path, [function])


def test_control_character_in_function_name_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="control characters"):
        _grammar(tmp_path, [definition("fn\nembedded")])


def test_control_character_in_parameter_name_is_rejected(
    tmp_path: Path,
) -> None:
    function = definition("fn")
    function.parameters["a\tb"] = function.returns.model_copy()
    with pytest.raises(ValueError, match="control characters"):
        _grammar(tmp_path, [function])


def test_unexpressible_function_name_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot express required text"):
        Grammar(
            make_vocabulary(tmp_path, ["{", "}", ":", ",", '"', "name"]),
            [definition("fn")],
        )


def test_invalid_open_and_colon_pieces(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn")])
    with pytest.raises(ValueError, match="opening brace"):
        grammar.advance("x")
    grammar.advance("{")
    _advance_all(grammar, ['"', "name", '"'])
    with pytest.raises(ValueError, match="expected colon"):
        grammar.advance(",")
    grammar.advance(":")
    _advance_all(grammar, ['"', "fn", '"'])
    with pytest.raises(ValueError, match="comma after function name"):
        grammar.advance(":")
    grammar.advance(",")
    _advance_all(grammar, ['"', "parameters", '"', ":"])
    with pytest.raises(ValueError, match="opening brace for parameters"):
        grammar.advance("x")
    grammar.advance("{")
    with pytest.raises(ValueError, match="expected comma or closing brace"):
        grammar.advance("x")
    grammar.advance("}")
    with pytest.raises(ValueError, match="expected closing brace"):
        grammar.advance("x")


def test_invalid_delimiter_after_value(tmp_path: Path) -> None:
    grammar = _grammar(tmp_path, [definition("fn", {"a": "boolean"})])
    _advance_all(grammar, _prefix("fn", "a") + ["true"])
    with pytest.raises(ValueError, match="expected comma or closing brace"):
        grammar.advance("x")
