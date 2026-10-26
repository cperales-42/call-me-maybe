"""Finite-state grammar for valid, schema-aware function-call JSON."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr

from .models import FunctionDefinition, SUPPORTED_PARAMETER_TYPES
from .vocabulary import QUOTE, BACKSLASH, Vocabulary


_BOOLEAN_TARGETS = ("true", "false")
_MAX_NUMBER_DIGITS = 15
_MAX_STRING_LENGTH = 256
_JSON_WHITESPACE = frozenset(" \t\r\n")
_NumberStatus = Literal["invalid", "prefix", "complete"]
_KeyRole = Literal["function_name", "parameters", "parameter_value"]


class _State(str, Enum):
    OPEN = "open"
    KEY = "key"
    COLON = "colon"
    FNAME = "fname"
    AFTER_NAME = "after_name"
    OPEN_PARAMS = "open_params"
    VALUE_STR = "value_str"
    VALUE_NUM = "value_num"
    VALUE_BOOL = "value_bool"
    END_VALUE = "end_value"
    CLOSE = "close"
    DONE = "done"


def _is_ascii_digit(character: str) -> bool:
    """Check whether a character is an ASCII decimal digit.

    Args:
        character: Single character to inspect.

    Returns:
        ``True`` for ``0`` through ``9``; otherwise ``False``.
    """
    return "0" <= character <= "9"


def _count_digits(text: str) -> int:
    """Count ASCII decimal digits in text.

    Args:
        text: Text to inspect.

    Returns:
        The number of ASCII decimal digits.
    """
    return sum(_is_ascii_digit(character) for character in text)


def _number_status(text: str, kind: str) -> _NumberStatus:
    """Classify text as an invalid, partial, or complete JSON number.

    Args:
        text: Accumulated numeric text.
        kind: Either ``number`` or ``integer``.

    Returns:
        ``invalid`` when the text cannot become a valid number, ``prefix``
        when it can be extended, and ``complete`` when it is already valid.
    """
    if kind not in {"number", "integer"}:
        return "invalid"
    if text == "":
        return "prefix"

    index = 0
    length = len(text)
    if text[index] == "-":
        index += 1
        if index == length:
            return "prefix"

    if not _is_ascii_digit(text[index]):
        return "invalid"

    if text[index] == "0":
        index += 1
        if index < length and _is_ascii_digit(text[index]):
            return "invalid"
    else:
        index += 1
        while index < length and _is_ascii_digit(text[index]):
            index += 1

    if index == length:
        return "complete"

    if kind == "integer":
        return "invalid"

    if text[index] == ".":
        index += 1
        if index == length:
            return "prefix"
        if not _is_ascii_digit(text[index]):
            return "invalid"
        while index < length and _is_ascii_digit(text[index]):
            index += 1
        if index == length:
            return "complete"

    if text[index] not in "eE":
        return "invalid"

    index += 1
    if index == length:
        return "prefix"
    if text[index] in "+-":
        index += 1
        if index == length:
            return "prefix"
    if not _is_ascii_digit(text[index]):
        return "invalid"
    while index < length and _is_ascii_digit(text[index]):
        index += 1
    return "complete" if index == length else "invalid"


class _Snapshot(NamedTuple):
    """Mutable grammar state captured for speculative token validation."""

    state: _State
    key_target: str
    key_role: _KeyRole
    key_pos: int
    acc: str
    selected: FunctionDefinition | None
    param_index: int
    params: dict[str, Any]
    value_acc: str
    value_kind: str
    value_started: bool
    name_started: bool
    nb_digits: int
    nb_chars: int


def _finite_number(text: str) -> bool:
    """Check whether a complete number converts to a finite float.

    Args:
        text: Complete JSON number.

    Returns:
        ``True`` if the value is finite; otherwise ``False``.
    """
    try:
        return isfinite(float(text))
    except (OverflowError, ValueError):
        return False


class Grammar(BaseModel):
    """Parser state and token masks for one function-call generation."""

    vocab: Vocabulary
    functions: list[FunctionDefinition]

    _state: _State = PrivateAttr()
    _pieces: list[str | None] = PrivateAttr()
    _piece_set: set[str] = PrivateAttr()
    _quote_token_ids: tuple[int, ...] = PrivateAttr()
    _piece_lengths: NDArray[np.int64] = PrivateAttr()
    _piece_digit_counts: NDArray[np.int64] = PrivateAttr()
    _secure_mask: NDArray[np.bool_] = PrivateAttr()
    _number_piece_mask: NDArray[np.bool_] = PrivateAttr()
    _max_piece_length: int = PrivateAttr()
    _prefix_cache: dict[str, tuple[int, ...]] = PrivateAttr()
    _key_target: str = PrivateAttr()
    _key_role: _KeyRole = PrivateAttr()
    _key_pos: int = PrivateAttr()
    _acc: str = PrivateAttr()
    _selected: FunctionDefinition | None = PrivateAttr()
    _param_index: int = PrivateAttr()
    _params: dict[str, Any] = PrivateAttr()
    _sym: dict[str, int] = PrivateAttr()
    _value_acc: str = PrivateAttr()
    _value_kind: str = PrivateAttr()
    _value_started: bool = PrivateAttr()
    _name_started: bool = PrivateAttr()
    _nb_digits: int = PrivateAttr()
    _nb_chars: int = PrivateAttr()

    def __init__(
        self,
        vocab: Vocabulary,
        functions: list[FunctionDefinition],
    ) -> None:
        """Initialize a grammar for a vocabulary and function definitions.

        Args:
            vocab: Vocabulary used to constrain generated tokens.
            functions: Function definitions that may be called.
        """
        super().__init__(vocab=vocab, functions=functions)

    def model_post_init(self, __context: Any) -> None:
        """Initialize parser state and validate schema expressibility.

        Args:
            __context: Pydantic initialization context, which is not used.

        Raises:
            ValueError: If required symbols are absent, a definition is
                unsupported, or the vocabulary cannot express required text.
        """
        self._state = _State.OPEN
        self._pieces = [
            self.vocab.piece_of(index)
            for index in range(self.vocab.get_vocab_size())
        ]
        self._piece_set = {
            piece for piece in self._pieces if piece is not None
        }
        self._quote_token_ids = tuple(
            token_id
            for token_id, piece in enumerate(self._pieces)
            if piece is not None and QUOTE in piece
        )
        self._piece_lengths = np.asarray(
            [
                len(piece) if piece is not None else 0
                for piece in self._pieces
            ],
            dtype=np.int64,
        )
        self._piece_digit_counts = np.asarray(
            [
                _count_digits(piece) if piece is not None else 0
                for piece in self._pieces
            ],
            dtype=np.int64,
        )
        self._secure_mask = np.asarray(
            self.vocab.get_secure_mask(),
            dtype=bool,
        )
        self._number_piece_mask = np.asarray(
            self.vocab.get_number_piece_mask(),
            dtype=bool,
        )
        self._max_piece_length = max(
            (len(piece) for piece in self._piece_set),
            default=0,
        )
        self._prefix_cache = {}

        self._sym = {}
        for symbol in ("{", "}", ":", ",", QUOTE):
            symbol_id = self.vocab.id_of(symbol)
            if symbol_id is None:
                raise ValueError(
                    f"vocabulary cannot represent JSON symbol {symbol!r}"
                )
            self._sym[symbol] = symbol_id

        self._key_target = ""
        self._key_role = "function_name"
        self._key_pos = 0
        self._acc = ""
        self._selected = None
        self._param_index = 0
        self._params = {}
        self._value_acc = ""
        self._value_kind = ""
        self._value_started = False
        self._name_started = False
        self._nb_digits = 0
        self._nb_chars = 0
        self._validate_definitions()
        self._validate_expressibility()

    def _live_candidates(self) -> list[str]:
        """Return function names compatible with the accumulated prefix.

        Returns:
            Function names that start with the current name prefix.
        """
        return [
            function.name
            for function in self.functions
            if function.name.startswith(self._acc)
        ]

    def get_state(self) -> str:
        """Return the current parser state.

        Returns:
            The parser state represented as a string.
        """
        return self._state.value

    def _is_expressible(self, target: str) -> bool:
        """Check whether target text can be formed from vocabulary tokens.

        Args:
            target: Text that must be expressible.

        Returns:
            ``True`` if the target can be segmented into vocabulary pieces;
            otherwise ``False``.
        """
        if target == "" or self._max_piece_length == 0:
            return False
        length = len(target)
        reachable = [False] * (length + 1)
        reachable[length] = True
        for start in range(length - 1, -1, -1):
            end_limit = min(length, start + self._max_piece_length)
            for end in range(start + 1, end_limit + 1):
                if (
                    reachable[end]
                    and target[start:end] in self._piece_set
                ):
                    reachable[start] = True
                    break
        return reachable[0]

    def _validate_definitions(self) -> None:
        """Validate function names, parameter keys, and parameter types.

        Raises:
            ValueError: If the list is empty, a parameter type is unsupported,
                a name contains unsupported characters, or a function name is
                duplicated.
        """
        if not self.functions:
            raise ValueError("at least one function definition is required")
        seen_names: set[str] = set()
        for function in self.functions:
            if function.name in seen_names:
                raise ValueError(
                    f"duplicate function definition: {function.name!r}"
                )
            seen_names.add(function.name)
            if (
                QUOTE in function.name
                or BACKSLASH in function.name
                or any(ord(character) < 0x20 for character in function.name)
            ):
                raise ValueError(
                    "function names with quotes or backslashes or control "
                    f"characters are not supported: {function.name!r}"
                )
            for parameter_name, specification in (
                function.parameters.items()
            ):
                if not parameter_name:
                    raise ValueError(
                        f"empty parameter name in {function.name!r}"
                    )
                if (
                    QUOTE in parameter_name
                    or BACKSLASH in parameter_name
                    or any(
                        ord(character) < 0x20
                        for character in parameter_name
                    )
                ):
                    raise ValueError(
                        "parameter names with quotes or backslashes or "
                        "control characters are not supported: "
                        f"{parameter_name!r}"
                    )
                if specification.type not in SUPPORTED_PARAMETER_TYPES:
                    raise ValueError(
                        "unsupported parameter type "
                        f"{specification.type!r} for {parameter_name!r}; "
                        "supported types are "
                        f"{sorted(SUPPORTED_PARAMETER_TYPES)}"
                    )

    def _validate_expressibility(self) -> None:
        """Validate that required grammar text can be generated.

        Raises:
            ValueError: If the vocabulary cannot express any required target.
        """
        targets = ["name", "parameters"]
        for function in self.functions:
            targets.append(function.name)
            targets.extend(function.parameters.keys())
        for target in targets:
            if not self._is_expressible(target):
                raise ValueError(
                    f"vocabulary cannot express required text {target!r}"
                )

    def _continuation_ids(self, remaining: str) -> tuple[int, ...]:
        """Find tokens that can begin the remaining text.

        Args:
            remaining: Text that still needs to be generated.

        Returns:
            Identifiers of non-empty tokens that are prefixes of ``remaining``.
        """
        if remaining == "":
            return ()
        cached = self._prefix_cache.get(remaining)
        if cached is not None:
            return cached
        ids = tuple(
            token_id
            for token_id, piece in enumerate(self._pieces)
            if piece is not None and remaining.startswith(piece)
        )
        self._prefix_cache[remaining] = ids
        return ids

    def _has_more_parameters(self) -> bool:
        """Check whether another parameter follows the current one.

        Returns:
            ``True`` when another declared parameter remains.

        Raises:
            ValueError: If no function has been selected.
        """
        if self._selected is None:
            raise ValueError("function is not selected")
        return self._param_index + 1 < len(self._selected.parameters)

    def _next_parameter_key(self) -> str:
        """Return the key of the parameter at the current index.

        Returns:
            The parameter name.

        Raises:
            ValueError: If no function is selected or the index is invalid.
        """
        if self._selected is None:
            raise ValueError("function is not selected")
        keys = list(self._selected.parameters)
        if not 0 <= self._param_index < len(keys):
            raise ValueError("parameter index out of range")
        return keys[self._param_index]

    def _reset_value(self) -> None:
        """Clear the accumulated value and its counters."""
        self._value_acc = ""
        self._value_started = False
        self._nb_digits = 0
        self._nb_chars = 0

    def _number_status(self) -> _NumberStatus:
        """Classify the accumulated numeric value.

        Returns:
            The status of the accumulated numeric text.
        """
        return _number_status(self._value_acc, self._value_kind)

    def _can_close_number(self) -> bool:
        """Check whether the accumulated numeric value can be completed.

        Returns:
            ``True`` if the value is a complete and finite number.
        """
        if self._number_status() != "complete":
            return False
        if self._value_kind == "number":
            return _finite_number(self._value_acc)
        return True

    def _store_number(self) -> None:
        """Store the accumulated numeric value in the result mapping.

        Raises:
            ValueError: If the number is incomplete, non-finite, or invalid
                for an integer parameter.
        """
        if not self._can_close_number():
            raise ValueError(
                f"cannot close incomplete number {self._value_acc!r}"
            )
        if self._value_kind == "integer":
            self._params[self._key_target] = int(self._value_acc)
        else:
            value = float(self._value_acc)
            if not isfinite(value):
                raise ValueError("non-finite number")
            self._params[self._key_target] = value

    def _numeric_piece_ids(self) -> set[int]:
        """Return token identifiers that continue the numeric value.

        Returns:
            Valid numeric tokens that respect the digit budget and parameter
            type, plus complete tokens when they are finite.
        """
        ids: set[int] = set()
        remaining_digits = _MAX_NUMBER_DIGITS - self._nb_digits
        if remaining_digits < 0:
            return ids
        for raw_id in np.flatnonzero(self._number_piece_mask):
            token_id = int(raw_id)
            piece = self._pieces[token_id]
            if piece is None:
                continue
            piece_digits = _count_digits(piece)
            if piece_digits > remaining_digits:
                continue
            candidate = self._value_acc + piece
            status = _number_status(candidate, self._value_kind)
            if status == "invalid":
                continue
            available = remaining_digits - piece_digits
            if status == "prefix" and available < 1:
                continue
            if status == "complete":
                if self._value_kind == "number" and not _finite_number(
                    candidate
                ):
                    continue
            ids.add(token_id)
        return ids

    def _delimiter_id(self) -> int:
        """Return the identifier of the delimiter after the current value.

        Returns:
            The comma identifier when more parameters follow, otherwise the
            parameters closing brace.
        """
        if self._has_more_parameters():
            return self._sym[","]
        return self._sym["}"]

    def _limited_string_mask(self, remaining_chars: int) -> NDArray[np.bool_]:
        """Build a string-token mask within a character budget.

        Args:
            remaining_chars: Maximum number of characters still allowed.

        Returns:
            A mask containing only valid string tokens within the limit.
        """
        if remaining_chars <= 0:
            return np.zeros(self.vocab.get_vocab_size(), dtype=bool)
        return (
            self._secure_mask
            & (self._piece_lengths > 0)
            & (self._piece_lengths <= remaining_chars)
        )

    def mask_for_next(self) -> NDArray[np.bool_]:
        """Build the set of tokens allowed by the current parser state.

        Returns:
            A boolean mask indexed by model token identifier.

        Raises:
            ValueError: If generation is complete, no function can match the
                current name, the selected function is missing, or numeric
                state is inconsistent.
        """
        if self._state == _State.DONE:
            raise ValueError("masking next when state is already done")

        ids: set[int] = set()

        if self._state == _State.OPEN:
            ids.add(self._sym["{"])
        elif self._state == _State.COLON:
            ids.add(self._sym[":"])
        elif self._state == _State.AFTER_NAME:
            ids.add(self._sym[","])
        elif self._state == _State.OPEN_PARAMS:
            ids.add(self._sym["{"])
        elif self._state == _State.CLOSE:
            ids.add(self._sym["}"])
        elif self._state == _State.KEY:
            if self._key_pos == -1:
                ids.add(self._sym[QUOTE])
            elif 0 <= self._key_pos < len(self._key_target):
                ids.update(
                    self._continuation_ids(
                        self._key_target[self._key_pos:]
                    )
                )
            elif self._key_pos == len(self._key_target):
                ids.add(self._sym[QUOTE])
            else:
                raise ValueError("key position out of range")
        elif self._state == _State.FNAME:
            if not self._name_started:
                ids.add(self._sym[QUOTE])
            else:
                candidates = self._live_candidates()
                if not candidates:
                    raise ValueError(
                        f"no function matches prefix {self._acc!r}"
                    )
                if self._acc in candidates:
                    ids.add(self._sym[QUOTE])
                for candidate in candidates:
                    ids.update(
                        self._continuation_ids(
                            candidate[len(self._acc):]
                        )
                    )
        elif self._state == _State.END_VALUE:
            ids.add(self._delimiter_id())
        elif self._state == _State.VALUE_STR:
            ids.add(self._sym[QUOTE])
            if self._value_started:
                remaining_chars = _MAX_STRING_LENGTH - self._nb_chars
                if remaining_chars > 0:
                    ids.update(
                        int(token_id)
                        for token_id in np.flatnonzero(
                            self._limited_string_mask(remaining_chars)
                        )
                    )
                for token_id in self._quote_token_ids:
                    piece = self._pieces[token_id]
                    if piece is None or piece == QUOTE:
                        continue
                    if self._piece_is_valid(piece):
                        ids.add(token_id)
        elif self._state == _State.VALUE_NUM:
            piece_ids = self._numeric_piece_ids()
            if self._value_started:
                if not self._can_close_number() and not piece_ids:
                    raise ValueError(
                        "numeric value cannot be completed within limits"
                    )
                if self._can_close_number():
                    ids.add(self._delimiter_id())
            ids.update(piece_ids)
        elif self._state == _State.VALUE_BOOL:
            if self._value_acc in _BOOLEAN_TARGETS:
                ids.add(self._delimiter_id())
            else:
                for target in _BOOLEAN_TARGETS:
                    if target.startswith(self._value_acc):
                        ids.update(
                            self._continuation_ids(
                                target[len(self._value_acc):]
                            )
                        )
        else:
            raise ValueError(f"unhandled grammar state: {self._state}")

        mask = np.zeros(self.vocab.get_vocab_size(), dtype=bool)
        if ids:
            mask[list(ids)] = True
        return mask

    def is_done(self) -> bool:
        """Check whether the function call is fully parsed.

        Returns:
            ``True`` if parsing is complete; otherwise ``False``.
        """
        return self._state == _State.DONE

    def name(self) -> str:
        """Return the parsed function name.

        Returns:
            The selected function's name.

        Raises:
            ValueError: If parsing is incomplete or no function was selected.
        """
        if not self.is_done():
            raise ValueError("tried to get name before parsing completed")
        if self._selected is None:
            raise ValueError("function is not selected")
        return self._selected.name

    def forced_next_token_id(self) -> int | None:
        """Return a token id that is forced by the current parser state.

        Only structurally unambiguous tokens are reported. Numeric values are
        deliberately never finalized here because the model must decide
        whether to continue a number or emit its delimiter.

        Returns:
            The forced token id, or ``None`` when the model must choose.
        """
        if self._state == _State.DONE:
            return None
        if self._state == _State.OPEN:
            return self._sym["{"]
        if self._state == _State.COLON:
            return self._sym[":"]
        if self._state == _State.AFTER_NAME:
            return self._sym[","]
        if self._state == _State.OPEN_PARAMS:
            return self._sym["{"]
        if self._state == _State.CLOSE:
            return self._sym["}"]
        if self._state == _State.KEY:
            if self._key_pos in (-1, len(self._key_target)):
                return self._sym[QUOTE]
            return None
        if self._state == _State.FNAME:
            if not self._name_started:
                return self._sym[QUOTE]
            candidates = self._live_candidates()
            if len(candidates) == 1 and self._acc == candidates[0]:
                return self._sym[QUOTE]
            return None
        if self._state == _State.VALUE_STR:
            if not self._value_started:
                return self._sym[QUOTE]
            if self._nb_chars >= _MAX_STRING_LENGTH:
                return self._sym[QUOTE]
            return None
        if self._state == _State.END_VALUE:
            return self._delimiter_id()
        return None

    def forced_next_piece(self) -> str | None:
        """Return the forced token piece for the current parser state.

        Returns:
            The token piece if forced, otherwise ``None``.

        Raises:
            ValueError: If a forced identifier has no usable vocabulary piece.
        """
        token_id = self.forced_next_token_id()
        if token_id is None:
            return None
        piece = self.vocab.piece_of(token_id)
        if piece is None:
            raise ValueError(
                f"forced token {token_id} has no usable vocabulary piece"
            )
        return piece

    def parameters(self) -> dict[str, Any]:
        """Return the parsed function arguments.

        Returns:
            A copy of the mapping from argument names to parsed values.
        """
        return dict(self._params)

    def _snapshot(self) -> _Snapshot:
        """Capture mutable parser state for atomic token validation.

        Returns:
            A detached snapshot of all mutable parser fields.
        """
        return _Snapshot(
            state=self._state,
            key_target=self._key_target,
            key_role=self._key_role,
            key_pos=self._key_pos,
            acc=self._acc,
            selected=self._selected,
            param_index=self._param_index,
            params=dict(self._params),
            value_acc=self._value_acc,
            value_kind=self._value_kind,
            value_started=self._value_started,
            name_started=self._name_started,
            nb_digits=self._nb_digits,
            nb_chars=self._nb_chars,
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        """Restore parser state captured by ``_snapshot``.

        Args:
            snapshot: State to restore.
        """
        self._state = snapshot.state
        self._key_target = snapshot.key_target
        self._key_role = snapshot.key_role
        self._key_pos = snapshot.key_pos
        self._acc = snapshot.acc
        self._selected = snapshot.selected
        self._param_index = snapshot.param_index
        self._params = dict(snapshot.params)
        self._value_acc = snapshot.value_acc
        self._value_kind = snapshot.value_kind
        self._value_started = snapshot.value_started
        self._name_started = snapshot.name_started
        self._nb_digits = snapshot.nb_digits
        self._nb_chars = snapshot.nb_chars

    def _piece_is_valid(self, piece: str) -> bool:
        """Check whether a token piece can be consumed without committing it.

        Args:
            piece: Token text to test from the current parser state.

        Returns:
            ``True`` when the complete token can be consumed atomically.
        """
        snapshot = self._snapshot()
        try:
            self.advance(piece)
        except ValueError:
            return False
        finally:
            self._restore(snapshot)
        return True

    def _skips_whitespace(self) -> bool:
        """Check whether whitespace is allowed between JSON tokens.

        Returns:
            ``True`` when the current parser state permits insignificant
            whitespace before the next structural character.
        """
        if self._state in {
            _State.OPEN,
            _State.COLON,
            _State.AFTER_NAME,
            _State.OPEN_PARAMS,
            _State.END_VALUE,
            _State.CLOSE,
            _State.DONE,
        }:
            return True
        if self._state == _State.KEY:
            return self._key_pos == -1
        if self._state == _State.FNAME:
            return not self._name_started
        return False

    def _advance_key_char(self, character: str) -> None:
        """Advance the key-matching sub-state by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character does not continue or close the key.
        """
        if self._key_pos == -1:
            if character != QUOTE:
                raise ValueError("expected opening quote for key")
            self._key_pos = 0
            return
        if self._key_pos < len(self._key_target):
            if character != self._key_target[self._key_pos]:
                raise ValueError(
                    f"character {character!r} does not continue key "
                    f"{self._key_target!r}"
                )
            self._key_pos += 1
            return
        if self._key_pos == len(self._key_target):
            if character != QUOTE:
                raise ValueError("expected closing quote for key")
            self._state = _State.COLON
            return
        raise ValueError("key position out of range")

    def _advance_function_name_char(self, character: str) -> None:
        """Advance the function-name matching sub-state by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character cannot continue or close a function
                name.
        """
        if not self._name_started:
            if character != QUOTE:
                raise ValueError("expected opening quote for function name")
            self._name_started = True
            return
        if character == QUOTE:
            candidates = self._live_candidates()
            if self._acc not in candidates:
                raise ValueError(
                    f"unknown function name {self._acc!r}"
                )
            selected = next(
                (
                    function
                    for function in self.functions
                    if function.name == self._acc
                ),
                None,
            )
            if selected is None:
                raise ValueError("function is not selected")
            self._selected = selected
            self._key_target = "parameters"
            self._key_role = "parameters"
            self._key_pos = -1
            self._state = _State.AFTER_NAME
            return
        candidate = self._acc + character
        if not any(
            function.name.startswith(candidate)
            for function in self.functions
        ):
            raise ValueError(
                f"character {character!r} does not continue any function name"
            )
        self._acc = candidate

    def _advance_string_char(self, character: str) -> None:
        """Advance the string-value sub-state by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character is invalid inside a JSON string or
                would exceed the configured length limit.
        """
        if not self._value_started:
            if character != QUOTE:
                raise ValueError("string value must start with a quote")
            self._value_started = True
            self._nb_chars = 0
            self._value_acc = ""
            return
        if character == QUOTE:
            self._params[self._key_target] = self._value_acc
            self._reset_value()
            self._state = _State.END_VALUE
            return
        if character == BACKSLASH or ord(character) < 0x20:
            raise ValueError("invalid string piece")
        if self._nb_chars + 1 > _MAX_STRING_LENGTH:
            raise ValueError("string exceeds maximum length")
        self._value_acc += character
        self._nb_chars += 1

    def _advance_number_char(self, character: str) -> None:
        """Advance the numeric-value sub-state by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character is not a valid numeric continuation
                or the number is closed while incomplete.
        """
        if character in (",", "}"):
            if not self._can_close_number():
                raise ValueError(
                    f"cannot close incomplete number {self._value_acc!r}"
                )
            self._store_number()
            self._reset_value()
            self._advance_end_value_char(character)
            return

        character_digits = 1 if _is_ascii_digit(character) else 0
        if self._nb_digits + character_digits > _MAX_NUMBER_DIGITS:
            raise ValueError("number exceeds maximum digit count")
        candidate = self._value_acc + character
        status = _number_status(candidate, self._value_kind)
        if status == "invalid":
            raise ValueError(
                f"invalid numeric continuation {character!r}"
            )
        if (
            status == "complete"
            and self._value_kind == "number"
            and not _finite_number(candidate)
        ):
            raise ValueError("numeric value is not finite")
        self._value_acc = candidate
        self._nb_digits += character_digits
        self._value_started = True

    def _advance_boolean_char(self, character: str) -> None:
        """Advance the boolean-value sub-state by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character cannot continue a boolean literal.
        """
        if character in (",", "}"):
            if self._value_acc not in _BOOLEAN_TARGETS:
                raise ValueError("incomplete boolean literal")
            value = self._value_acc == "true"
            self._params[self._key_target] = value
            self._reset_value()
            self._advance_end_value_char(character)
            return

        candidate = self._value_acc + character
        if not any(
            target.startswith(candidate)
            for target in _BOOLEAN_TARGETS
        ):
            raise ValueError(
                f"invalid boolean continuation {character!r}"
            )
        self._value_acc = candidate
        self._value_started = True
        if candidate in _BOOLEAN_TARGETS:
            self._params[self._key_target] = candidate == "true"
            self._reset_value()
            self._state = _State.END_VALUE

    def _advance_end_value_char(self, character: str) -> None:
        """Advance after a completed value by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character is not the required delimiter.
        """
        has_more = self._has_more_parameters()
        if character == ",":
            if not has_more:
                raise ValueError(
                    "unexpected comma after the final parameter"
                )
            self._param_index += 1
            self._key_target = self._next_parameter_key()
            self._key_role = "parameter_value"
            self._key_pos = -1
            self._state = _State.KEY
            return
        if character == "}":
            if has_more:
                raise ValueError(
                    "cannot close parameters before all values exist"
                )
            self._state = _State.CLOSE
            return
        raise ValueError("expected comma or closing brace")

    def _advance_char(self, character: str) -> None:
        """Advance the parser by one character.

        Args:
            character: Character to consume.

        Raises:
            ValueError: If the character is invalid for the current state.
        """
        if character in _JSON_WHITESPACE and self._skips_whitespace():
            return

        if self._state == _State.DONE:
            raise ValueError("tried to advance after parsing completed")

        if self._state == _State.OPEN:
            if character != "{":
                raise ValueError("expected opening brace")
            self._state = _State.KEY
            self._key_target = "name"
            self._key_role = "function_name"
            self._key_pos = -1
            return

        if self._state == _State.KEY:
            self._advance_key_char(character)
            return

        if self._state == _State.COLON:
            if character != ":":
                raise ValueError("expected colon")
            if self._key_role == "function_name":
                self._state = _State.FNAME
                self._acc = ""
                self._name_started = False
            elif self._key_role == "parameters":
                self._state = _State.OPEN_PARAMS
            else:
                if self._selected is None:
                    raise ValueError("function is not selected")
                parameter = self._selected.parameters.get(
                    self._key_target
                )
                if parameter is None:
                    raise ValueError(
                        f"unknown parameter {self._key_target!r}"
                    )
                self._reset_value()
                self._value_kind = parameter.type
                if parameter.type == "string":
                    self._state = _State.VALUE_STR
                elif parameter.type == "boolean":
                    self._state = _State.VALUE_BOOL
                elif parameter.type in {"integer", "number"}:
                    self._state = _State.VALUE_NUM
                else:
                    raise ValueError(
                        f"unsupported parameter type {parameter.type!r}"
                    )
            return

        if self._state == _State.FNAME:
            self._advance_function_name_char(character)
            return

        if self._state == _State.AFTER_NAME:
            if character != ",":
                raise ValueError("expected comma after function name")
            self._state = _State.KEY
            self._key_target = "parameters"
            self._key_role = "parameters"
            self._key_pos = -1
            return

        if self._state == _State.OPEN_PARAMS:
            if character != "{":
                raise ValueError("expected opening brace for parameters")
            if self._selected is None:
                raise ValueError("function is not selected")
            self._param_index = 0
            if not self._selected.parameters:
                self._state = _State.END_VALUE
            else:
                self._key_target = self._next_parameter_key()
                self._key_role = "parameter_value"
                self._key_pos = -1
                self._state = _State.KEY
            return

        if self._state == _State.VALUE_STR:
            self._advance_string_char(character)
            return

        if self._state == _State.VALUE_NUM:
            self._advance_number_char(character)
            return

        if self._state == _State.VALUE_BOOL:
            self._advance_boolean_char(character)
            return

        if self._state == _State.END_VALUE:
            self._advance_end_value_char(character)
            return

        if self._state == _State.CLOSE:
            if character != "}":
                raise ValueError("expected closing brace")
            self._state = _State.DONE
            return

        raise ValueError(f"unhandled grammar state: {self._state}")

    def advance(self, piece: str) -> None:
        """Consume one token and atomically advance the parser state.

        A token may span multiple grammar states because byte-level BPE can
        combine string content, closing quotes, and structural characters.
        Invalid tokens leave the parser exactly as it was before the call.

        Args:
            piece: Vocabulary token to consume.

        Raises:
            ValueError: If the token is empty or invalid for the current
                parser state.
        """
        if piece == "":
            raise ValueError("token piece must not be empty")
        snapshot = self._snapshot()
        try:
            for character in piece:
                self._advance_char(character)
        except ValueError:
            self._restore(snapshot)
            raise
