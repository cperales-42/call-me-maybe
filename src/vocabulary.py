"""Vocabulary indexes and token masks used by constrained decoding."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, PrivateAttr

from .tokenizer import token_key_to_text


QUOTE = chr(34)
BACKSLASH = chr(92)
_REQUIRED_JSON_SYMBOLS = ("{", "}", ":", ",", QUOTE)
_NUMBER_CHARACTERS = frozenset("0123456789+-.eE")
_JSON_NUMBER_RE = re.compile(
    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
)


class Vocabulary(BaseModel):
    """Token lookup tables and masks derived from a model vocabulary."""

    vocab_path: str
    logits_size: int

    _direct_map: list[str | None] = PrivateAttr()
    _inverse_map: dict[str, int] = PrivateAttr()
    _symbols: dict[str, int] = PrivateAttr()
    _structural_mask: np.ndarray = PrivateAttr()
    _secure_mask: np.ndarray = PrivateAttr()
    _numeric_mask: np.ndarray = PrivateAttr()
    _number_piece_mask: np.ndarray = PrivateAttr()

    def __init__(self, vocab_path: str, logits_size: int) -> None:
        """Initialize and validate a model vocabulary.

        Args:
            vocab_path: Path to the JSON token-to-identifier mapping.
            logits_size: Number of logits emitted by the model.
        """
        super().__init__(vocab_path=vocab_path, logits_size=logits_size)

    def model_post_init(self, __context: Any) -> None:
        """Load the vocabulary and build lookup tables and token masks.

        Args:
            __context: Pydantic initialization context, which is not used.

        Raises:
            ValueError: If the vocabulary is unreadable, malformed, incomplete,
                or incompatible with ``logits_size``.
        """
        if self.logits_size <= 0:
            raise ValueError("logits_size must be positive")
        data = self._read_vocabulary()
        self._direct_map = [None] * self.logits_size
        seen_ids: set[int] = set()

        for key, raw_id in data.items():
            if not isinstance(key, str) or not isinstance(raw_id, int):
                raise ValueError(
                    "vocabulary entries must be string-int pairs"
                )
            if not 0 <= raw_id < self.logits_size:
                raise ValueError(
                    f"token identifier {raw_id} is outside "
                    f"0..{self.logits_size - 1}"
                )
            if raw_id in seen_ids:
                raise ValueError(f"duplicate token identifier: {raw_id}")
            seen_ids.add(raw_id)
            text = token_key_to_text(key)
            self._direct_map[raw_id] = text

        self._inverse_map = {}
        for token_id, text in enumerate(self._direct_map):
            if text is None:
                continue
            if text in self._inverse_map:
                raise ValueError(
                    f"same decoded token appears twice: {text!r}"
                )
            self._inverse_map[text] = token_id

        missing = [
            symbol
            for symbol in _REQUIRED_JSON_SYMBOLS
            if symbol not in self._inverse_map
        ]
        if missing:
            raise ValueError(
                "vocabulary is missing required JSON symbols: "
                f"{missing}"
            )

        self._symbols = {
            symbol: self._inverse_map[symbol]
            for symbol in _REQUIRED_JSON_SYMBOLS
        }
        structural_ids = set(self._symbols.values())
        self._structural_mask = np.asarray(
            [
                index in structural_ids
                for index in range(self.logits_size)
            ],
            dtype=bool,
        )
        self._secure_mask = np.asarray(
            [
                self._is_valid_string_piece(piece)
                for piece in self._direct_map
            ],
            dtype=bool,
        )
        self._numeric_mask = np.asarray(
            [
                piece is not None
                and _JSON_NUMBER_RE.fullmatch(piece) is not None
                for piece in self._direct_map
            ],
            dtype=bool,
        )
        self._number_piece_mask = np.asarray(
            [
                piece is not None
                and len(piece) > 0
                and all(
                    character in _NUMBER_CHARACTERS
                    for character in piece
                )
                for piece in self._direct_map
            ],
            dtype=bool,
        )

    def _read_vocabulary(self) -> dict[str, int]:
        """Read the raw vocabulary JSON object.

        Returns:
            The decoded JSON object.

        Raises:
            ValueError: If the file cannot be read or is not an object.
        """
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(
                f"vocabulary file not found: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"vocabulary file contains invalid JSON: {error}"
            ) from error
        except PermissionError as error:
            raise ValueError(
                f"permission denied for vocabulary file: {error}"
            ) from error
        except OSError as error:
            raise ValueError(
                f"could not read vocabulary file: {error}"
            ) from error
        if not isinstance(data, dict):
            raise ValueError("vocabulary file must contain a JSON object")
        return {str(key): value for key, value in data.items()}

    @staticmethod
    def _is_valid_string_piece(piece: str | None) -> bool:
        """Check whether a token may appear inside a JSON string.

        Args:
            piece: Decoded token text, or ``None`` for a partial byte token.

        Returns:
            ``True`` if the token is non-empty and contains no quote,
            backslash, or control character; otherwise ``False``.
        """
        return (
            piece is not None
            and len(piece) > 0
            and QUOTE not in piece
            and BACKSLASH not in piece
            and all(ord(character) >= 0x20 for character in piece)
        )

    def piece_of(self, id_: int) -> str | None:
        """Return the token text associated with a model identifier.

        Args:
            id_: Model token identifier.

        Returns:
            The decoded token text, or ``None`` when the identifier is unused
            or represents only a fragment of a UTF-8 sequence.

        Raises:
            ValueError: If ``id_`` is outside the model vocabulary.
        """
        if not 0 <= id_ < self.logits_size:
            raise ValueError(f"token identifier out of range: {id_}")
        return self._direct_map[id_]

    def id_of(self, piece: str) -> int | None:
        """Return the identifier associated with decoded token text.

        Args:
            piece: Token text to find.

        Returns:
            The corresponding identifier, or ``None`` if the text is absent.
        """
        return self._inverse_map.get(piece)

    def ids_for_prefix(self, prefix: str) -> list[int]:
        """Find identifiers whose decoded token text starts with a prefix.

        Args:
            prefix: Prefix to match against decoded token text.

        Returns:
            Matching identifiers in vocabulary order.
        """
        return [
            index
            for index, piece in enumerate(self._direct_map)
            if piece is not None and piece.startswith(prefix)
        ]

    def structural_ids(self) -> set[int]:
        """Return identifiers for structural JSON symbols.

        Returns:
            The identifiers of all required structural symbols.
        """
        return set(self._symbols.values())

    def get_structural_mask(self) -> np.ndarray:
        """Return the mask for structural JSON symbols.

        Returns:
            A boolean mask indexed by model token identifier.
        """
        return self._structural_mask

    def get_secure_mask(self) -> np.ndarray:
        """Return the mask for safe string-content tokens.

        Returns:
            A boolean mask excluding quotes, backslashes, control characters,
            unused entries, and partial byte tokens.
        """
        return self._secure_mask

    def get_numeric_mask(self) -> np.ndarray:
        """Return the mask for complete JSON-number tokens.

        Returns:
            A boolean mask indexed by model token identifier.
        """
        return self._numeric_mask

    def get_number_piece_mask(self) -> np.ndarray:
        """Return the mask for tokens made only of number characters.

        Returns:
            A boolean mask indexed by model token identifier.
        """
        return self._number_piece_mask

    def get_vocab_size(self) -> int:
        """Return the number of model vocabulary entries.

        Returns:
            The configured vocabulary size.
        """
        return self.logits_size
