from __future__ import annotations

import json
import re
import numpy as np
from pydantic import BaseModel, model_validator


QUOTE = chr(34)
BACKSLASH = chr(92)


class Vocabulary(BaseModel):
    vocab_path: str
    logits_size: int
    _direct_map: list[str | None]
    _inverse_map: dict[str, int]
    _symbols: dict[str, int]
    _structural_mask: np.ndarray
    _secure_mask: np.ndarray
    _numeric_mask: np.ndarray
    _valid_json_re: re.Pattern[str] = re.compile(
        r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
    )

    def __init__(self, vocab_path: str, logits_size: int) -> None:
        """Initialize and validate a model vocabulary.

        Args:
            vocab_path: Path to the JSON token-to-identifier mapping.
            logits_size: Number of logits emitted by the model.
        """
        super().__init__(vocab_path=vocab_path, logits_size=logits_size)

    @model_validator(mode="after")
    def _load(self) -> Vocabulary:
        """Load the vocabulary and build lookup tables and token masks.

        Returns:
            The initialized vocabulary instance.

        Raises:
            ValueError: If the vocabulary is unreadable, malformed, incomplete,
                or incompatible with ``logits_size``.
        """
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as file:
                data: dict[str, int] = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(f"file not found: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"broken json: {error}")
        except PermissionError as error:
            raise ValueError(f"Permission denied for file: {error}")
        except OSError as error:
            raise ValueError(f"File error: {error}")

        self._direct_map = [None] * self.logits_size
        for piece, id_ in data.items():
            piece = self._normalize_piece(piece)
            if 0 <= int(id_) < self.logits_size:
                if self._direct_map[id_] is not None:
                    raise ValueError(f"duplicate id_ in vocab: {id_}")
                self._direct_map[id_] = piece
            else:
                raise ValueError("Index out of range")

        self._inverse_map = {}
        for piece, id_ in data.items():
            piece = self._normalize_piece(piece)
            if piece in self._inverse_map:
                raise ValueError(f"same piece in 2 id_s: {piece}")
            self._inverse_map[piece] = id_

        self._symbols = {}
        for symbol in ["{", "}", ":", ",", QUOTE, "[", "]"]:
            if symbol not in self._inverse_map:
                raise ValueError(f"symbol not in json: {symbol}")
            self._symbols[symbol] = self._inverse_map[symbol]
        ids = set(self._symbols.values())
        self._structural_mask = np.array(
            [index in ids for index in range(self.logits_size)],
            dtype=bool,
        )
        self._secure_mask = np.array(
            [
                piece is not None
                and QUOTE not in piece
                and BACKSLASH not in piece
                for piece in self._direct_map
            ],
            dtype=bool,
        )
        self._numeric_mask = np.array(
            [
                piece is not None and self._is_json_number(piece)
                for piece in self._direct_map
            ],
            dtype=bool,
        )

        return self

    @staticmethod
    def _normalize_piece(piece: str) -> str:
        """Normalize a token's whitespace representation.

        Args:
            piece: Raw token text.

        Returns:
            The token text with encoded spaces restored.
        """
        return piece.replace("Ġ", " ")

    def _is_json_number(self, piece: str) -> bool:
        """Check whether a token is a complete JSON number.

        Args:
            piece: Token text to validate.

        Returns:
            ``True`` if the token is a valid JSON number; otherwise ``False``.
        """
        return self._valid_json_re.fullmatch(piece) is not None

    def piece_of(self, id_: int) -> str | None:
        """Return the token associated with a model identifier.

        Args:
            id_: Model token identifier.

        Returns:
            The corresponding token, or ``None`` when the identifier is unused.

        Raises:
            ValueError: If ``id_`` is outside the model vocabulary.
        """
        if 0 <= int(id_) < self.logits_size:
            piece = self._direct_map[id_]
        else:
            raise ValueError("Index out of range")
        return piece

    def id_of(self, piece: str) -> int | None:
        """Return the identifier associated with a token.

        Args:
            piece: Token text to find.

        Returns:
            The corresponding identifier, or ``None`` if the token is absent.
        """
        try:
            id_ = self._inverse_map[piece]
        except KeyError:
            return None
        return id_

    def ids_for_prefix(self, prefix: str) -> list[int]:
        """Find identifiers whose tokens start with a prefix.

        Args:
            prefix: Prefix to match against normalized token text.

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
            A boolean mask excluding quotes, backslashes, and unused entries.
        """
        return self._secure_mask

    def get_numeric_mask(self) -> np.ndarray:
        """Return the mask for complete JSON-number tokens.

        Returns:
            A boolean mask indexed by model token identifier.
        """
        return self._numeric_mask

    def get_vocab_size(self) -> int:
        """Return the number of model vocabulary entries.

        Returns:
            The configured vocabulary size.
        """
        return self.logits_size
