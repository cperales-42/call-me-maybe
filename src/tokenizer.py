"""Byte-level BPE tokenizer built from the public model vocabulary files."""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Final

from pydantic import BaseModel, PrivateAttr

from .models import ModelProtocol


def _build_byte_encoder() -> dict[int, str]:
    """Build the GPT-2 byte-to-unicode mapping.

    Returns:
        A mapping from each byte value to its printable unicode character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = printable[:]
    next_character = 0
    for byte in range(256):
        if byte in printable:
            continue
        printable.append(byte)
        characters.append(256 + next_character)
        next_character += 1
    return {
        byte: chr(character)
        for byte, character in zip(printable, characters)
    }


_BYTE_ENCODER: Final[dict[int, str]] = _build_byte_encoder()
_BYTE_DECODER: Final[dict[str, int]] = {
    character: byte for byte, character in _BYTE_ENCODER.items()
}
_CONTRACTIONS: Final[tuple[str, ...]] = (
    "'s",
    "'t",
    "'re",
    "'ve",
    "'m",
    "'ll",
    "'d",
)


def text_to_token_key(text: str) -> str:
    """Convert text into the byte-level representation used by the vocabulary.

    Args:
        text: Arbitrary unicode text.

    Returns:
        A string whose characters represent the UTF-8 bytes of ``text``.
    """
    return "".join(_BYTE_ENCODER[byte] for byte in text.encode("utf-8"))


def token_key_to_text(key: str) -> str | None:
    """Convert a byte-level vocabulary key into unicode text.

    Args:
        key: Vocabulary key written with the byte-level alphabet.

    Returns:
        The decoded text, or ``None`` when the key contains only a fragment of
        a multi-byte UTF-8 sequence.

    Raises:
        ValueError: If the key contains a character outside the byte alphabet.
    """
    data = bytearray()
    for character in key:
        byte = _BYTE_DECODER.get(character)
        if byte is None:
            raise ValueError(
                f"token key contains an unknown byte symbol: {character!r}"
            )
        data.append(byte)
    try:
        return bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_letter(character: str) -> bool:
    """Check whether a character belongs to a unicode letter category.

    Args:
        character: Single character to inspect.

    Returns:
        ``True`` if the character is a unicode letter; otherwise ``False``.
    """
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    """Check whether a character belongs to a unicode number category.

    Args:
        character: Single character to inspect.

    Returns:
        ``True`` if the character is a unicode number; otherwise ``False``.
    """
    return unicodedata.category(character).startswith("N")


def _pretokenize(text: str) -> list[str]:
    """Split text with the Qwen/GPT-2 byte-level pre-tokenizer rules.

    The implementation mirrors the regex used by the Qwen tokenizer without
    requiring a regular-expression engine with unicode-property support.

    Args:
        text: Text to split into byte-level BPE chunks.

    Returns:
        Non-empty chunks in their original order.
    """
    chunks: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        matched_contraction = False
        for contraction in _CONTRACTIONS:
            candidate = text[index:index + len(contraction)].lower()
            if candidate == contraction:
                chunks.append(text[index:index + len(contraction)])
                index += len(contraction)
                matched_contraction = True
                break
        if matched_contraction:
            continue

        character = text[index]

        if _is_letter(character):
            end = index + 1
            while end < length and _is_letter(text[end]):
                end += 1
            chunks.append(text[index:end])
            index = end
            continue

        if (
            character not in "\r\n"
            and not _is_letter(character)
            and not _is_number(character)
            and index + 1 < length
            and _is_letter(text[index + 1])
        ):
            end = index + 2
            while end < length and _is_letter(text[end]):
                end += 1
            chunks.append(text[index:end])
            index = end
            continue

        if _is_number(character):
            chunks.append(character)
            index += 1
            continue

        punctuation_start = index + 1 if character == " " else index
        punctuation_end = punctuation_start
        while (
            punctuation_end < length
            and not text[punctuation_end].isspace()
            and not _is_letter(text[punctuation_end])
            and not _is_number(text[punctuation_end])
        ):
            punctuation_end += 1
        if punctuation_end > punctuation_start:
            while (
                punctuation_end < length
                and text[punctuation_end] in "\r\n"
            ):
                punctuation_end += 1
            chunks.append(text[index:punctuation_end])
            index = punctuation_end
            continue

        whitespace_end = index
        while whitespace_end < length and text[whitespace_end].isspace():
            whitespace_end += 1

        last_newline = whitespace_end - 1
        while (
            last_newline >= index
            and text[last_newline] not in "\r\n"
        ):
            last_newline -= 1
        if last_newline >= index:
            chunks.append(text[index:last_newline + 1])
            index = last_newline + 1
            continue

        if whitespace_end == length:
            chunks.append(text[index:whitespace_end])
            index = whitespace_end
            continue

        if whitespace_end - index >= 2:
            chunks.append(text[index:whitespace_end - 1])
            index = whitespace_end - 1
            continue

        if whitespace_end > index:
            chunks.append(text[index:whitespace_end])
            index = whitespace_end
            continue

        chunks.append(character)
        index += 1

    return chunks


class SimpleTokenizer(BaseModel):
    """Public byte-level BPE tokenizer for a model vocabulary."""

    vocab_path: str
    merges_path: str | None = None

    _vocab: dict[str, int] = PrivateAttr()
    _id_to_key: dict[int, str] = PrivateAttr()
    _text_to_id: dict[str, int] = PrivateAttr()
    _ranks: dict[tuple[str, str], int] = PrivateAttr()
    _merge_cache: dict[str, tuple[str, ...]] = PrivateAttr()
    _max_text_length: int = PrivateAttr()

    def __init__(
        self,
        vocab_path: str,
        merges_path: str | None = None,
    ) -> None:
        """Initialize the tokenizer from vocabulary and merge files.

        Args:
            vocab_path: Path to the JSON token-to-identifier mapping.
            merges_path: Optional path to a byte-level BPE merges file.
        """
        super().__init__(vocab_path=vocab_path, merges_path=merges_path)

    def model_post_init(self, __context: Any) -> None:
        """Load vocabulary data and merge ranks.

        Args:
            __context: Pydantic initialization context, which is not used.

        Raises:
            ValueError: If a vocabulary or merge file is missing, malformed, or
                inconsistent.
        """
        self._vocab = self._load_vocab()
        self._id_to_key = {
            token_id: key for key, token_id in self._vocab.items()
        }
        self._text_to_id = {}
        for token_id, key in sorted(self._id_to_key.items()):
            text = token_key_to_text(key)
            if text is None:
                continue
            self._text_to_id.setdefault(text, token_id)
        self._max_text_length = max(
            (len(text) for text in self._text_to_id),
            default=0,
        )
        self._merge_cache = {}
        if self.merges_path is None:
            self._ranks = {}
        else:
            self._ranks = self._load_merges(self.merges_path)

    def _load_vocab(self) -> dict[str, int]:
        """Read and validate the vocabulary JSON file.

        Returns:
            A mapping from byte-level token keys to identifiers.

        Raises:
            ValueError: If the file cannot be read or contains invalid data.
        """
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(f"vocabulary file not found: {error}") from error
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
        vocabulary: dict[str, int] = {}
        seen_ids: set[int] = set()
        for key, raw_id in data.items():
            if not isinstance(key, str) or not isinstance(raw_id, int):
                raise ValueError("vocabulary entries must be string-int pairs")
            if raw_id < 0:
                raise ValueError(f"negative token identifier: {raw_id}")
            if raw_id in seen_ids:
                raise ValueError(f"duplicate token identifier: {raw_id}")
            seen_ids.add(raw_id)
            vocabulary[key] = raw_id
        if not vocabulary:
            raise ValueError("vocabulary file must not be empty")
        return vocabulary

    @staticmethod
    def _load_merges(path: str) -> dict[tuple[str, str], int]:
        """Read a byte-level BPE merges file.

        Args:
            path: Path to the merges file.

        Returns:
            A mapping from token pairs to their merge priority.

        Raises:
            ValueError: If the file cannot be read or contains invalid lines.
        """
        ranks: dict[tuple[str, str], int] = {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line_number, raw_line in enumerate(file, start=1):
                    line = raw_line.rstrip("\r\n")
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ")
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        raise ValueError(
                            f"invalid merge on line {line_number}: {line!r}"
                        )
                    pair = (parts[0], parts[1])
                    if pair not in ranks:
                        ranks[pair] = len(ranks)
        except FileNotFoundError as error:
            raise ValueError(f"merges file not found: {error}") from error
        except PermissionError as error:
            raise ValueError(
                f"permission denied for merges file: {error}"
            ) from error
        except OSError as error:
            raise ValueError(
                f"could not read merges file: {error}"
            ) from error
        return ranks

    @classmethod
    def from_model(cls, model: ModelProtocol) -> SimpleTokenizer:
        """Build a tokenizer from the public paths exposed by a model.

        Args:
            model: Model implementing the public SDK interface.

        Returns:
            A tokenizer configured with the model vocabulary and merges.

        Raises:
            ValueError: If the vocabulary path is unavailable or invalid.
        """
        vocab_path = model.get_path_to_vocab_file()
        merges_path: str | None = None
        try:
            merges_path = model.get_path_to_merges_file()
        except (AttributeError, NotImplementedError, OSError, ValueError):
            merges_path = None

        if merges_path is None or not os.path.isfile(merges_path):
            sibling = os.path.join(
                os.path.dirname(vocab_path),
                "merges.txt",
            )
            if os.path.isfile(sibling):
                merges_path = sibling
            else:
                merges_path = None
        return cls(vocab_path=vocab_path, merges_path=merges_path)

    def _merge(self, key: str) -> tuple[str, ...]:
        """Apply byte-level BPE merges to one pre-tokenized chunk.

        Args:
            key: Byte-level representation of a chunk.

        Returns:
            The resulting BPE symbols.

        Raises:
            ValueError: If a merge pair is missing from the merge table.
        """
        cached = self._merge_cache.get(key)
        if cached is not None:
            return cached
        symbols = list(key)
        while len(symbols) > 1:
            best_rank: int | None = None
            best_index: int | None = None
            for index in range(len(symbols) - 1):
                rank = self._ranks.get(
                    (symbols[index], symbols[index + 1])
                )
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
            if best_index is None:
                break
            symbols[best_index:best_index + 2] = [
                symbols[best_index] + symbols[best_index + 1]
            ]
        result = tuple(symbols)
        self._merge_cache[key] = result
        return result

    def _encode_greedy(self, text: str) -> list[int]:
        """Encode text with greedy longest-piece matching.

        This fallback is used when the model does not expose BPE merges.

        Args:
            text: Text to encode.

        Returns:
            A list of token identifiers.

        Raises:
            ValueError: If a character cannot be represented by the vocabulary.
        """
        if self._max_text_length == 0:
            raise ValueError("vocabulary cannot encode text")
        ids: list[int] = []
        index = 0
        while index < len(text):
            longest = min(self._max_text_length, len(text) - index)
            match_id: int | None = None
            match_length = 0
            for candidate_length in range(longest, 0, -1):
                candidate = text[index:index + candidate_length]
                token_id = self._text_to_id.get(candidate)
                if token_id is not None:
                    match_id = token_id
                    match_length = candidate_length
                    break
            if match_id is None:
                raise ValueError(
                    "vocabulary cannot encode character "
                    f"{text[index]!r} at position {index}"
                )
            ids.append(match_id)
            index += match_length
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into model token identifiers.

        Args:
            text: Text to encode. Unicode is normalized to NFC first.

        Returns:
            A list of token identifiers.

        Raises:
            ValueError: If the vocabulary cannot represent a chunk or a BPE
                symbol.
        """
        if text == "":
            return []
        normalized = unicodedata.normalize("NFC", text)
        if not self._ranks:
            return self._encode_greedy(normalized)

        ids: list[int] = []
        for chunk in _pretokenize(normalized):
            key = text_to_token_key(chunk)
            for symbol in self._merge(key):
                token_id = self._vocab.get(symbol)
                if token_id is None:
                    raise ValueError(
                        f"BPE produced an unknown token symbol: {symbol!r}"
                    )
                ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token identifiers into text.

        Args:
            ids: Token identifiers to decode.

        Returns:
            The reconstructed text. Invalid UTF-8 fragments are replaced with
            the Unicode replacement character.

        Raises:
            ValueError: If an identifier is not present in the vocabulary.
        """
        data = bytearray()
        for token_id in ids:
            key = self._id_to_key.get(token_id)
            if key is None:
                raise ValueError(f"unknown token identifier: {token_id}")
            for character in key:
                byte = _BYTE_DECODER.get(character)
                if byte is None:
                    raise ValueError(
                        "token key contains an unknown byte symbol: "
                        f"{character!r}"
                    )
                data.append(byte)
        return bytes(data).decode("utf-8", errors="replace")

    def get_vocab_size(self) -> int:
        """Return the smallest vocabulary size covering every identifier.

        Returns:
            The maximum identifier plus one, or zero for an empty vocabulary.
        """
        if not self._id_to_key:
            return 0
        return max(self._id_to_key) + 1

    def first_token_id(self) -> int:
        """Return a deterministic valid identifier for model probing.

        Returns:
            The smallest identifier in the vocabulary.

        Raises:
            ValueError: If the vocabulary is empty.
        """
        if not self._id_to_key:
            raise ValueError("vocabulary is empty")
        return min(self._id_to_key)
