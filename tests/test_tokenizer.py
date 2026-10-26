"""Tests for the public byte-level BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tokenizer import (
    SimpleTokenizer,
    text_to_token_key,
    token_key_to_text,
)

from .helpers import write_vocab


def _write(path: Path, data: object) -> str:
    """Write JSON data and return the file path as a string.

    Args:
        path: Destination file.
        data: JSON-serializable payload.

    Returns:
        The stringified path.
    """
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_text_key_roundtrip_for_ascii_and_unicode() -> None:
    for text in ["hello", "hello world", "año", "日本語", "🙂"]:
        key = text_to_token_key(text)
        assert token_key_to_text(key) == text


def test_partial_utf8_key_returns_none() -> None:
    assert token_key_to_text(text_to_token_key("é")[:1]) is None


def test_token_key_rejects_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="unknown byte symbol"):
        token_key_to_text("☃")


def test_greedy_fallback_uses_longest_piece(tmp_path: Path) -> None:
    vocab_path, _ = write_vocab(
        tmp_path,
        ["hello", " ", "world"],
    )
    tokenizer = SimpleTokenizer(str(vocab_path))
    ids = tokenizer.encode("hello world")
    assert ids == [0, 1, 2]
    assert tokenizer.decode(ids) == "hello world"
    assert tokenizer.get_vocab_size() == 3
    assert tokenizer.first_token_id() == 0


def test_greedy_fallback_rejects_unrepresentable_text(
    tmp_path: Path,
) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    tokenizer = SimpleTokenizer(vocab_path)
    with pytest.raises(ValueError, match="cannot encode character"):
        tokenizer.encode("ab")


def test_exact_bpe_merges_and_space_piece(tmp_path: Path) -> None:
    pieces = [
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "Ġ",
        "he",
        "hel",
        "hell",
        "hello",
        "Ġw",
        "Ġwo",
        "Ġwor",
        "Ġworl",
        "Ġworld",
    ]
    vocab = {
        piece: index
        for index, piece in enumerate(pieces)
    }
    vocab_path = _write(tmp_path / "vocab.json", vocab)
    merges = tmp_path / "merges.txt"
    merges.write_text(
        "h e\nhe l\nhel l\nhell o\nĠ w\nĠw o\nĠwo r\nĠwor l\nĠworl d\n",
        encoding="utf-8",
    )
    tokenizer = SimpleTokenizer(vocab_path, str(merges))
    ids = tokenizer.encode("hello world")
    assert ids == [pieces.index("hello"), pieces.index("Ġworld")]
    assert tokenizer.decode(ids) == "hello world"


def test_nfc_normalization_and_unicode(tmp_path: Path) -> None:
    vocab_path = _write(
        tmp_path / "vocab.json",
        {text_to_token_key("é"): 0},
    )
    tokenizer = SimpleTokenizer(vocab_path)
    assert tokenizer.encode("e\u0301") == [0]
    assert tokenizer.decode([0]) == "é"


def test_decode_replaces_partial_byte_sequences(tmp_path: Path) -> None:
    partial_key = text_to_token_key("é")[:1]
    vocab_path = _write(tmp_path / "vocab.json", {partial_key: 0})
    tokenizer = SimpleTokenizer(vocab_path)
    assert tokenizer.decode([0]) == "\ufffd"


def test_empty_text_encodes_to_no_tokens(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    tokenizer = SimpleTokenizer(vocab_path)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_decode_rejects_unknown_identifier(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    tokenizer = SimpleTokenizer(vocab_path)
    with pytest.raises(ValueError, match="unknown token identifier"):
        tokenizer.decode([7])


def test_missing_vocabulary_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        SimpleTokenizer(str(tmp_path / "missing.json"))


def test_malformed_vocabulary_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        SimpleTokenizer(str(path))


def test_vocabulary_must_be_an_object(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", ["a"])
    with pytest.raises(ValueError, match="JSON object"):
        SimpleTokenizer(path)


@pytest.mark.parametrize(
    "data,message",
    [
        ({"a": "0"}, "string-int pairs"),
        ({"a": -1}, "negative"),
        ({"a": 0, "b": 0}, "duplicate"),
        ({}, "must not be empty"),
    ],
)
def test_invalid_vocabulary_entries(
    tmp_path: Path,
    data: dict[str, object],
    message: str,
) -> None:
    path = _write(tmp_path / "vocab.json", data)
    with pytest.raises(ValueError, match=message):
        SimpleTokenizer(path)


def test_invalid_merge_lines(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    merges = tmp_path / "merges.txt"
    merges.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid merge"):
        SimpleTokenizer(vocab_path, str(merges))


def test_missing_merges_file(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    with pytest.raises(ValueError, match="not found"):
        SimpleTokenizer(vocab_path, str(tmp_path / "none.txt"))


def test_bpe_symbol_missing_from_vocabulary(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    merges = tmp_path / "merges.txt"
    merges.write_text("a b\n", encoding="utf-8")
    tokenizer = SimpleTokenizer(vocab_path, str(merges))
    with pytest.raises(ValueError, match="unknown token symbol"):
        tokenizer.encode("ab")


def test_unknown_byte_symbol_in_vocabulary(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", {"☃": 0})
    with pytest.raises(ValueError, match="unknown byte symbol"):
        SimpleTokenizer(path)


def test_first_token_id_is_deterministic(tmp_path: Path) -> None:
    vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
    tokenizer = SimpleTokenizer(vocab_path)
    assert tokenizer.first_token_id() == 0
