"""Tests for vocabulary indexes and token masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tokenizer import text_to_token_key
from src.vocabulary import Vocabulary

from .helpers import make_vocabulary, write_vocab


REQUIRED = ["{", "}", ":", ",", '"']
EXTRA = ["a", "hello", "1", "2", "-", ".", "e", "true", " ", "é"]


def test_piece_and_identifier_lookups(tmp_path: Path) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + EXTRA)
    assert vocab.get_vocab_size() == len(REQUIRED + EXTRA)
    assert vocab.piece_of(0) == "{"
    assert vocab.id_of("hello") == EXTRA.index("hello") + len(REQUIRED)
    assert vocab.id_of("missing") is None
    assert vocab.structural_ids() == {
        vocab.id_of(symbol) for symbol in REQUIRED
    }
    with pytest.raises(ValueError, match="out of range"):
        vocab.piece_of(vocab.get_vocab_size())


def test_prefix_search_returns_matching_identifiers(
    tmp_path: Path,
) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + EXTRA)
    ids = vocab.ids_for_prefix("hell")
    assert ids == [vocab.id_of("hello")]


def test_masks_have_one_entry_per_identifier(tmp_path: Path) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + EXTRA, padding=3)
    size = vocab.get_vocab_size()
    for mask in (
        vocab.get_structural_mask(),
        vocab.get_secure_mask(),
        vocab.get_numeric_mask(),
        vocab.get_number_piece_mask(),
    ):
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (size,)
        assert mask.dtype == bool


def test_structural_mask_contains_required_symbols(
    tmp_path: Path,
) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + EXTRA)
    mask = vocab.get_structural_mask()
    assert set(np.flatnonzero(mask).tolist()) == vocab.structural_ids()


def test_secure_mask_excludes_partial_and_unsafe_tokens(
    tmp_path: Path,
) -> None:
    partial_key = text_to_token_key("é")[:1]
    path = tmp_path / "vocab.json"
    data = {
        text_to_token_key(piece): index
        for index, piece in enumerate(REQUIRED + ["safe", "with space"])
    }
    data[partial_key] = len(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    vocab = Vocabulary(str(path), len(data))
    mask = vocab.get_secure_mask()
    safe_id = vocab.id_of("safe")
    space_id = vocab.id_of("with space")
    quote_id = vocab.id_of('"')
    assert safe_id is not None and mask[safe_id]
    assert space_id is not None and mask[space_id]
    assert not mask[len(data) - 1]
    assert quote_id is not None and not mask[quote_id]


def test_secure_mask_rejects_control_characters(tmp_path: Path) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + ["line\nbreak"])
    mask = vocab.get_secure_mask()
    token_id = vocab.id_of("line\nbreak")
    assert token_id is not None and not mask[token_id]


def test_numeric_masks_cover_valid_pieces(tmp_path: Path) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + EXTRA)
    numeric = vocab.get_numeric_mask()
    pieces = vocab.get_number_piece_mask()
    digit_id = vocab.id_of("1")
    exponent_id = vocab.id_of("e")
    word_id = vocab.id_of("hello")
    assert digit_id is not None and numeric[digit_id]
    assert word_id is not None and not numeric[word_id]
    assert exponent_id is not None and pieces[exponent_id]
    assert word_id is not None and not pieces[word_id]


def test_numeric_mask_accepts_complete_multi_character_tokens(
    tmp_path: Path,
) -> None:
    vocab = make_vocabulary(tmp_path, REQUIRED + ["-2.5e3", "01"])
    numeric = vocab.get_numeric_mask()
    valid_id = vocab.id_of("-2.5e3")
    invalid_id = vocab.id_of("01")
    assert valid_id is not None and numeric[valid_id]
    assert invalid_id is not None and not numeric[invalid_id]


def test_missing_symbols_are_rejected(tmp_path: Path) -> None:
    path, _ = write_vocab(tmp_path, ["{", "}", ":", ","])
    with pytest.raises(ValueError, match="missing required JSON symbols"):
        Vocabulary(str(path), 4)


@pytest.mark.parametrize(
    "data,message",
    [
        ({"{": 0}, "missing required JSON symbols"),
        ({"{": 0, "}": 0, ":": 2, ",": 3, '"': 4}, "duplicate"),
        ({"{": 0, "}": 1, ":": 2, ",": 3, '"': 9}, "outside"),
    ],
)
def test_invalid_vocabulary_structures(
    tmp_path: Path,
    data: dict[str, int],
    message: str,
) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Vocabulary(str(path), 5)


def test_non_positive_logits_size_is_rejected(tmp_path: Path) -> None:
    path, _ = write_vocab(tmp_path, REQUIRED)
    with pytest.raises(ValueError, match="logits_size must be positive"):
        Vocabulary(str(path), 0)


def test_unreadable_vocabulary_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        Vocabulary(str(tmp_path / "missing.json"), 5)


def test_malformed_vocabulary_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        Vocabulary(str(path), 5)


def test_vocabulary_json_must_be_an_object(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        Vocabulary(str(path), 5)
