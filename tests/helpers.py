"""Shared fixtures and deterministic test doubles."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import FunctionDefinition, ModelProtocol
from src.tokenizer import SimpleTokenizer, text_to_token_key
from src.vocabulary import Vocabulary


def write_vocab(
    tmp_path: Path,
    pieces: list[str],
    *,
    filename: str = "vocab.json",
) -> tuple[Path, dict[str, int]]:
    """Write a byte-level vocabulary file for the supplied text pieces.

    Args:
        tmp_path: Directory where the vocabulary file is created.
        pieces: Decoded token texts in identifier order.
        filename: Name of the generated JSON file.

    Returns:
        The vocabulary path and its decoded-text-to-identifier mapping.

    Raises:
        ValueError: If two pieces decode to the same text.
    """
    mapping: dict[str, int] = {}
    keys: dict[str, int] = {}
    for piece in pieces:
        if piece in mapping:
            raise ValueError(f"duplicate decoded piece: {piece!r}")
        mapping[piece] = len(mapping)
        keys[text_to_token_key(piece)] = mapping[piece]
    path = tmp_path / filename
    path.write_text(json.dumps(keys), encoding="utf-8")
    return path, mapping


def make_vocabulary(
    tmp_path: Path,
    pieces: list[str],
    *,
    padding: int = 0,
) -> Vocabulary:
    """Build a vocabulary whose logits size may include unused identifiers.

    Args:
        tmp_path: Directory where the vocabulary file is created.
        pieces: Decoded token texts in identifier order.
        padding: Number of unused identifiers appended after the pieces.

    Returns:
        The validated vocabulary.
    """
    path, mapping = write_vocab(tmp_path, pieces)
    return Vocabulary(vocab_path=str(path), logits_size=len(mapping) + padding)


def base_pieces() -> list[str]:
    """Return structural JSON pieces plus common grammar text.

    Returns:
        A list of decoded token texts suitable for grammar tests.
    """
    return [
        "{",
        "}",
        ":",
        ",",
        '"',
        "name",
        "parameters",
        "fn",
        "fn_long",
        "a",
        "b",
        "flag",
        "x",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "-",
        "+",
        ".",
        "e",
        "E",
        "true",
        "false",
        "hello",
        " ",
        "fn_add",
        "fn_add_numbers",
    ]


def definition(
    name: str,
    parameters: dict[str, str] | None = None,
) -> FunctionDefinition:
    """Build a function definition with declared parameter types.

    Args:
        name: Function name.
        parameters: Mapping from parameter name to JSON type.

    Returns:
        The validated function definition.
    """
    return FunctionDefinition.model_validate(
        {
            "name": name,
            "description": f"Description of {name}.",
            "parameters": {
                parameter_name: {"type": parameter_type}
                for parameter_name, parameter_type in (
                    parameters or {}
                ).items()
            },
            "returns": {"type": "string"},
        }
    )


def ascii_pieces() -> list[str]:
    """Return printable ASCII pieces plus multi-character name tokens.

    Returns:
        Decoded pieces covering every character used by the example prompts.
    """
    pieces = [chr(code) for code in range(32, 127)]
    pieces.append("\n")
    pieces.extend(
        [
            "name",
            "parameters",
            "fn_add_numbers",
            "fn_greet",
            "fn_ping",
            "fn_toggle",
            "fn_count",
            "string_value",
            "prompt",
        ]
    )
    return pieces


class TargetModel(ModelProtocol):
    """Deterministic model that steers generation toward a target JSON text."""

    target: str
    tokenizer: SimpleTokenizer
    logits_size: int
    pieces: dict[str, int]
    calls: int = 0

    def _decoded_suffix(self, input_ids: list[int]) -> int:
        """Find how much of the target is already present at the end.

        Args:
            input_ids: Model input identifiers.

        Returns:
            The length of the longest target prefix that suffixes the input.
        """
        decoded = self.tokenizer.decode(input_ids)
        upper = min(len(decoded), len(self.target))
        for length in range(upper, -1, -1):
            if decoded.endswith(self.target[:length]):
                return length
        return 0

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return logits that favour the next target token.

        Args:
            input_ids: Model input identifiers.

        Returns:
            One logit per vocabulary identifier.

        Raises:
            RuntimeError: If the target cannot be continued.
        """
        self.calls += 1
        prefix_length = self._decoded_suffix(input_ids)
        remaining = self.target[prefix_length:]
        if not remaining:
            raise RuntimeError("target model has no remaining target text")
        desired_id: int | None = None
        desired_length = 0
        for piece, token_id in self.pieces.items():
            if (
                piece
                and len(piece) > desired_length
                and remaining.startswith(piece)
            ):
                desired_id = token_id
                desired_length = len(piece)
        if desired_id is None:
            raise RuntimeError(
                f"target model cannot continue {remaining!r}"
            )
        logits = [0.0] * self.logits_size
        logits[desired_id] = 100.0
        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return the tokenizer vocabulary path.

        Returns:
            The vocabulary file path.
        """
        return self.tokenizer.vocab_path

    def get_path_to_merges_file(self) -> str:
        """Return an empty merges path because the test tokenizer is greedy.

        Returns:
            An empty string.
        """
        return ""


def build_target_model(
    tmp_path: Path,
    target: str,
    pieces: list[str] | None = None,
    *,
    vocab_path: Path | None = None,
) -> TargetModel:
    """Build a deterministic model and tokenizer for a target JSON text.

    Args:
        tmp_path: Directory used when no vocabulary path is supplied.
        target: JSON text that generation should follow.
        pieces: Vocabulary pieces; defaults to printable ASCII.
        vocab_path: Optional pre-existing vocabulary file.

    Returns:
        A model whose logits steer generation toward ``target``.
    """
    selected_pieces = pieces if pieces is not None else ascii_pieces()
    if vocab_path is None:
        vocab_path, mapping = write_vocab(tmp_path, selected_pieces)
    else:
        mapping = {
            piece: index
            for index, piece in enumerate(selected_pieces)
        }
    tokenizer = SimpleTokenizer(vocab_path=str(vocab_path))
    logits_size = max(mapping.values()) + 1 if mapping else 0
    return TargetModel(
        target=target,
        tokenizer=tokenizer,
        logits_size=logits_size,
        pieces=mapping,
    )
