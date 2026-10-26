"""Constrained token generation for function calls."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel

from .grammar import Grammar
from .models import FunctionCallResult, ModelProtocol
from .tokenizer import SimpleTokenizer


class GenerationEvent(BaseModel):
    """One observable token-selection step."""

    step: int
    token_id: int
    piece: str
    state: str
    source: Literal["model", "forced"]
    allowed_tokens: int
    score: float | None = None


GenerationObserver = Callable[[GenerationEvent], None]


def _select_singleton(mask: np.ndarray) -> int:
    """Return the identifier of the only allowed token.

    Args:
        mask: Boolean token mask with exactly one true element.

    Returns:
        The index of the allowed token.

    Raises:
        ValueError: If the mask does not contain exactly one true element.
    """
    indices = np.flatnonzero(mask)
    if len(indices) != 1:
        raise ValueError("expected exactly one allowed token")
    return int(indices[0])


def generate_prompt(
    model: ModelProtocol,
    grammar: Grammar,
    prompt: str,
    inference_prompt: str,
    max_iter: int,
    *,
    tokenizer: SimpleTokenizer | None = None,
    observer: GenerationObserver | None = None,
) -> FunctionCallResult:
    """Generate a function call by repeatedly selecting constrained tokens.

    The loop bypasses the model whenever the grammar leaves exactly one valid
    token. Numeric values are never closed by the fast-path because the model
    must decide whether to continue the number or emit a delimiter.

    Args:
        model: Model used to produce next-token logits.
        grammar: Grammar that constrains generation to a valid function call.
        prompt: Original user request stored in the result.
        inference_prompt: Expanded prompt supplied to the model.
        max_iter: Maximum number of tokens to generate.
        tokenizer: Optional shared tokenizer; created from the model if absent.
        observer: Optional callback invoked once for every generated token.

    Returns:
        The completed and validated function-call result.

    Raises:
        ValueError: If inputs are invalid, generation exceeds ``max_iter``, or
            a token violates the grammar.
    """
    if grammar.is_done():
        raise ValueError("grammar was already completed")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if prompt == "":
        raise ValueError("prompt must not be empty")

    active_tokenizer = tokenizer
    if active_tokenizer is None:
        active_tokenizer = SimpleTokenizer.from_model(model)
    ids = active_tokenizer.encode(inference_prompt)
    if not ids:
        raise ValueError("inference prompt produced no tokens")

    iteration = 0
    while not grammar.is_done() and iteration < max_iter:
        state = grammar.get_state()
        score: float | None = None
        allowed_tokens = 1
        forced_token_id = grammar.forced_next_token_id()

        if forced_token_id is not None:
            forced_piece = grammar.forced_next_piece()
            if forced_piece is None:
                raise ValueError(
                    f"forced token {forced_token_id} has no usable piece"
                )
            token_id = forced_token_id
            piece = forced_piece
            source: Literal["model", "forced"] = "forced"
        else:
            mask = grammar.mask_for_next()
            allowed_tokens = int(np.count_nonzero(mask))
            if allowed_tokens == 0:
                raise ValueError(
                    f"no allowed token in grammar state {state!r}"
                )
            if allowed_tokens == 1:
                token_id = _select_singleton(mask)
                singleton_piece = grammar.vocab.piece_of(token_id)
                if singleton_piece is None:
                    raise ValueError(
                        f"token {token_id} has no usable vocabulary piece"
                    )
                piece = singleton_piece
                source = "forced"
            else:
                logits = model.get_logits_from_input_ids(ids)
                if len(logits) != grammar.vocab.get_vocab_size():
                    raise ValueError(
                        "logits length differs from the model vocabulary size"
                    )
                array_logits = np.asarray(logits, dtype="float64")
                sanitized_logits = np.where(
                    np.isnan(array_logits),
                    -np.inf,
                    array_logits,
                )
                filtered_logits = np.where(
                    mask,
                    sanitized_logits,
                    -np.inf,
                )
                if not np.isfinite(filtered_logits).any():
                    raise ValueError(
                        "no finite logit among the allowed tokens"
                    )
                token_id = int(np.argmax(filtered_logits))
                if not mask[token_id]:
                    raise ValueError(
                        f"selected token {token_id} is not allowed"
                    )
                selected_piece = grammar.vocab.piece_of(token_id)
                if selected_piece is None:
                    raise ValueError(
                        f"token {token_id} has no usable vocabulary piece"
                    )
                piece = selected_piece
                score = float(filtered_logits[token_id])
                source = "model"

        grammar.advance(piece)
        ids.append(token_id)
        if observer is not None:
            observer(
                GenerationEvent(
                    step=iteration,
                    token_id=token_id,
                    piece=piece,
                    state=state,
                    source=source,
                    allowed_tokens=allowed_tokens,
                    score=score,
                )
            )
        iteration += 1

    if not grammar.is_done():
        raise ValueError(
            "generation did not finish within max_iter tokens "
            f"(state={grammar.get_state()!r})"
        )

    return FunctionCallResult(
        prompt=prompt,
        name=grammar.name(),
        parameters=grammar.parameters(),
    )
