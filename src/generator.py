from __future__ import annotations

from math import isfinite
from typing import Any, Protocol

import numpy as np

from .grammar import Grammar
from .models import FunctionCallResult


class ModelProtocol(Protocol):
    def encode(self, text: str) -> Any:
        """Encode text as model input token identifiers.

        Args:
            text: Text to encode.

        Returns:
            The encoded input, including its batch dimension.
        """
        ...

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Compute the next-token logits for an encoded input.

        Args:
            input_ids: Token identifiers to use as model input.

        Returns:
            One logit for each token in the model vocabulary.
        """
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the model vocabulary file.

        Returns:
            The vocabulary file path.
        """
        ...


def generate_prompt(
    model: ModelProtocol,
    grammar: Grammar,
    prompt: str,
    inference_prompt: str,
    max_iter: int,
) -> FunctionCallResult:
    """Generate a function call by repeatedly sampling constrained tokens.

    Args:
        model: Model used to encode prompts and produce logits.
        grammar: Grammar that constrains generation to a valid function call.
        prompt: Original user request to store in the result.
        inference_prompt: Expanded prompt supplied to the model.
        max_iter: Maximum number of tokens to generate.

    Returns:
        The completed and validated function-call result.

    Raises:
        ValueError: If the grammar is already complete, an input is invalid,
            generation exceeds ``max_iter``, or a token violates the grammar.
    """
    if grammar.is_done():
        raise ValueError("Grammar already used")
    if max_iter <= 0:
        raise ValueError("Negative number not allowed for max_iter param")
    if prompt == "":
        raise ValueError("Please introduce a prompt")
    ids = model.encode(inference_prompt).tolist()[0]
    iteration = 0
    while not grammar.is_done() and iteration < max_iter:
        logits = model.get_logits_from_input_ids(ids)
        if len(logits) != grammar.vocab.get_vocab_size():
            raise ValueError(
                "Logits len differs from len of the model vocabulary"
            )
        mask = grammar.mask_for_next()
        if not mask.any():
            raise ValueError("No mask found")
        array_logits = np.asarray(logits, dtype="float64")
        sanitized_logits = np.where(
            np.isnan(array_logits),
            -np.inf,
            array_logits,
        )
        filtered_logits = np.where(mask, sanitized_logits, -np.inf)
        token = int(np.argmax(filtered_logits))
        if not mask[token]:
            raise ValueError(f"Token {token} not found in mask")
        piece = grammar.vocab.piece_of(token)
        if piece is None:
            raise ValueError(f"Token {token} not in vocab")
        iteration += 1
        grammar.advance(piece)
        ids.append(token)
    if grammar.is_done():
        function_name = grammar.name()
        parameters = grammar.parameters()
        for value in parameters.values():
            if isinstance(value, float) and isfinite(value) is False:
                raise ValueError("Not finite float on parameters")
        return FunctionCallResult(
            prompt=prompt,
            name=function_name,
            parameters=parameters,
        )
    raise ValueError("Time out :/")
