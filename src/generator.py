from __future__ import annotations
from math import isfinite
from .grammar import Grammar
from .models import FunctionCallResult
from llm_sdk import Small_LLM_Model
import numpy as np


def generate_prompt(
	model: Small_LLM_Model,
	grammar: Grammar,
	prompt: str,
	inference_prompt: str,
	max_iter: int,
) -> FunctionCallResult:
	if grammar.is_done():
		raise ValueError("Grammar already used")
	if max_iter <= 0:
		raise ValueError("Negative number not allowed for max_iter param")
	if prompt == "":
		raise ValueError("Please introduce a prompt")
	ids = model.encode(inference_prompt).tolist()[0]
	i = 0
	while not grammar.is_done() and i < max_iter:
		logits = model.get_logits_from_input_ids(ids)
		if len(logits) != grammar.vocab.get_vocab_size():
			raise ValueError("Logits len differs from len of the model vocabulary")
		mask = grammar.mask_for_next()
		if not mask.any():
			raise ValueError("No mask found")
		array_logits = np.asarray(logits, dtype="float64")
		sanitized_logits = np.where(np.isnan(array_logits), -np.inf, array_logits)
		logits = np.where(mask, sanitized_logits, -np.inf)
		token = int(np.argmax(logits))
		if not mask[token]:
			raise ValueError(f"Token {token} not found in mask")
		piece = grammar.vocab.piece_of(token)
		if piece is None:
			raise ValueError(f"Token {token} not in vocab")
		i += 1
		grammar.advance(piece)
		ids.append(token)
	if grammar.is_done():
		f_name = grammar.name()
		f_parameters = grammar.parameters()
		for value in f_parameters.values():
			if isinstance(value, float) and isfinite(value) is False:
				raise ValueError("Not finite float on parameters")
		result = FunctionCallResult(prompt=prompt,
									name=f_name,
									parameters=f_parameters)
		return result
	else:
		raise ValueError("Time out :/")