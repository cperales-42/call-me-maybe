from __future__ import annotations

import re
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel

from .models import FunctionDefinition
from .vocabulary import Vocabulary


_JSON_NUMBER_RE = re.compile(
	r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
)

_MAX_NUMBER_DIGITS = 9
_MAX_STRING_LENGTH = 128


class _State(str, Enum):
	OPEN = "open"
	KEY = "key"
	COLON = "colon"
	FNAME = "fname"
	AFTER_NAME = "after_name"
	OPEN_PARAMS = "open_params"
	VALUE_STR = "value_str"
	VALUE_NUM = "value_num"
	END_VALUE = "end_value"
	CLOSE = "close"
	DONE = "done"


class Grammar(BaseModel):
	vocab: Vocabulary
	functions: list[FunctionDefinition]
	_state: _State
	_pieces: list[str | None]
	_piece_lengths: np.ndarray
	_piece_digit_counts: np.ndarray
	_string_piece_mask: np.ndarray
	_key_target: str
	_key_pos: int
	_acc: str
	_selected: FunctionDefinition | None
	_param_index: int
	_params: dict[str, Any]
	_sym: dict[str, int]
	_value_acc: str
	_value_started: bool
	_name_started: bool
	_nb_digits: int
	_nb_chars: int

	def model_post_init(self, __context: Any) -> None:
		self._state = _State.OPEN
		self._pieces = [
			self.vocab.piece_of(i)
			for i in range(self.vocab.get_vocab_size())
		]
		self._piece_lengths = np.asarray(
			[
				len(piece) if piece is not None else 0
				for piece in self._pieces
			],
			dtype=np.int64,
		)
		self._piece_digit_counts = np.asarray(
			[
				self._count_digits(piece) if piece is not None else 0
				for piece in self._pieces
			],
			dtype=np.int64,
		)
		self._string_piece_mask = np.asarray(
			[
				piece is not None and self._is_valid_string_piece(piece)
				for piece in self._pieces
			],
			dtype=bool,
		)

		self._sym = {}
		json_chars = ["{", "}", ":", ",", '"']
		for char in json_chars:
			symbol_id = self.vocab.id_of(char)
			if symbol_id is None:
				raise ValueError(f"Unknown id {char}")
			self._sym[char] = symbol_id

		self._key_target = ""
		self._key_pos = 0
		self._acc = ""
		self._selected = None
		self._param_index = 0
		self._value_acc = ""
		self._value_started = False
		self._name_started = False
		self._nb_digits = 0
		self._nb_chars = 0
		self._params = {}
		self._validate_expressibility()

	def _live_candidates(self) -> list[str]:
		return [
			function.name
			for function in self.functions
			if function.name.startswith(self._acc)
		]

	def get_state(self) -> str:
		return self._state

	def _is_expressible(self, target: str) -> bool:
		queue = deque([target])
		visited = set()

		while queue:
			remaining = queue.popleft()
			if remaining == "":
				return True
			if remaining in visited:
				continue
			visited.add(remaining)

			for token_id in self._continuation_ids(remaining):
				piece = self._pieces[token_id]
				assert piece is not None
				queue.append(remaining[len(piece):])

		return False

	def _validate_expressibility(self) -> None:
		targets = ["name", "parameters"]

		for function in self.functions:
			targets.extend(function.parameters.keys())
			targets.append(function.name)

		for target in targets:
			if not self._is_expressible(target):
				raise ValueError(
					f"Vocabulary cannot express {target}"
				)

	def _continuation_ids(self, remaining: str) -> set[int]:
		return {
			token_id
			for token_id, piece in enumerate(self._pieces)
			if piece is not None and remaining.startswith(piece)
		}

	def _is_valid_string_piece(self, piece: str) -> bool:
		return (
			len(piece) > 0
			and '"' not in piece
			and "\\" not in piece
			and all(ord(char) >= 0x20 for char in piece)
		)

	def _count_digits(self, piece: str) -> int:
		return sum("0" <= char <= "9" for char in piece)

	def _limited_numeric_mask(
		self,
		remaining_digits: int,
		numeric_mask: np.ndarray,
	) -> np.ndarray:
		if remaining_digits <= 0:
			return np.zeros(
				self.vocab.get_vocab_size(),
				dtype=bool,
			)

		return (
			np.asarray(numeric_mask, dtype=bool)
			& (self._piece_digit_counts <= remaining_digits)
		)

	def _limited_string_mask(
		self,
		remaining_chars: int,
	) -> np.ndarray:
		if remaining_chars <= 0:
			return np.zeros(
				self.vocab.get_vocab_size(),
				dtype=bool,
			)

		return (
			np.asarray(
				self.vocab.get_secure_mask(),
				dtype=bool,
			)
			& self._string_piece_mask
			& (self._piece_lengths > 0)
			& (self._piece_lengths <= remaining_chars)
		)

	def _can_close_number(self) -> bool:
		return _JSON_NUMBER_RE.fullmatch(self._value_acc) is not None

	def mask_for_next(self) -> np.ndarray:
		ids: set[int] = set()

		if self._state == _State.DONE:
			raise ValueError("Masking next when state already done")

		elif self._state == _State.OPEN:
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
				ids.add(self._sym['"'])
			elif 0 <= self._key_pos < len(self._key_target):
				ids.update(
					self._continuation_ids(
						self._key_target[self._key_pos:]
					)
				)
			elif self._key_pos == len(self._key_target):
				ids.add(self._sym['"'])

		elif self._state == _State.FNAME:
			if self._name_started is False:
				ids.add(self._sym['"'])
			else:
				candidates = self._live_candidates()

				if len(candidates) == 0:
					raise ValueError("No valid candidates")

				if (
					len(candidates) == 1
					and self._acc == candidates[0]
				):
					ids.add(self._sym['"'])
				else:
					for candidate in candidates:
						ids.update(
							self._continuation_ids(
								candidate[len(self._acc):]
							)
						)

		elif self._state == _State.END_VALUE:
			if self._selected is None:
				raise ValueError("Function is not selected")

			if self._param_index + 1 < len(
				self._selected.parameters
			):
				ids.add(self._sym[","])

			ids.add(self._sym["}"])

		elif self._state == _State.VALUE_STR:
			ids.add(self._sym['"'])

		elif self._state == _State.VALUE_NUM:
			if self._value_started and self._can_close_number():
				if self._param_index + 1 < len(
					self._selected.parameters
				):
					ids.add(self._sym[","])
				else:
					ids.add(self._sym["}"])

			if (
				self._value_started
				and self._nb_digits >= _MAX_NUMBER_DIGITS
				and not self._can_close_number()
			):
				raise ValueError(
					"Cannot continue incomplete number"
				)

		mask = np.zeros(
			self.vocab.get_vocab_size(),
			dtype=bool,
		)
		mask[list(ids)] = True

		if (
			self._state == _State.VALUE_STR
			and self._value_started is True
		):
			remaining_chars = _MAX_STRING_LENGTH - self._nb_chars
			if remaining_chars > 0:
				mask = mask | self._limited_string_mask(
					remaining_chars
				)

		elif self._state == _State.VALUE_NUM:
			if self._value_started is False:
				mask = mask | self._limited_numeric_mask(
					_MAX_NUMBER_DIGITS,
					self.vocab.get_numeric_mask(),
				)
			elif self._nb_digits < _MAX_NUMBER_DIGITS:
				mask = mask | self._limited_numeric_mask(
					_MAX_NUMBER_DIGITS - self._nb_digits,
					self.vocab.get_numeric_mask(),
				)

		return mask

	def is_done(self) -> bool:
		return self._state == _State.DONE

	def name(self) -> str:
		if not self.is_done():
			raise ValueError(
				"Tried to get name when state isn't done"
			)
		if self._selected is None:
			raise ValueError("Function is not selected")
		return self._selected.name

	def parameters(self) -> dict[str, Any]:
		return self._params

	def _parse_number(self) -> float:
		return float(self._value_acc)

	def advance(self, piece: str) -> None:
		if self._state == _State.DONE:
			raise ValueError("Tried to advance out of range")

		elif self._state == _State.OPEN:
			self._state = _State.KEY
			self._key_target = "name"
			self._key_pos = -1

		elif self._state == _State.KEY:
			if self._key_pos == -1:
				self._key_pos = 0
			elif 0 <= self._key_pos < len(self._key_target):
				self._key_pos += len(piece)
			elif self._key_pos == len(self._key_target):
				self._state = _State.COLON

		elif self._state == _State.COLON:
			if (
				self._selected is None
				and self._key_target == "name"
			):
				self._state = _State.FNAME
				self._acc = ""
				self._name_started = False

			elif self._key_target == "parameters":
				self._state = _State.OPEN_PARAMS
				self._key_target = "parameters"

			else:
				if self._selected is None:
					raise ValueError("Function is not selected")

				parameter = self._selected.parameters.get(
					self._key_target
				)
				if parameter is None:
					raise ValueError(
						f"Unknown parameter {self._key_target}"
					)

				if parameter.type == "string":
					self._state = _State.VALUE_STR
				else:
					self._state = _State.VALUE_NUM

				self._value_acc = ""
				self._value_started = False
				self._nb_digits = 0
				self._nb_chars = 0

		elif self._state == _State.FNAME:
			if not self._name_started:
				self._name_started = True
			elif piece != '"':
				self._acc += piece
			else:
				candidates = self._live_candidates()
				if (
					len(candidates) != 1
					or self._acc != candidates[0]
				):
					raise ValueError("Invalid function name")

				selected = next(
					(
						function
						for function in self.functions
						if function.name == self._acc
					),
					None,
				)
				if selected is None:
					raise ValueError("Function is not selected")

				self._selected = selected
				self._key_target = "parameters"
				self._key_pos = -1
				self._state = _State.AFTER_NAME

		elif self._state == _State.AFTER_NAME:
			self._state = _State.KEY

		elif self._state == _State.OPEN_PARAMS:
			if self._selected is None:
				raise ValueError("Function is not selected")

			if not self._selected.parameters:
				self._state = _State.CLOSE
			else:
				self._param_index = 0
				self._key_target = list(
					self._selected.parameters
				)[0]
				self._key_pos = -1
				self._state = _State.KEY

		elif self._state == _State.VALUE_STR:
			if not self._value_started:
				if piece != '"':
					raise ValueError(
						"String value must start with a quote"
					)
				self._value_started = True
				self._nb_chars = 0

			elif piece == '"':
				self._params[self._key_target] = self._value_acc
				self._value_acc = ""
				self._value_started = False
				self._nb_chars = 0
				self._state = _State.END_VALUE

			else:
				piece_length = len(piece)

				if (
					piece_length == 0
					or not self._is_valid_string_piece(piece)
				):
					raise ValueError("Invalid string piece")

				if (
					self._nb_chars + piece_length
					> _MAX_STRING_LENGTH
				):
					raise ValueError(
						"String exceeds maximum length"
					)

				self._value_acc += piece
				self._nb_chars += piece_length

		elif self._state == _State.VALUE_NUM:
			if not self._value_started:
				if _JSON_NUMBER_RE.fullmatch(piece) is None:
					raise ValueError(
						"Invalid first numeric piece"
					)

				piece_digits = self._count_digits(piece)
				if piece_digits > _MAX_NUMBER_DIGITS:
					raise ValueError(
						"Number exceeds maximum digits"
					)

				self._value_acc += piece
				self._nb_digits = piece_digits
				self._value_started = True

			elif piece == ",":
				if not self._can_close_number():
					raise ValueError(
						"Cannot close incomplete number"
					)

				self._params[self._key_target] = (
					self._parse_number()
				)
				self._value_acc = ""
				self._value_started = False
				self._nb_digits = 0
				self._nb_chars = 0
				self._param_index += 1
				self._key_pos = -1
				self._state = _State.KEY
				self._key_target = list(
					self._selected.parameters
				)[self._param_index]

			elif piece == "}":
				if not self._can_close_number():
					raise ValueError(
						"Cannot close incomplete number"
					)

				self._params[self._key_target] = (
					self._parse_number()
				)
				self._value_acc = ""
				self._value_started = False
				self._nb_digits = 0
				self._nb_chars = 0
				self._state = _State.CLOSE

			else:
				piece_digits = self._count_digits(piece)

				if (
					self._nb_digits + piece_digits
					> _MAX_NUMBER_DIGITS
				):
					raise ValueError(
						"Number exceeds maximum digits"
					)

				candidate = self._value_acc + piece
				if _JSON_NUMBER_RE.fullmatch(candidate) is None:
					raise ValueError(
						"Invalid numeric continuation"
					)

				self._value_acc += piece
				self._nb_digits += piece_digits

		elif self._state == _State.END_VALUE:
			if piece == ",":
				self._param_index += 1
				self._key_target = list(
					self._selected.parameters
				)[self._param_index]
				self._key_pos = -1
				self._state = _State.KEY

			elif piece == "}":
				self._state = _State.CLOSE

			else:
				raise ValueError(
					"Expected comma or closing brace"
				)

		elif self._state == _State.CLOSE:
			if piece != "}":
				raise ValueError(
					"Expected closing brace"
				)
			self._state = _State.DONE
