from __future__ import annotations
from typing import Any
from .vocabulary import Vocabulary
from .models import FunctionDefinition
from enum import Enum
from collections import deque

import numpy as np
from pydantic import BaseModel

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
	

	def	model_post_init(self, __context: Any) -> None:
		self._state = _State.OPEN
		self._pieces = [self.vocab.piece_of(i) for i in range(self.vocab.get_vocab_size())]
		self._sym = {}
		json_chars = ["{", "}", ":", ",", '"']
		for id_, char in enumerate(json_chars):
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
		self._params = {}
		self._validate_expressibility()

	def	_live_candidates(self) -> list[str]:
		return [f.name for f in self.functions if f.name.startswith(self._acc)]


	def	_is_expressible(self, target: str) -> bool:
		queue = deque([target])
		visited = set()
		while queue:
			rem = queue.popleft()
			if rem == "":
				return True
			if rem in visited:
				continue
			visited.add(rem)
			for id_ in self._continuation_ids(rem):
				piece = self._pieces[id_]
				assert piece is not None
				new_rem = rem[len(piece):]
				queue.append(new_rem)
		return False
	
	def	_validate_expressibility(self) -> None:
		targets = ["name", "parameters"]
		for f in self.functions:
			targets.extend(f.parameters.keys())
			targets.append(f.name)
		for t in targets:
			if not self._is_expressible(t):
				raise ValueError(f"Vocabulary cannot express {t}")


	def	_continuation_ids(self, rem: str) -> set[int]:
		return {i for i, piece in enumerate(self._pieces) if piece is not None and rem.startswith(piece)}
	

	def	mask_for_next(self) -> np.ndarray:
		ids = set()
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
				ids.update(self._continuation_ids(self._key_target[self._key_pos:]))
			elif self._key_pos == len(self._key_target):
				ids.add(self._sym['"'])
		elif self._state == _State.FNAME:
			if self._name_started is False:
				ids.add(self._sym['"'])
			else:
				candidates = self._live_candidates()
				if len(candidates) == 0:
					raise ValueError("No valid candidates")
				elif len(candidates) == 1 and self._acc == candidates[0]:
					ids.add(self._sym['"'])
				else:
					for c in candidates:
						ids.update(self._continuation_ids(c[len(self._acc):]))
		elif self._state == _State.END_VALUE:
			if self._param_index + 1 < len(self._selected.parameters):
				ids.add(self._sym[","])
			ids.add(self._sym["}"])
		elif self._state == _State.VALUE_STR:
			ids.add(self._sym['"'])
		elif self._state == _State.VALUE_NUM:
			if not self._value_started:
				pass
			elif self._param_index + 1 < len(self._selected.parameters):
				ids.add(self._sym[","])
				ids.add(self._sym["}"])
			else:
				ids.add(self._sym["}"])
		mask = np.zeros(self.vocab.get_vocab_size(), dtype=bool)
		mask[list(ids)] = True
		if self._state == _State.VALUE_STR and self._value_started is True:
			mask = mask | self.vocab.get_secure_mask()
		elif self._state == _State.VALUE_NUM:
			mask = mask | self.vocab.get_numeric_mask()
		return mask
	

	def	is_done(self) -> bool:
		return self._state == _State.DONE
	
	
	def	name(self) -> str:
		if not self.is_done():
			raise ValueError("Tried to get name when state isn't done")
		return self._selected.name
	

	def	parameters(self) -> dict[str, Any]:
		return self._params
	
	
	def	_parse_number(self) -> float:
		return float(self._value_acc)
	

	def	advance(self, piece: str) -> None:
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
			if self._selected is None and self._key_target == "name":
				self._state = _State.FNAME
				self._acc = ""
				self._name_started = False
			elif self._key_target == "parameters":
				self._state = _State.OPEN_PARAMS
				self._key_target = "parameters"
			else:
				if self._selected.parameters[self._key_target].type == "string":
					self._state = _State.VALUE_STR
				else:
					self._state = _State.VALUE_NUM
				self._value_acc = ""
				self._value_started = False
		elif self._state == _State.FNAME:
			if not self._name_started:
				self._name_started = True
			elif piece != '"':
				self._acc += piece
			elif len(self._live_candidates()) == 1 and self._acc == self._live_candidates()[0]:
				for func in self.functions:
					if func.name == self._acc:
						self._selected = func
				self._key_target = "parameters"
				self._key_pos = -1
				self._state = _State.AFTER_NAME
		elif self._state == _State.AFTER_NAME:
			self._state = _State.KEY
		elif self._state == _State.OPEN_PARAMS:
			self._param_index = 0
			self._key_target = list(self._selected.parameters)[0]
			self._key_pos = -1
			self._state = _State.KEY
		elif self._state == _State.VALUE_STR:
			if self._value_started is False:
				self._value_started = True
			elif piece != '"':
				self._value_acc += piece
			else:
				self._params[self._key_target] = self._value_acc
				self._value_acc = ""
				self._state = _State.END_VALUE
		elif self._state == _State.VALUE_NUM:
			if self._value_started is False:
				self._value_acc += piece
				self._value_started = True
			elif self._value_started is True:
				if piece == ",":
					self._params[self._key_target] = self._parse_number()
					self._value_acc = ""
					self._param_index += 1
					self._key_pos = -1
					self._state = _State.KEY
					self._key_target = list(self._selected.parameters)[self._param_index]
				elif piece == "}":
					self._params[self._key_target] = self._parse_number()
					self._state = _State.CLOSE
				else:
					self._value_acc += piece
				
		elif self._state == _State.END_VALUE:
			if piece == ",":
				self._param_index += 1
				self._key_target = list(self._selected.parameters)[self._param_index]
				self._key_pos = -1
				self._state = _State.KEY
			else:
				self._state = _State.CLOSE
		elif self._state == _State.CLOSE:
			self._state = _State.DONE