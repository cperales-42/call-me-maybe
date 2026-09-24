from __future__ import annotations
import json
import re
import numpy as np
from pydantic import BaseModel, model_validator

class Vocabulary(BaseModel):
    vocab_path: str
    logits_size: int
    _direct_map: list[str | None]
    _inverse_map: dict[str, int]
    _symbols: dict[str, int]
    _structural_mask: np.ndarray
    _secure_mask: np.ndarray
    _numeric_mask: np.ndarray
    _valid_json_re: re.Pattern = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")


    @model_validator(mode="after")
    def _load(self) -> Vocabulary:
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as fnf:
            raise ValueError(f"file not found: {fnf}")
        except json.JSONDecodeError as je:
            raise ValueError(f"broken json: {je}")
        except PermissionError as pe:
            raise ValueError(f"Permission denied for file: {pe}")
        except OSError as e:
            raise ValueError(f"File error: {e}")
        
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
        for symbol in ["{", "}", ":", ",", '"', "[", "]"]:
            if symbol not in self._inverse_map:
                raise ValueError(f"symbol not in json: {symbol}")
            self._symbols[symbol] = self._inverse_map[symbol]
        _ids = set(self._symbols.values())
        self._structural_mask = np.array([i in _ids for i in range(self.logits_size)], dtype=bool)
        self._secure_mask = np.array([piece is not None and '"' not in piece and '\\' not in piece for piece in self._direct_map], dtype=bool)
        self._numeric_mask = np.array([piece is not None and self._is_json_number(piece) for piece in self._direct_map], dtype=bool)
        
        return self
    
    @staticmethod
    def _normalize_piece(piece: str) -> str:
        return piece.replace("Ġ", " ")


    def _is_json_number(self, piece: str) -> bool:
        return self._valid_json_re.fullmatch(piece) is not None


    def piece_of(self, id_: int) -> str | None:
        if 0 <= int(id_) < self.logits_size:
            piece = self._direct_map[id_]
        else:
            raise ValueError("Index out of range")
        return piece
    

    def id_of(self, piece: str) -> int | None:
        try:
            id_ = self._inverse_map[piece]
        except KeyError:
            return None
        return id_
    

    def ids_for_prefix(self, prefix: str) -> list[int]:
        return [i for i, piece in enumerate(self._direct_map) if piece is not None and piece.startswith(prefix)]
    

    def structural_ids(self) -> set[int]:
        return set(self._symbols.values())
    

    def get_structural_mask(self) -> np.ndarray:
        return self._structural_mask
    

    def get_secure_mask(self) -> np.ndarray:
        return self._secure_mask
    

    def get_numeric_mask(self) -> np.ndarray:
        return self._numeric_mask
    

    def get_vocab_size(self) -> int:
        return self.logits_size