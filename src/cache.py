"""Persistent, versioned cache for generated function calls."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Any

from pydantic import BaseModel, PrivateAttr, ValidationError

from .models import FunctionCallResult


CACHE_VERSION = 4


class ResultCache(BaseModel):
    """In-memory cache with optional atomic JSON persistence."""

    cache_path: str | None = None

    _entries: dict[str, dict[str, Any]] = PrivateAttr(
        default_factory=dict
    )

    def model_post_init(self, __context: Any) -> None:
        """Load an existing cache file when one is configured.

        Args:
            __context: Pydantic initialization context, which is not used.
        """
        if self.cache_path is None or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            warnings.warn(
                f"ignoring unreadable cache file {self.cache_path!r}: "
                f"{error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if not isinstance(data, dict):
            warnings.warn(
                f"ignoring cache file {self.cache_path!r}: "
                "top-level value is not an object",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, dict):
                self._entries[key] = value

    @staticmethod
    def hash_functions_def(functions_def: list[dict[str, Any]]) -> str:
        """Hash a normalized function definition list.

        Args:
            functions_def: Function definitions serialized as dictionaries.

        Returns:
            A SHA-256 hexadecimal digest.
        """
        content = json.dumps(
            functions_def,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _make_key(
        self,
        functions_def_hash: str,
        prompt: str,
    ) -> str:
        """Create a versioned cache key.

        Args:
            functions_def_hash: Hash of the function definitions.
            prompt: Input prompt.

        Returns:
            A SHA-256 hexadecimal digest.
        """
        content = json.dumps(
            {
                "version": CACHE_VERSION,
                "functions": functions_def_hash,
                "prompt": prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(
        self,
        functions_def_hash: str,
        prompt: str,
    ) -> FunctionCallResult | None:
        """Return a cached result when it is valid.

        Args:
            functions_def_hash: Hash of the function definitions.
            prompt: Input prompt.

        Returns:
            The cached result, or ``None`` when absent or invalid.
        """
        key = self._make_key(functions_def_hash, prompt)
        data = self._entries.get(key)
        if data is None:
            return None
        try:
            return FunctionCallResult.model_validate(data)
        except ValidationError:
            return None

    def set(
        self,
        functions_def_hash: str,
        prompt: str,
        result: FunctionCallResult,
    ) -> None:
        """Store a result and persist the cache atomically.

        Cache write failures are reported as warnings and never interrupt
        generation.

        Args:
            functions_def_hash: Hash of the function definitions.
            prompt: Input prompt.
            result: Result to cache.
        """
        key = self._make_key(functions_def_hash, prompt)
        self._entries[key] = result.model_dump(mode="json")
        self._flush()

    def _flush(self) -> None:
        """Write the cache to disk when persistence is configured."""
        if self.cache_path is None:
            return
        temporary_path = f"{self.cache_path}.tmp"
        try:
            parent = os.path.dirname(self.cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(self._entries, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.replace(temporary_path, self.cache_path)
        except OSError as error:
            warnings.warn(
                f"could not write cache file {self.cache_path!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clear in-memory entries and remove the persistent cache file."""
        self._entries.clear()
        if self.cache_path is None:
            return
        try:
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
        except OSError as error:
            warnings.warn(
                f"could not remove cache file {self.cache_path!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def size(self) -> int:
        """Return the number of cached entries.

        Returns:
            The number of entries currently held in memory.
        """
        return len(self._entries)
