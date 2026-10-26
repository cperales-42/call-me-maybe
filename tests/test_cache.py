"""Tests for the versioned result cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cache import ResultCache
from src.models import FunctionCallResult


def _result() -> FunctionCallResult:
    """Build a valid result for cache tests.

    Returns:
        A simple function-call result.
    """
    return FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0},
    )


def test_set_and_get_roundtrip(tmp_path: Path) -> None:
    cache = ResultCache(cache_path=str(tmp_path / "cache.json"))
    cache.set("functions", "prompt", _result())
    assert cache.size() == 1
    cached = cache.get("functions", "prompt")
    assert cached == _result()
    assert cache.get("functions", "other") is None
    assert cache.get("other", "prompt") is None


def test_persistence_reloads_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    first = ResultCache(cache_path=str(path))
    first.set("functions", "prompt", _result())
    assert path.exists()

    second = ResultCache(cache_path=str(path))
    assert second.get("functions", "prompt") == _result()


def test_in_memory_cache_without_path() -> None:
    cache = ResultCache()
    cache.set("functions", "prompt", _result())
    assert cache.get("functions", "prompt") == _result()
    cache.clear()
    assert cache.size() == 0


def test_clear_removes_persistent_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = ResultCache(cache_path=str(path))
    cache.set("functions", "prompt", _result())
    cache.clear()
    assert not path.exists()
    assert cache.size() == 0


def test_invalid_entry_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = ResultCache(cache_path=str(path))
    cache.set("functions", "prompt", _result())
    raw = json.loads(path.read_text(encoding="utf-8"))
    key = next(iter(raw))
    raw[key] = {"prompt": "x", "name": "", "parameters": {}}
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = ResultCache(cache_path=str(path))
    assert reloaded.get("functions", "prompt") is None


def test_unreadable_cache_file_warns(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unreadable cache"):
        cache = ResultCache(cache_path=str(path))
    assert cache.size() == 0


def test_non_object_cache_warns(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="not an object"):
        cache = ResultCache(cache_path=str(path))
    assert cache.size() == 0


def test_cache_write_failure_warns(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = ResultCache(cache_path=str(blocker / "cache.json"))
    with pytest.warns(RuntimeWarning, match="could not write cache"):
        cache.set("functions", "prompt", _result())
    assert cache.get("functions", "prompt") == _result()


def test_cache_remove_failure_warns(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = ResultCache(cache_path=str(path))
    cache.set("functions", "prompt", _result())
    path.unlink()
    path.mkdir()
    with pytest.warns(RuntimeWarning, match="could not remove cache"):
        cache.clear()


def test_functions_hash_is_stable_and_order_independent() -> None:
    first = [{"name": "fn", "parameters": {"a": {"type": "number"}}}]
    second = [{"parameters": {"a": {"type": "number"}}, "name": "fn"}]
    assert ResultCache.hash_functions_def(first) == (
        ResultCache.hash_functions_def(second)
    )
    assert ResultCache.hash_functions_def(first) != (
        ResultCache.hash_functions_def([{"name": "other"}])
    )


def test_corrupt_cache_value_types_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"key": 3}), encoding="utf-8")
    cache = ResultCache(cache_path=str(path))
    assert cache.size() == 0
