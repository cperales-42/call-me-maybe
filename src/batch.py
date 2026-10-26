"""Batch orchestration for sequential, single-sequence model inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, field_validator


T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor(BaseModel):
    """Split work into deterministic batches while preserving order."""

    batch_size: int = 1

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        """Validate that the batch size is positive.

        Args:
            value: Requested batch size.

        Returns:
            The validated batch size.

        Raises:
            ValueError: If the batch size is not positive.
        """
        if value < 1:
            raise ValueError("batch_size must be at least 1")
        return value

    def chunk(self, items: Sequence[T]) -> list[list[T]]:
        """Split items into consecutive batches.

        Args:
            items: Items to split.

        Returns:
            A list of batches, each containing between one and ``batch_size``
            items in the original order.
        """
        return [
            list(items[start:start + self.batch_size])
            for start in range(0, len(items), self.batch_size)
        ]

    def process(
        self,
        items: Sequence[T],
        operation: Callable[[T], R],
    ) -> list[R]:
        """Process every item in batches and return results in order.

        The public SDK exposes single-sequence inference, so batching here is
        request-level orchestration rather than tensor-level batching.

        Args:
            items: Items to process.
            operation: Callable applied to each item.

        Returns:
            Results in the same order as ``items``.

        Raises:
            Exception: Any exception raised by ``operation`` is propagated.
        """
        results: list[R] = []
        for batch in self.chunk(items):
            for item in batch:
                results.append(operation(item))
        return results
