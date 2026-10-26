"""Real-time console visualization of constrained generation."""

from __future__ import annotations

import json

from pydantic import BaseModel

from .generator import GenerationEvent, generate_prompt
from .grammar import Grammar
from .models import FunctionCallResult, ModelProtocol
from .tokenizer import SimpleTokenizer


class CLIVisualizer(BaseModel):
    """Print generation events as they are produced."""

    show_scores: bool = True

    def start(self, prompt: str) -> None:
        """Print the header for a prompt.

        Args:
            prompt: Natural-language request being processed.
        """
        print(f"\n[visualize] prompt: {prompt}", flush=True)

    def on_event(self, event: GenerationEvent) -> None:
        """Print one token-selection event.

        Args:
            event: Event emitted by the constrained generator.
        """
        score = ""
        if self.show_scores and event.score is not None:
            score = f" score={event.score:.4f}"
        print(
            f"[step {event.step:03d}] state={event.state:<10} "
            f"source={event.source:<6} allowed={event.allowed_tokens:<3} "
            f"token={event.token_id} piece={event.piece!r}{score}",
            flush=True,
        )

    def finish(self, result: FunctionCallResult) -> None:
        """Print the completed function call.

        Args:
            result: Completed function-call result.
        """
        payload = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
        )
        print(f"[visualize] result: {payload}", flush=True)


def visualize_generation(
    model: ModelProtocol,
    grammar: Grammar,
    prompt: str,
    inference_prompt: str,
    max_iter: int,
    *,
    tokenizer: SimpleTokenizer | None = None,
) -> FunctionCallResult:
    """Generate a call while printing its token-by-token progress.

    Args:
        model: Model used to produce next-token logits.
        grammar: Grammar that constrains generation.
        prompt: Original user request stored in the result.
        inference_prompt: Expanded prompt supplied to the model.
        max_iter: Maximum number of tokens to generate.
        tokenizer: Optional shared tokenizer.

    Returns:
        The completed and validated function-call result.
    """
    visualizer = CLIVisualizer()
    visualizer.start(prompt)
    result = generate_prompt(
        model,
        grammar,
        prompt,
        inference_prompt,
        max_iter,
        tokenizer=tokenizer,
        observer=visualizer.on_event,
    )
    visualizer.finish(result)
    return result
