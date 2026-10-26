"""Command-line interface for constrained function-call generation."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from typing import cast

from .cache import ResultCache
from .generator import generate_prompt
from .grammar import Grammar
from .io_utils import IOUtils
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    ModelProtocol,
    TestCase,
)
from .tokenizer import SimpleTokenizer
from .vocabulary import Vocabulary


DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calls.json"
DEFAULT_CACHE_PATH = ".cache/call_me_maybe.json"
DEFAULT_MAX_ITER = 512
QUOTE = chr(34)
llm_sdk = import_module("llm_sdk")


def _build_inference_prompt(
    functions: list[FunctionDefinition],
    prompt: str,
) -> str:
    """Build the inference prompt for function-call generation.

    Args:
        functions: Function definitions available to the model.
        prompt: User request that describes the desired function call.

    Returns:
        A complete prompt containing definitions and the user request.
    """
    definitions = json.dumps(
        [function.model_dump(mode="json") for function in functions],
        ensure_ascii=False,
    )
    return (
        "You are a function-calling assistant. "
        "Select exactly one function from the available definitions. "
        "Use only the declared parameter names and types. "
        "Extract argument values exactly from the user request; "
        "do not invent, repeat, or append characters. "
        "Return only a JSON object with the keys "
        f"{QUOTE}name{QUOTE} and {QUOTE}parameters{QUOTE}.\n"
        f"Available functions:\n{definitions}\n"
        f"User request: {prompt}\n"
        "JSON:"
    )


def _load_model() -> ModelProtocol:
    """Instantiate the model exposed by the bundled SDK.

    Returns:
        The model wrapped through the public protocol.

    Raises:
        RuntimeError: If the SDK does not expose the expected model class.
    """
    try:
        model_class = getattr(llm_sdk, "Small_LLM_Model")
    except AttributeError as error:
        raise RuntimeError(
            "llm_sdk does not expose Small_LLM_Model"
        ) from error
    return cast(ModelProtocol, model_class())


def _infer_logits_size(
    model: ModelProtocol,
    tokenizer: SimpleTokenizer,
) -> int:
    """Infer the model logits size with a one-token probe.

    Args:
        model: Model used to produce logits.
        tokenizer: Tokenizer providing a valid probe identifier.

    Returns:
        The number of logits produced for one input token.

    Raises:
        ValueError: If the probe returns no logits.
    """
    probe_id = tokenizer.first_token_id()
    logits = model.get_logits_from_input_ids([probe_id])
    if not logits:
        raise ValueError("model returned no logits for the probe token")
    return len(logits)


def _result_for_case(
    case: TestCase,
    functions: list[FunctionDefinition],
    vocab: Vocabulary,
    model: ModelProtocol,
    tokenizer: SimpleTokenizer,
    max_iter: int,
    visualize: bool,
) -> FunctionCallResult:
    """Generate one result with contextual error reporting.

    Args:
        case: Test case to process.
        functions: Available function definitions.
        vocab: Shared vocabulary.
        model: Model used for inference.
        tokenizer: Shared tokenizer.
        max_iter: Maximum tokens generated per prompt.
        visualize: Whether to print generation events.

    Returns:
        The generated function-call result.

    Raises:
        RuntimeError: If generation fails for this prompt.
    """
    grammar = Grammar(vocab=vocab, functions=functions)
    inference_prompt = _build_inference_prompt(functions, case.prompt)
    try:
        if visualize:
            from .visualizer import visualize_generation

            return visualize_generation(
                model,
                grammar,
                case.prompt,
                inference_prompt,
                max_iter,
                tokenizer=tokenizer,
            )
        return generate_prompt(
            model,
            grammar,
            case.prompt,
            inference_prompt,
            max_iter,
            tokenizer=tokenizer,
        )
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"failed to generate a call for prompt {case.prompt!r}: {error}"
        ) from error


def run(
    funcs_path: str,
    tests_path: str,
    output_path: str,
    use_cache: bool = False,
    visualize: bool = False,
    *,
    model: ModelProtocol | None = None,
    tokenizer: SimpleTokenizer | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    max_iter: int = DEFAULT_MAX_ITER,
) -> list[FunctionCallResult]:
    """Generate function-call results for every test case.

    Args:
        funcs_path: Path to the JSON function definitions.
        tests_path: Path to the JSON test cases.
        output_path: Path where generated results will be written.
        use_cache: Whether to use the persistent result cache.
        visualize: Whether to print token-selection events.
        model: Optional pre-built model for testing or embedding.
        tokenizer: Optional pre-built tokenizer.
        cache_path: Persistent cache file path.
        max_iter: Maximum tokens generated per prompt.

    Returns:
        The generated results in test-case order.

    Raises:
        ValueError: If inputs or configuration are invalid.
        RuntimeError: If the model or generation fails.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    parse_utils = IOUtils(
        functions_definition_path=funcs_path,
        test_cases_path=tests_path,
        output_path=output_path,
    )
    functions = parse_utils.load_functions_definition()
    test_cases = parse_utils.load_test_cases()

    active_model = model if model is not None else _load_model()
    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else SimpleTokenizer.from_model(active_model)
    )
    logits_size = _infer_logits_size(active_model, active_tokenizer)
    vocab = Vocabulary(
        vocab_path=active_model.get_path_to_vocab_file(),
        logits_size=logits_size,
    )

    cache: ResultCache | None = None
    functions_hash = ""
    if use_cache:
        cache = ResultCache(cache_path=cache_path)
        functions_hash = ResultCache.hash_functions_def(
            [function.model_dump(mode="json") for function in functions]
        )

    def process_case(case: TestCase) -> FunctionCallResult:
        if cache is not None:
            cached = cache.get(functions_hash, case.prompt)
            if cached is not None:
                return cached
        result = _result_for_case(
            case,
            functions,
            vocab,
            active_model,
            active_tokenizer,
            max_iter,
            visualize,
        )
        if cache is not None:
            cache.set(functions_hash, case.prompt, result)
        return result

    results = [process_case(case) for case in test_cases]
    parse_utils.write_output(results)
    return results


def main() -> int:
    """Run the command-line interface.

    Returns:
        ``0`` on success, ``1`` on a handled failure, or ``130`` when
        interrupted.
    """
    parser = argparse.ArgumentParser(
        description="Generate function calls with constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the test cases JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated results JSON file.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable the versioned persistent result cache.",
    )
    parser.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Disable the result cache (default).",
    )
    parser.add_argument(
        "--visualize",
        "--cli",
        dest="visualize",
        action="store_true",
        default=False,
        help="Print the generation process in real time.",
    )

    args = parser.parse_args()
    try:
        results = run(
            args.functions_definition,
            args.input,
            args.output,
            use_cache=args.cache,
            visualize=args.visualize,
        )
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(results)} results to {args.output}")
    return 0
