"""Command-line interface for function-call generation."""

from __future__ import annotations

import argparse
import json
import os
from importlib import import_module
from typing import cast

from .generator import ModelProtocol, generate_prompt
from .grammar import Grammar
from .io_utils import IOUtils
from .models import FunctionCallResult, FunctionDefinition
from .vocabulary import Vocabulary


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
        A complete prompt containing the function definitions and user request.
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
        f"{QUOTE}name{QUOTE} and parameters{QUOTE}.\n"
        f"Available functions:\n{definitions}\n"
        f"User request: {prompt}\n"
        "JSON:"
    )


def run(
    funcs_path: str,
    tests_path: str,
    output_path: str,
) -> list[FunctionCallResult]:
    """Generate function-call results for every test case.

    Args:
        funcs_path: Path to the JSON function definitions.
        tests_path: Path to the JSON test cases.
        output_path: Path where generated results will be written.

    Returns:
        The generated results in test-case order.
    """
    parse_utils = IOUtils(
        functions_definition_path=funcs_path,
        test_cases_path=tests_path,
        output_path=output_path,
    )
    functions = parse_utils.load_functions_definition()
    test_cases = parse_utils.load_test_cases()
    model = cast(
        ModelProtocol,
        getattr(llm_sdk, "Small_LLM_Model")(),
    )
    test = "test prompt"
    ids = model.encode(test).tolist()[0]
    logits_size = len(model.get_logits_from_input_ids(ids))
    vocab = Vocabulary(
        vocab_path=model.get_path_to_vocab_file(),
        logits_size=logits_size,
    )
    results: list[FunctionCallResult] = []
    for case in test_cases:
        grammar = Grammar(vocab=vocab, functions=functions)
        inference_prompt = _build_inference_prompt(functions, case.prompt)
        result = generate_prompt(
            model,
            grammar,
            case.prompt,
            inference_prompt,
            512,
        )
        results.append(result)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    parse_utils.write_output(results)
    return results


def main() -> int:
    """Run the command-line interface.

    Returns:
        ``0`` on success or ``1`` if processing fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )
    args = parser.parse_args()
    try:
        run(args.functions_definition, args.input, args.output)
    except Exception as error:
        print(f"Error: {error}")
        return 1
    return 0
