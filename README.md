*This project has been created as part of the 42 curriculum by caperale.*

# Call Me Maybe

## Description

**Call Me Maybe** converts natural-language requests into structured function calls by using `Qwen/Qwen3-0.6B` and token-level constrained decoding. The program receives the available functions and a set of prompts from JSON files, lets the model choose one function per prompt, and writes the selected name and arguments to a JSON output file.

The project does not execute the selected functions. Its responsibility is to produce a machine-readable call containing:

- the original `prompt`;
- the selected function `name`;
- the typed function `parameters`.

The implementation uses the bundled `llm_sdk` to access the model and its vocabulary. Function definitions, test cases, results, grammar state, vocabulary data, and I/O configuration are represented with Pydantic models.

### Implemented capabilities

- Reads function definitions and test cases from configurable JSON files.
- Validates inputs with Pydantic and reports missing, malformed, inaccessible, or structurally invalid files through clear errors.
- Builds an inference prompt containing the available functions and the user request.
- Uses the LLM to choose the function name and generate its argument values.
- Restricts every generated token with a grammar derived from the selected function definition.
- Guarantees the completed response structure: `prompt`, `name`, and `parameters`.
- Supports flat function definitions with string and numeric arguments.
- Writes one result per input prompt as valid JSON.
- Provides default paths, custom CLI paths, Make targets, type hints, and function/method docstrings.
- Catches top-level execution errors and exits with a nonzero status instead of crashing unexpectedly.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Access to the `Qwen/Qwen3-0.6B` model files when they are not already cached

No compilation step is required.

### Installation

Install and synchronize the project environment from the repository root:

```bash
uv sync
```

The same operation is available through the Makefile:

```bash
make install
```

### Run with the bundled examples

```bash
uv run python -m src
```

The default paths are:

- function definitions: `data/input/functions_definition.json`
- test cases: `data/input/function_calling_tests.json`
- output: `data/output/function_calls.json`

The equivalent Make target is:

```bash
make run
```

### Run with custom files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The program creates the output directory when it does not already exist.

### Other Make targets

```bash
make debug         # Launch the entry point through pdb
make lint          # Run flake8 and the configured mypy checks
make lint-strict   # Run flake8 and mypy with strict mode
make clean         # Remove generated caches
```

## Input and output format

### Function definitions

`functions_definition.json` must contain a JSON array. Each object provides a function name, description, ordered parameters, and return type:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

### Test cases

`function_calling_tests.json` must contain an array of prompt objects:

```json
[
  { "prompt": "What is the sum of 2 and 3?" }
]
```

### Output

The program serializes the generated results with `json.dump`. For the first bundled example, the corresponding result is:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Algorithm: constrained decoding

The decoder combines the model probabilities with a deterministic grammar. The model still decides which valid token to select, but it cannot emit a token that would break the required JSON structure or parameter schema.

### 1. Load and validate the inputs

`IOUtils` loads both input files and converts their entries into Pydantic models. JSON decoding errors, missing files, permission failures, and other filesystem errors are translated into descriptive `ValueError` exceptions.

### 2. Build the vocabulary indexes

`Vocabulary` reads the tokenizer vocabulary exposed by `llm_sdk` and creates:

- a token-to-identifier map;
- an identifier-to-token map;
- indexes for required JSON symbols;
- a structural-token mask;
- a safe string-token mask;
- a valid numeric-token mask.

Duplicate identifiers, duplicate pieces, missing symbols, out-of-range identifiers, malformed JSON, and incompatible vocabulary sizes are rejected.

### 3. Prepare each function-call grammar

A new `Grammar` is created for every prompt from the shared vocabulary and function definitions. Before generation starts, the grammar verifies that the required JSON keys and function names can be represented by the available tokenizer pieces.

The grammar is a finite-state machine with states for:

- opening the object and selecting the `name` key;
- matching one of the declared function names;
- opening the `parameters` object;
- selecting parameter keys in their declared order;
- generating string or numeric values;
- separating values and closing the call.

The function name is therefore chosen by the LLM through its generated tokens. No keyword, regular-expression, or heuristic matching selects a function in the main generation loop.

### 4. Select one constrained token at a time

For every generation step, the implementation:

1. obtains the model logits for the current token sequence;
2. asks the grammar for the identifiers that are valid in the current state;
3. converts `NaN` logits to negative infinity;
4. replaces the logits of every disallowed token with negative infinity;
5. selects the highest-scoring allowed token with `numpy.argmax`;
6. advances the grammar using the selected vocabulary piece; and
7. appends the identifier to the input sequence for the next model call.

A generation step cannot continue when no allowed token exists, when the logits do not match the vocabulary size, when a selected piece is missing, or when a numeric value is non-finite. The CLI converts these failures into a user-facing error and returns a nonzero exit code.

### 5. Serialize the completed call

Generation stops only after the closing brace has been consumed. The parser then returns the selected function name and the values accumulated for its declared parameters. Pydantic serializes those values into the final result object.

## Design decisions

- **Grammar instead of post-processing:** invalid structure is rejected during decoding rather than repaired after generation.
- **LLM-based selection:** the grammar narrows the choices, while model logits choose the function and values.
- **Deterministic selection:** `argmax` makes each decoding step reproducible for a given model state.
- **Pydantic models:** the same models validate input data, hold runtime state, and serialize output.
- **Separate modules:** CLI handling, generation, grammar logic, vocabulary handling, models, and file access have distinct responsibilities.
- **Reused model and vocabulary:** the model and vocabulary are initialized once per run; only the grammar is recreated for each prompt.
- **Explicit limits:** generated numbers are limited to nine digits and string values to 128 characters to keep the decoder bounded.
- **Public SDK interface only:** the project uses `encode`, `get_logits_from_input_ids`, and `get_path_to_vocab_file` from the provided SDK.

## Performance analysis

- **Accuracy:** function selection and argument extraction are performed by the LLM. The grammar guarantees the allowed structure and types, but it cannot prove that a value has the correct semantic meaning. The repository does not include a reproducible accuracy benchmark, so no numeric accuracy claim is made here.
- **Speed:** inference is autoregressive and processes one token at a time. Prompts share the loaded model and vocabulary, but generation is sequential and does not use batching or result caching. Runtime therefore depends mainly on the number of generated tokens and the available hardware; no unmeasured speed claim is made.
- **Reliability:** structural validity is enforced during every decoding step. The final object contains only the required keys, selected parameters are built from the function definition, non-finite numbers are rejected, and output is serialized with Python's JSON encoder.

## Challenges faced

- **Tokenizer pieces are not JSON characters:** model tokens can contain several characters or encoded leading spaces. The vocabulary normalizes token text and uses prefix matching to determine which pieces can continue a required string.
- **Function names are generated incrementally:** the decoder tracks all names compatible with the current prefix and permits only tokens that continue one of those candidates.
- **Numbers are produced in multiple pieces:** a piece may finish a number, extend an exponent, or continue a fractional value. The grammar permits a delimiter only when the accumulated text is already a complete JSON number.
- **Model logits can contain non-finite values:** `NaN` values are sanitized before masking so that they cannot influence token selection.
- **Schema enforcement is different from value understanding:** the finite-state grammar can guarantee a valid function call, but the model can still select a valid argument value with incorrect meaning. This distinction is kept explicit rather than presenting structural validity as semantic accuracy.

## Testing strategy

The project uses the bundled example files for end-to-end validation and static checks for code quality:

```bash
uv sync
make lint
uv run python -m src
```

After execution, `data/output/function_calls.json` can be inspected to verify that:

- the file is valid JSON;
- every input prompt has one result;
- each result contains exactly `prompt`, `name`, and `parameters`;
- function names belong to the supplied definitions; and
- parameter names and value types match the selected function.

`make lint` currently passes both Flake8 and the configured mypy checks. `make lint-strict` is also available for stricter static analysis.

## Project structure

```text
.
├── src/
│   ├── __main__.py       # Package entry point
│   ├── main.py           # CLI and orchestration
│   ├── generator.py      # Constrained token-selection loop
│   ├── grammar.py        # JSON/function-call state machine
│   ├── vocabulary.py     # Token indexes and masks
│   ├── io_utils.py       # JSON loading and writing
│   └── models.py         # Pydantic data models
├── data/input/           # Demonstration input files
├── llm_sdk/              # Bundled model SDK
├── pyproject.toml        # Dependencies and tool configuration
├── uv.lock               # Locked dependency versions
└── Makefile              # Common project commands
```

## Resources and AI assistance

### AI assistance

AI was used for repetitive documentation work: reviewing the Python source to add PEP 257 Google-style function and method docstrings, and organizing this README from the supplied project subject. The README describes the behavior present in the repository and does not claim automated tests, benchmark results, or bonus features that have not been implemented.

### References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model architecture, usage, and tokenizer information.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — model validation, validators, and serialization.
- [NumPy documentation](https://numpy.org/doc/stable/) — arrays, masks, `where`, and `argmax` used during constrained decoding.
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html) — command-line interface definition.
- [PEP 257](https://peps.python.org/pep-0257/) — Python docstring conventions.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) — docstring section layout used in the source.
