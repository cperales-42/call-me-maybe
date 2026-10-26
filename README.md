*This project has been created as part of the 42 curriculum by caperale.*

# Call Me Maybe

## Description

**Call Me Maybe** converts natural-language requests into structured function calls with `Qwen/Qwen3-0.6B` and token-level constrained decoding. It reads the available functions and a list of prompts from JSON files, lets the model select one function per prompt, validates every generated token against a schema-aware grammar, and writes a JSON result containing:

- the original `prompt`;
- the selected function `name`;
- the typed function `parameters`.

The program does not execute the selected functions. It only produces a machine-readable call.

The implementation uses the public interface exposed by the bundled `llm_sdk`: `get_logits_from_input_ids`, `get_path_to_vocab_file`, and `get_path_to_merges_file`. Tokenization, generation, grammar state, caching, and input/output handling are implemented in this repository. Function definitions, test cases, results, tokenizer data, cache entries, and runtime configuration are represented with Pydantic models where appropriate.

### Implemented capabilities

- Reads configurable function-definition and test-case JSON files.
- Validates inputs with Pydantic and reports missing, malformed, inaccessible, or structurally invalid files through clear errors.
- Builds an inference prompt containing the available functions and the user request.
- Lets the model choose the function name and generate its argument values.
- Restricts every generated token with a grammar derived from the selected function definition.
- Supports `string`, `number`, `integer`, and `boolean` parameters.
- Handles vocabulary pieces that span several grammar states, such as `]",` or `"}}`.
- Skips model inference when the grammar leaves exactly one structurally forced token.
- Uses a repository-owned byte-level BPE tokenizer built from the model vocabulary and merge files.
- Provides an optional versioned result cache, request-level batching, and a real-time visualization mode.
- Includes a comprehensive `pytest` suite, `flake8`, and strict `mypy` checks.
- Writes one result per input prompt as atomic, valid JSON.
- Converts handled failures into clear messages and a nonzero exit status instead of crashing unexpectedly.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Access to the `Qwen/Qwen3-0.6B` model files when they are not already cached

No compilation step is required. The local development environment on a machine with an unsupported GPU may need CPU inference:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src
```

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

The program creates the output directory when it does not already exist. It writes to a temporary file and atomically replaces the destination, so a partial result is not left behind after a write failure.

### Optional features

```bash
# Reproduce the generation process token by token.
uv run python -m src --visualize

# Enable the versioned persistent result cache.
uv run python -m src --cache

# Group request-level processing in batches while preserving result order.
uv run python -m src --batch-size 4

# Same visualization through the Makefile.
make cli
```

Caching is disabled by default. When enabled, entries are keyed by the cache version, a normalized hash of the function definitions, and the exact prompt. Malformed or stale cache data is ignored or reported with a warning rather than silently changing generation behavior.

Batching is request-level orchestration. The public SDK exposes single-sequence inference, so a batch controls how prompts are grouped in the pipeline, not tensor-level parallel decoding.

### Other Make targets

```bash
make debug         # Launch the entry point through pdb
make lint          # Run flake8 and the configured mypy checks
make lint-strict   # Run flake8 and mypy --strict
make test          # Run the pytest suite
make clean         # Remove generated caches and output
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

Supported parameter types are `string`, `number`, `integer`, and `boolean`. Parameter order is preserved and follows the order in the JSON object.

### Test cases

`function_calling_tests.json` must contain an array of prompt objects:

```json
[
  { "prompt": "What is the sum of 2 and 3?" }
]
```

### Output

Results are serialized with `json.dump` and written atomically. For the first bundled example:

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

`number` values are represented as floats, `integer` values as integers, and `boolean` values as JSON booleans. Every result contains exactly `prompt`, `name`, and `parameters`.

## Algorithm: constrained decoding

The decoder combines model probabilities with a deterministic grammar. The model still chooses among valid tokens, but it cannot emit a token that would break the required JSON structure or parameter schema.

### 1. Load and validate inputs

`IOUtils` loads both input files and converts their entries into Pydantic models. JSON decoding errors, missing files, permission failures, and other filesystem errors are translated into descriptive `ValueError` exceptions. Duplicate function names and unsupported parameter types are rejected before generation starts.

### 2. Build the vocabulary indexes

`Vocabulary` reads the JSON vocabulary exposed by the model and creates:

- token-to-identifier and identifier-to-token maps;
- indexes for the required JSON symbols `{`, `}`, `:`, `,`, and `"`;
- a structural-token mask;
- a safe string-content mask;
- a numeric-token mask and a numeric-character mask.

Duplicate identifiers, duplicate decoded pieces, missing required symbols, out-of-range identifiers, malformed JSON, and incompatible vocabulary sizes are rejected.

`SimpleTokenizer` builds an independent byte-level BPE tokenizer from the model vocabulary and merge files. It implements the Qwen/GPT-2 pre-tokenization rules with the standard library, applies the real BPE merge ranks, and falls back to deterministic greedy longest-piece matching when no merge file is available. This keeps the generation code independent of the SDK tokenizer's public `encode`/`decode` methods.

### 3. Prepare each function-call grammar

A new `Grammar` is created for every prompt from the shared vocabulary and function definitions. Before generation starts, the grammar verifies that required keys and function names can be represented by the available tokenizer pieces.

The grammar is a finite-state machine with states for:

- opening the object and selecting the `name` key;
- matching one of the declared function names;
- opening the `parameters` object;
- selecting parameter keys in their declared order;
- generating string, numeric, or boolean values;
- separating values and closing the call.

The function name is chosen by the LLM through its generated tokens. No keyword, regular-expression, or heuristic matching selects a function in the generation loop.

### 4. Token-level and character-level validation

A vocabulary token may contain several characters and may cross grammar boundaries. For example, the decoded piece `]",` can finish a regular-expression string and emit the following comma, and `"}}` can close a string and both JSON objects. The grammar therefore consumes each token character by character.

Token consumption is atomic: if any character is invalid, the parser restores its previous state, including accumulated values. A rejected multi-character token cannot leave a partially updated call behind. Multi-character tokens containing a quote are added to the string mask only after a speculative parse proves that the whole token is valid from the current state.

### 5. Fast-path for forced structural tokens

Some states have exactly one structurally valid next token: opening braces, colons, mandatory commas or closing braces, and opening/closing quotes for keys and names. Before asking the model for logits, the generator checks `forced_next_token_id()`.

- If the token is forced, the generator obtains the exact vocabulary piece, advances the grammar, appends the identifier to the context, and continues without a model call.
- Otherwise, it requests logits and applies the grammar mask.
- If the mask contains exactly one allowed token, that token is also selected without a model call.
- Numeric values are deliberately never completed by the fast path: the model must decide whether to continue a number or emit its delimiter.

This preserves LLM-driven choices while eliminating unnecessary inference on deterministic JSON structure.

### 6. Select one constrained token at a time

For every non-forced step, the implementation:

1. obtains the model logits for the current token sequence;
2. asks the grammar for the identifiers that are valid in the current state;
3. converts `NaN` logits to negative infinity;
4. replaces the logits of every disallowed token with negative infinity;
5. selects the highest-scoring allowed token with `numpy.argmax`;
6. advances the grammar atomically using the selected vocabulary piece; and
7. appends the identifier to the input sequence for the next model call.

Generation fails explicitly when no allowed token exists, when logits do not match the vocabulary size, when a selected piece is missing, when a numeric value is incomplete or non-finite, or when generation does not finish within the token limit.

### 7. Serialize the completed call

Generation stops only after the closing brace has been consumed. The parser returns the selected function name and the values accumulated for its declared parameters. Pydantic validates and serializes the final result.

## Design decisions

- **Grammar instead of post-processing:** invalid structure is rejected during decoding rather than repaired after generation.
- **LLM-based selection:** the grammar narrows the choices, while model logits choose the function and values.
- **Deterministic selection:** `argmax` makes each decoding step reproducible for a given model state.
- **Character-level state transitions:** multi-character BPE pieces can finish strings and structural JSON in one token without forcing the model into an unnatural token trajectory.
- **Atomic advancement:** speculative parsing and rollback make token rejection safe.
- **Pydantic models:** the same models validate input data, hold runtime state, and serialize output.
- **Separate modules:** CLI, generation, grammar, tokenization, vocabulary, caching, batching, visualization, models, and file access have distinct responsibilities.
- **Reused model and vocabulary:** the model, tokenizer, and vocabulary are initialized once per run; only the grammar is recreated per prompt.
- **Explicit limits:** generated numbers are limited to 15 digits and string values to 256 characters to keep decoding bounded.
- **Public SDK interface only:** the project uses `get_logits_from_input_ids`, `get_path_to_vocab_file`, and `get_path_to_merges_file`; it does not call the SDK tokenizer's `encode` or `decode`.
- **Opt-in caching:** caching never changes default behavior and uses a versioned key to avoid stale results.

## Performance analysis

- **Accuracy:** function selection and argument extraction are performed by the LLM. The grammar guarantees the allowed structure and types, but cannot prove that a value has the correct semantic meaning. This repository does not claim a numeric semantic-accuracy benchmark.
- **Speed:** inference is autoregressive and processes one token at a time. The structural fast path removes model calls at deterministic JSON steps, and the result cache avoids regenerating identical requests when enabled. Runtime still depends mainly on the number of generated tokens and the available hardware. Batching is request-level orchestration, not parallel tensor inference.
- **Reliability:** structural validity is enforced at every decoding step. Final values are validated, non-finite numbers are rejected, token advancement is atomic, output writes are atomic, and cache failures are non-fatal.
- **Memory:** vocabulary indexes and masks are shared across prompts; each prompt only creates a lightweight grammar and token sequence.

## Challenges faced

- **Numeric arguments were truncated:** an early fast path forced a delimiter as soon as a number was syntactically complete, turning `265` into `2`. Numeric closure is now left to the model, and the fast path is reserved for unambiguous structural tokens.
- **Tokenizer pieces are not JSON characters:** model tokens can contain several characters, encoded leading spaces, quotes, or structural JSON. The grammar consumes tokens character by character and validates them atomically.
- **Quote-spanning tokens:** greedy decoding can select pieces that close a string and continue with JSON punctuation. Rejecting them derailed generation, so safe quote-containing tokens are now proved valid by speculative parsing before entering the string mask.
- **Function names are generated incrementally:** the decoder tracks all names compatible with the current prefix and permits only tokens that continue one of those candidates. Names sharing a prefix can still select the shorter name.
- **Parameters may be named like top-level keys:** an explicit key role distinguishes the function-name key, the `parameters` key, and parameter keys, so a parameter named `name` or `parameters` is not confused with top-level JSON.
- **Model logits can contain non-finite values:** `NaN` values are sanitized before masking so they cannot influence token selection.
- **Schema enforcement is different from value understanding:** the finite-state grammar guarantees a valid function call, but the model can still choose a valid argument value with incorrect meaning.

## Testing strategy

The repository includes unit and integration tests for tokenization, vocabulary indexes, grammar states and masks, atomic token rejection, forced tokens, generation, caching, batching, input/output validation, visualization, and end-to-end orchestration.

Run the full validation suite with:

```bash
uv run flake8 .
uv run mypy . --strict
uv run pytest -q
```

The test suite contains more than 130 tests and uses deterministic model doubles for integration coverage, so it does not require downloading model weights. A real end-to-end run is still useful for validating model behavior:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src
```

After execution, `data/output/function_calls.json` can be inspected to verify that:

- the file is valid JSON;
- every input prompt has one result;
- each result contains exactly `prompt`, `name`, and `parameters`;
- function names belong to the supplied definitions; and
- parameter names and value types match the selected function.

## Project structure

```text
.
├── src/
│   ├── __main__.py       # Package entry point
│   ├── main.py           # CLI and orchestration
│   ├── generator.py      # Constrained token-selection loop and events
│   ├── grammar.py        # JSON/function-call state machine and masks
│   ├── tokenizer.py      # Repository-owned byte-level BPE tokenizer
│   ├── vocabulary.py     # Token indexes and masks
│   ├── cache.py          # Versioned, optional result cache
│   ├── batch.py          # Request-level batch orchestration
│   ├── visualizer.py     # Real-time generation view
│   ├── io_utils.py       # JSON loading and atomic writing
│   └── models.py         # Pydantic data models
├── tests/                # Unit and integration tests
├── data/input/           # Demonstration input files
├── llm_sdk/              # Bundled model SDK
├── pyproject.toml        # Dependencies and tool configuration
├── uv.lock               # Locked dependency versions
└── Makefile              # Common project commands
```

## Resources and AI assistance

### AI assistance

AI assistance was used to implement the optimization plan, refactor the constrained-decoding grammar, build the tokenizer and tests, review the Python source for PEP 257 docstrings, and organize this README. The documented behavior follows the code in the repository; no benchmark result or unimplemented feature is claimed.

### References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model architecture, usage, and tokenizer information.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — model validation, validators, and serialization.
- [NumPy documentation](https://numpy.org/doc/stable/) — arrays, masks, `where`, and `argmax` used during constrained decoding.
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html) — command-line interface definition.
- [PEP 257](https://peps.python.org/pep-0257/) — Python docstring conventions.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) — docstring section layout used in the source.
