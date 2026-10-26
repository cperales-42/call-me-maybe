*This project has been created as part of the 42 curriculum by caperale.*

# Call Me Maybe

## Description

**Call Me Maybe** converts natural-language requests into structured function calls with `Qwen/Qwen3-0.6B` and token-level constrained decoding. It reads the available functions and a list of prompts from JSON files, lets the model select one function per prompt, validates every generated token against a schema-aware grammar, and writes a JSON result containing:

- the original `prompt`;
- the selected function `name`;
- the typed function `parameters`.

The program does not execute the selected functions. It only produces a machine-readable call.

The implementation uses the public interface exposed by the bundled `llm_sdk`: `get_logits_from_input_ids`, `get_path_to_vocab_file`, and `get_path_to_merges_file`. Tokenization, generation, grammar state, caching, visualization, and input/output handling are implemented in this repository. Function definitions, test cases, results, tokenizer data, cache entries, and runtime configuration are represented with Pydantic models.

The grammar guarantees valid JSON and a schema-compliant call; the semantic correctness of the chosen function and argument values still depends on the model.

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
- Provides an optional versioned result cache.
- Provides an optional terminal dashboard that follows the generation token by token.
- Includes `flake8` and strict `mypy` checks through the Makefile.
- Writes one result per input prompt as atomic, valid JSON.
- Converts handled failures into clear messages and a nonzero exit status instead of crashing unexpectedly.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Access to the `Qwen/Qwen3-0.6B` model files when they are not already cached

No compilation step is required. If CUDA is unavailable or incompatible with the installed drivers, force CPU inference by clearing the visible devices:

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

### Example usage

#### Bundled inputs

From the repository root:

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

#### Custom input and output files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The program creates the output directory when it does not already exist. It writes to a temporary file and atomically replaces the destination, so a partial result is not left behind after a write failure. The output directory is generated at runtime and does not need to be committed.

#### Terminal visualization

```bash
uv run python -m src --visualize
```

When standard output is attached to an interactive terminal, the visualizer takes over the screen and updates a dashboard while each prompt is generated. The dashboard shows:

- the original prompt;
- the last token piece selected by the model;
- the last constraint forced by the grammar, visible briefly before the panel clears;
- the number of tokens allowed at the current step; and
- the compact JSON results accumulated for all prompts in the run.

The main column shows the JSON document being written, then the final formatted result. When the output is redirected or standard output is not a terminal, the visualizer prints a static final dashboard instead of animating. The `make cli` target runs the same mode.

#### Cache

```bash
# Enable the versioned persistent result cache.
uv run python -m src --cache

# Same operation through the Makefile.
make cache
```

Caching is disabled by default. When enabled, entries are keyed by the cache version, a normalized hash of the function definitions, and the exact prompt. Malformed or stale cache data is ignored or reported with a warning rather than silently changing generation behavior. Cache write failures never interrupt generation.

### Other Make targets

```bash
make cache         # Run with the persistent result cache enabled
make debug         # Launch the entry point through pdb
make lint          # Run flake8 and the configured mypy checks
make lint-strict   # Run flake8 and mypy --strict
make clean         # Remove generated caches and output
```

## Input and output format

### Function definitions

`functions_definition.json` must contain a JSON array. Each object provides a function name, description, ordered parameters, and return type:

```json
[
  {
    "name": "fn_multiply_numbers",
    "description": "Multiply two numbers together and return their product.",
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
  { "prompt": "What is the product of 3 and 5?" }
]
```

### Output

By default, results are written to `data/output/function_calls.json`; use `--output` to choose another path or filename. Results are serialized with `json.dump` and written atomically. For the first bundled prompt, the output entry is:

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 3.0,
      "b": 5.0
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
- **Separate modules:** CLI, generation, grammar, tokenization, vocabulary, caching, visualization, models, and file access have distinct responsibilities.
- **Reused model and vocabulary:** the model, tokenizer, and vocabulary are initialized once per run; only the grammar is recreated per prompt.
- **Explicit limits:** generated numbers are limited to 15 digits and string values to 256 characters to keep decoding bounded.
- **Public SDK interface only:** the project uses `get_logits_from_input_ids`, `get_path_to_vocab_file`, and `get_path_to_merges_file`; it does not call the SDK tokenizer's `encode` or `decode` or any private SDK member.
- **Opt-in caching:** caching never changes default behavior and uses a versioned key to avoid stale results.
- **Side-effect-free visualization:** the dashboard renders from the same generation events as normal output; when stdout is not an interactive terminal it prints a static final view.
- **Bounded generation:** every prompt has a token limit, and the grammar bounds numeric and string values.

## Implemented bonus features

The following optional features from the subject are implemented and can be exercised from the CLI:

- **Repository-owned tokenizer:** `SimpleTokenizer` builds a byte-level BPE tokenizer from the public vocabulary and merge paths and exposes public `encode` and `decode` methods. The generation pipeline uses this tokenizer instead of the SDK tokenizer.
- **Versioned result cache:** `--cache` persists results keyed by the cache version, a normalized function-definition hash, and the prompt.
- **Generation visualization:** `--visualize` renders the prompt, the current model token, the last forced constraint, the allowed-token count, and the accumulated results in a terminal dashboard.
- **Graceful error recovery:** malformed inputs, invalid schemas, unsupported parameter types, missing files, cache failures, and generation failures produce explicit messages instead of unhandled tracebacks.
- **Tokenizer/decoding integration:** model input comes from `SimpleTokenizer.encode`, and every generated vocabulary piece is mapped back to JSON text before the grammar accepts it.

The project does not implement model selection beyond the SDK-provided `Qwen/Qwen3-0.6B`, nested function arguments, or a shipped automated test suite.

## Performance analysis

- **Accuracy:** function selection and argument extraction are performed by the LLM. The grammar guarantees the allowed structure and types, but cannot prove that a value has the correct semantic meaning. This repository does not claim a numeric semantic-accuracy benchmark.
- **Speed:** inference is autoregressive and processes one token at a time. The structural fast path removes model calls at deterministic JSON steps, and the result cache avoids regenerating identical requests when enabled. Runtime still depends mainly on the number of generated tokens and the available hardware.
- **Reliability:** every emitted document is valid JSON and schema-compliant by construction. Final values are validated, non-finite numbers are rejected, token advancement is atomic, output writes are atomic, and cache failures are non-fatal.
- **Memory:** vocabulary indexes and masks are shared across prompts; each prompt only creates a lightweight grammar and token sequence.
- **Workload:** the bundled inputs contain 11 prompts and 6 function definitions. The five-minute target depends on model inference speed and the evaluation machine; no benchmark timing is stored in the repository.

## Challenges faced

- **Numeric arguments were truncated:** an early fast path forced a delimiter as soon as a number was syntactically complete, turning `265` into `2`. Numeric closure is now left to the model, and the fast path is reserved for unambiguous structural tokens.
- **Tokenizer pieces are not JSON characters:** model tokens can contain several characters, encoded leading spaces, quotes, or structural JSON. The grammar consumes tokens character by character and validates them atomically.
- **Quote-spanning tokens:** greedy decoding can select pieces that close a string and continue with JSON punctuation. Rejecting them derailed generation, so safe quote-containing tokens are now proved valid by speculative parsing before entering the string mask.
- **Function names are generated incrementally:** the decoder tracks all names compatible with the current prefix and permits only tokens that continue one of those candidates. Names sharing a prefix can still select the shorter name.
- **Parameters may be named like top-level keys:** an explicit key role distinguishes the function-name key, the `parameters` key, and parameter keys, so a parameter named `name` or `parameters` is not confused with top-level JSON.
- **Model logits can contain non-finite values:** `NaN` values are sanitized before masking so they cannot influence token selection.
- **Schema enforcement is different from value understanding:** the finite-state grammar guarantees a valid function call, but the model can still choose a valid argument value with incorrect meaning.

## Testing strategy

The subject marks test programs as non-graded, so the repository does not ship a `tests/` directory. Validation of the submitted code uses static analysis, end-to-end runs, and a manual output audit.

### Static analysis

```bash
make lint-strict
```

This runs `flake8 src` and `mypy src --strict`. Both must finish without errors. The looser `make lint` target uses the non-strict mypy flags listed in the subject.

### End-to-end runs

The bundled inputs exercise function selection, string, number, integer, and boolean parameters:

```bash
uv run python -m src
```

The same run can be repeated with the optional paths enabled:

```bash
uv run python -m src --cache
uv run python -m src --visualize
```

Error paths are checked with temporary files: missing input, invalid JSON, a non-array top level, duplicate function names, unsupported parameter types, and empty prompts. Each case must produce a clear message and a nonzero exit status without a traceback.

Visualization is checked both in an interactive terminal and with redirected output, where the static final dashboard must still be printed. The cache is checked by running the same request twice and confirming that the second run reuses the stored result.

### Output audit

After a real run, `data/output/function_calls.json` is inspected to verify that:

- the file is valid JSON;
- every input prompt has exactly one result;
- each result contains exactly `prompt`, `name`, and `parameters`;
- function names belong to the supplied definitions;
- parameter names and value types match the selected function; and
- no prose or extra keys appear anywhere in the output.

Because the grammar validates every token, structural validity does not depend on the prompt or on the model producing correct JSON. The audit focuses on semantic accuracy: whether the selected function and extracted values are the ones requested by the prompt.

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
│   ├── visualizer.py     # Real-time generation view
│   ├── io_utils.py       # JSON loading and atomic writing
│   └── models.py         # Pydantic data models
├── data/input/           # Demonstration input files
├── data/output/          # Generated results (created at runtime)
├── llm_sdk/              # Bundled model SDK
├── pyproject.toml        # Dependencies and tool configuration
├── uv.lock               # Locked dependency versions
└── Makefile              # Common project commands
```

## Resources

### AI usage

AI assistance was used for:

- reviewing the constrained-decoding design and the finite-state grammar;
- drafting and refactoring the repository-owned tokenizer and the vocabulary masks;
- implementing the optional result cache and terminal visualizer;
- reviewing the source for PEP 257 docstrings, `flake8`, and `mypy` compliance; and
- drafting and organizing this README.

All AI-generated code and text were reviewed, executed, and validated against the project requirements. No private members of `llm_sdk` are used.

### References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model architecture, usage, and tokenizer information.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — model validation, validators, and serialization.
- [NumPy documentation](https://numpy.org/doc/stable/) — arrays, masks, `where`, and `argmax` used during constrained decoding.
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html) — command-line interface definition.
- [RFC 8259: JSON](https://www.rfc-editor.org/rfc/rfc8259) — JSON syntax and validity rules.
- [Byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) — the tokenization algorithm reimplemented in `src/tokenizer.py`.
- [PEP 257](https://peps.python.org/pep-0257/) — Python docstring conventions.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) — docstring section layout used in the source.
