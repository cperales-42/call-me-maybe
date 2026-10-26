# Plan 01: Optimizations and bonuses (structural fast-path, caching/batching, visualization CLI, tests, tokenizer recoding)

## 1. Objetivo
Mejorar el rendimiento del sistema de función-calling con constrained decoding añadiendo:
- Fast-path para tokens estructurales deterministas (no llamar al modelo cuando solo cabe `{`, `,`, `:`, `"`, `}` por estado del parser JSON).
- Soporte de caching (resultados por `(functions_definition_hash, test_case_prompt)` ) y batching (procesado agrupado cuando aplique) como funcionalidad bonus.
- Cliente/visualizador del proceso de generación activable/desactivable con flag, con regla opcional `make cli` para arrancarlo.
- Comprehensive test suite (bonus).
- Recodificación del tokenizer evitando uso directo de `encode`/`decode` en código principal, usando `get_logits_from_input_ids` y `get_path_to_vocab_file` (bonus), con demostración de integración con constrained decoding.

Todo ello manteniendo 100% JSON válido, sin heurísticas para elegir función, usando únicamente la interfaz pública de `llm_sdk`, sin dependencias prohibidas, con Pydantic, type hints, docstrings PEP 257, flake8 y `mypy --strict` pasando.

## 2. Contexto y estado actual (evidencia)

### 2.1 Flujo actual
- `src/main.py:68-102` (`run`): instancia `Small_LLM_Model` (`77-78`), obtiene `logits_size` vía `encode("test")` + `get_logits_from_input_ids` (`80-81`), crea `Vocabulary(vocab_path=model.get_path_to_vocab_file(), logits_size=logits_size)` (`82-85`). Por cada test case crea un nuevo `Grammar(vocab, functions)`, construye `inference_prompt`, llama `generate_prompt(...)` con `max_iter=512`, escribe resultados con `IOUtils.write_output`.
- `src/generator.py:47-114` (`generate_prompt`): codifica prompt con `model.encode(inference_prompt).tolist()[0]` (`76`). Bucle mientras `not grammar.is_done() and iteration < max_iter` (`78`): llama `model.get_logits_from_input_ids(ids)` (`79`), comprueba tamaño (`80-83`), obtiene máscara `grammar.mask_for_next()` (`84`), comprueba que haya algún True (`85-86`), sanea NaN a -inf (`88-93`), aplica máscara (`93`), elige `np.argmax` (`94`), valida máscara/token/pieza (`95-99`), incrementa iteración, `grammar.advance(piece)`, `ids.append(token)` (`100-102`). Al acabar valida floats finitos y devuelve `FunctionCallResult` (`103-113`).
- `src/grammar.py:299-425` (`mask_for_next`): construye máscara según estado. Muchos estados fuerzan **exactamente un token**: `OPEN` añade solo `{` (`315-316`), `COLON` solo `:` (`318-319`), `AFTER_NAME` solo `,` (`321-322`), `OPEN_PARAMS` solo `{` (`324-325`), `CLOSE` solo `}` (`327-328`). `KEY` avanza sobre clave requerida; `FNAME` permite prefijos/cierre de nombre; `END_VALUE` permite `,` o `}` según posición; `VALUE_STR` permite `"` inicial/final y tokens seguros con límite; `VALUE_NUM` permite cierre si número completo o tokens numéricos con presupuesto. `Grammar` valida expresibilidad de claves/nombres/175-191.
- `src/vocabulary.py:27-213`: carga vocab JSON vía `get_path_to_vocab_file()` contextualmente, normaliza `Ġ`->espacio (`104-115`), construye máscaras estructurales/seguras/numéricas, expone `get_vocab_size()`, `piece_of()`, `id_of()`, `ids_for_prefix()`, `get_*_mask()`. Requiere símbolos `{ } : , "` y opcionalmente `[` `]` (`76-79`).
- `src/models.py`: `ModelProtocol` define `encode`, `get_logits_from_input_ids`, `get_path_to_vocab_file` (generator:12-45). `llm_sdk.Small_LLM_Model` implementa esos métodos usando transformers (llm_sdk/llm_sdk/__init__.py:77-126).
- `pyproject.toml`: deps `llm-sdk`, `numpy>=2.0`, `pydantic>=2.0`; mypy estricto configurado; paquete `src`. `Makefile` tiene `install/run/debug/clean/lint/lint-strict` (sin `cli`).
- `README.md`: indica "uses the bundled `llm_sdk` to access the model and its vocabulary", "The project uses `encode`, `get_logits_from_input_ids`, and `get_path_to_vocab_file` from the provided SDK." (líneas ~16, 202). No menciona bonus implementados.

### 2.2 Problema identificado
1. **Llamadas innecesarias al modelo en tokens estructurales**: en estados con máscara de un solo elemento (o cuando la gramática determina token único), `generate_prompt` sigue llamando a `get_logits_from_input_ids`, aplicando máscara/argmax. Esto es coste puro innecesario. El subject pide explícitamente evitar pasar por modelo/logits cuando solo cabe un carácter específico (`{`, `,`, `:`, `"`, `}`) por parsing JSON.
2. **Sin caching ni batching**: cada test case genera secuencialmente, sin reutilización de resultados. Bonus pide ambos.
3. **Sin visualización**: no hay forma de ver "qué va pensando" en tiempo real; bonus pide cliente visualizable activable/desactivable con flag y `make cli` opcional.
4. **Sin comprehensive test suite**: solo validación manual descrita. Bonus lo pide.
5. **Uso directo de `encode`/`decode` en código principal**: `src/generator.py:76`, `src/main.py:80` usan `model.encode()`. Bonus pide evitarlo usando `get_logits_from_input_ids` y `get_path_to_vocab_file` (y poder demostrar integración). También `llm_sdk` tiene `decode` pero no se usa en src; hay que asegurar que no se use directamente.

### 2.3 Causa raíz
- Diseño actual: `generate_prompt` es agnóstico al hecho de que la máscara puede ser singleton; siempre consulta al modelo.
- Ausencia de infraestructura para cache/batch/visualización/tests.
- Inicialización usa `encode` para medir `logits_size` (main:80) - eso es uso directo de `encode` en código principal.
- No hay abstracción para tokenización alternativa basada en vocab file.

## 3. Restricciones
- **Obligatorias (subject):** Python 3.10+, flake8, mypy con type hints, docstrings PEP 257, Pydantic para clases cuando corresponda, errores manejados, sin torch/transformers/huggingface/outlines/dspy/equivalentes en nuestro código, usar solo interfaz pública de `llm_sdk` (nunca privados), no hardcodear prompts/funciones/valores/salidas, elección de función debe salir del LLM + máscaras constrained decoding (sin heurísticas/regex/keywords), `uv sync` debe seguir funcionando, mantener CLI `uv run python -m src [--functions_definition ...] [--input ...] [--output ...]` con mismos defaults.
- **Bonus objetivos:** no romper funcionalidad existente.
- **Prohibido:** tocar tests ajenos (no hay tests aún), modificar configuración lint para ocultar errores, silenciar con `# type: ignore`/`# noqa` salvo justificación explícita e imprescindible (evitar), añadir deps sin justificar y actualizar `pyproject.toml`/`uv.lock` coherentemente.
- **Estructurales:** solo escribir en `plans/`. No editar `src/`, `tests/`, `Makefile`, `pyproject.toml`, etc. (esto es tarea del builder siguiendo plan).

## 4. Diseño propuesto

### 4.1 Fast-path para tokens estructurales
**Idea:** cuando el siguiente token esté completamente determinado por el estado del parser (máscara con exactamente 1 elemento True, o gramática puede identificar token forzado), evitar llamar a `model.get_logits_from_input_ids`. En su lugar, obtener ese token/pieza directamente del vocabulario y avanzar la gramática.

**Dónde:** 
- Añadir método a `Grammar` para detectar token forzado: `forced_token_id() -> int | None` o `next_forced_piece() -> str | None`. Debe basarse en estado actual (OPEN/COLON/AFTER_NAME/OPEN_PARAMS/CLOSE) y también considerar casos donde solo quede una opción válida (p.ej. cierre de nombre cuando único candidato coincide). Hay que tener cuidado de no romper la lógica de `FNAME` (elección de función debe seguir viniendo del LLM cuando hay múltiples candidatos o prefijo ambiguo).
- Modificar `src/generator.py::generate_prompt`: antes de llamar al modelo, preguntar a gramática si hay token forzado. Si sí, obtener pieza vía `grammar.vocab.piece_of(token_id)`, validar, `advance`, añadir token a `ids` (¿añadirlo aunque no venga de logits? Sí, forma parte de la secuencia para próximas llamadas al modelo - el contexto debe incluirlo). Incrementar iteración, continuar. Si no, hacer flujo actual con logits+mask.

**Notas:** 
- El token forzado debe seguir pasando por validación (comprobar que existe en vocab, etc.).
- La secuencia `ids` debe mantenerse coherente (incluye tokens forzados) para que `get_logits_from_input_ids(ids)` reciba el contexto correcto en siguientes pasos.
- No usar heurísticas para elegir función - solo forzar tokens estructurales/obligatorios por gramática JSON.

### 4.2 Caching y batching
**Caching:**
- Cache a nivel de resultado: clave = hash de `functions_definition` (normalizado) + hash de `prompt` normalizado. Almacenar `FunctionCallResult` serializable. Cache persistente opcional (JSON) o en memoria. Debe ser configurable (flag/env) y no interferir con comportamiento por defecto.
- Cache a nivel de sesión/opcional: evitar regenerar mismo prompt.

**Batching:**
- Agrupar test cases con mismas `functions_definition`? O procesar en lotes. Dado que modelo se carga una vez y gramática se crea por prompt, batching real a nivel de inferencia puede ser complejo con API actual (`get_logits_from_input_ids` toma un solo `input_ids` list). Alternativa: batch de procesamiento (agrupar I/O, paralelizar validación) o preparar prompts en batch. También considerar "batching" como procesar múltiples prompts eficientemente. Hay que ser pragmático y compatible con API existente.

**Diseño:**
- Crear módulo `src/cache.py` con `ResultCache` (in-memory + opcional disk). 
- Crear módulo `src/batch.py` o integrar en `main.py`/`generator.py`.
- Añadir flags CLI: `--cache/--no-cache`, `--batch-size` etc., con valores por defecto que mantengan comportamiento actual.

### 4.3 Visualización/cliente CLI
**Requisitos:** cliente desde el cual se pueda visualizar el proceso de generación, ver qué va pensando el modelo en tiempo real mientras responde preguntas, activable/desactivable con flag, crear regla opcional `make cli` para arrancarlo.

**Diseño:**
- Añadir flag `--visualize` / `--cli` / `--stream` (nombre a decidir) a CLI.
- Crear `src/visualizer.py` o `src/cli_viewer.py` con interfaz (rich si disponible? Pero no está en deps; añadirlo requiere justificar y actualizar pyproject/uv.lock. Hay que evaluar: ¿está permitido? Subject no prohíbe rich explícitamente, solo prohíbe torch/transformers/huggingface/outlines/dspy/equivalentes. rich es UI - probablemente aceptable si justificado, pero hay que tener cuidado. Alternativa: usar solo stdlib `curses` o simple print con colores ANSI. Más portable.)
- Modificar `generate_prompt` para aceptar callback/opción de visualización que reciba: paso, token/pieza, estado gramática, logits top-k (opcional), máscara info. Emitir eventos en tiempo real.
- Añadir regla en `Makefile`: `cli:` opcional que ejecute con flag de visualización.
- Asegurar que sin flag no haya overhead.

### 4.4 Comprehensive test suite
**Requisitos:** bonus "comprehensive test suite". 
- Crear directorio `tests/` con tests unitarios e integración.
- Cubrir: vocabulary, grammar (estados, máscaras), generator (fast-path), io_utils, models validation, integración end-to-end con mocks/stubs del modelo.
- Usar pytest (común) - añadir como dev dependency si necesario? pyproject tiene dev group con flake8,mypy; añadir pytest requiere actualizar pyproject.toml y uv.lock. Justificar en plan.
- Añadir target en Makefile opcional `test` o integrar en verificación.
- Tests deben pasar con `uv run pytest` o equivalente.

### 4.5 Tokenizer recoding
**Requisitos:** evitar uso directo de `encode` y `decode` en código principal, usar `get_logits_from_input_ids` y `get_path_to_vocab_file`. También "public implementation of tokenizer encode and optional decode methods", "demonstration of how encoding and decoding integrate with constrained decoding".

**Análisis actual:**
- `src/generator.py:76`: `ids = model.encode(inference_prompt).tolist()[0]` - directo
- `src/main.py:80`: `ids = model.encode(test).tolist()[0]` para obtener `logits_size` - directo
- `llm_sdk` tiene `encode`/`decode` públicos

**Solución propuesta:**
1. **Eliminar uso directo en src/**: reemplazar llamadas a `model.encode()` por nuestro propio mecanismo.
2. **Implementar tokenizer propio en src/**: `src/tokenizer.py` que use `vocab_path = model.get_path_to_vocab_file()` para cargar vocabulario y proporcione `encode(text) -> list[int]` y opcional `decode(ids) -> str`. Debe integrarse con constrained decoding (usar mismo vocab/mapeo).
3. **Para obtener logits_size sin encode**: alternativa - llamar `get_path_to_vocab_file()`, cargar vocab JSON para contar entradas, o llamar `get_logits_from_input_ids` con input dummy y medir longitud. Pero `get_logits_from_input_ids` necesita `input_ids` válidos. También vocab JSON tiene mapeo token->id; longitud puede inferirse del vocab o del propio modelo. Pero hay que ser compatible con `Vocabulary` que recibe `logits_size`. 
   - Opción A: cargar vocab desde `get_path_to_vocab_file()` y usar `len(vocab_json)` o max id+1. Pero vocab puede no tener todos los ids? Vocabulary valida que ids estén en rango 0..logits_size-1.
   - Opción B: hacer llamada con `[0]` (si existe) o construir input mínimo y obtener len(logits).
4. **Integración con constrained decoding**: el tokenizer propio debe producir ids consistentes con el vocab usado por Grammar/Vocabulary.

**Restricciones:** no usar transformers/huggingface en nuestro código. Podemos usar json, re, stdlib. llm_sdk sigue usándose (solo interfaz pública).

## 5. Fases

### Fase 1: Preparación y análisis de puntos de cambio
- **Objetivo:** Identificar todos los puntos exactos a modificar y crear estructura base sin romper nada.
- **Archivos:** `src/generator.py`, `src/main.py`, `src/grammar.py`, `src/vocabulary.py`, `Makefile`, `pyproject.toml` (solo lectura/análisis), `plans/01-...md`
- **Cambios:** 
  1. Documentar puntos exactos (ya hecho en análisis) - confirmar líneas.
  2. Verificar estado git (`git status`, `git diff`) - no ejecutar comandos aquí pero builder debe hacerlo.
  3. Crear archivos vacíos/esqueleto para nuevos módulos si necesario: `src/tokenizer.py`, `src/cache.py`, `src/visualizer.py`, `src/cli_viewer.py`, `tests/__init__.py`, etc. (solo estructura, sin lógica compleja)
- **Verificación:** `uv run flake8 . && uv run mypy . --strict` (debe pasar estado actual)
- **Criterio de aceptación:** Estado actual pasa lint estricto, inventario de cambios completo.
- **Rollback:** `git checkout -- .` si se crearon archivos inesperados (pero solo en plans permitido al planner; builder decidirá)

### Fase 2: Fast-path para tokens estructurales deterministas
- **Objetivo:** Evitar llamadas al modelo cuando solo cabe un token estructural específico.
- **Archivos:** `src/grammar.py`, `src/generator.py`
- **Cambios:**
  1. **Grammar** (`src/grammar.py`):
     - Añadir método `forced_next_token_id(self) -> int | None`: 
       - Si `self._state == _State.DONE`: None
       - Si estado fuerza un solo token vía `mask_for_next()` lógica: OPEN -> `{`, COLON -> `:`, AFTER_NAME -> `,`, OPEN_PARAMS -> `{`, CLOSE -> `}`. También considerar casos donde solo hay una opción (FNAME con único candidato igual a acc y se puede cerrar con `"`, o END_VALUE con única opción). Ser conservador: solo forzar cuando inequívocamente único y estructural/obligatorio.
     - Añadir método `forced_next_piece(self) -> str | None` que use `forced_next_token_id()` y `self.vocab.piece_of()`.
     - Añadir validación: asegurar que token forzado existe en vocab.
     - Mantener comportamiento existente intacto.
  2. **Generator** (`src/generator.py`):
     - En `generate_prompt()`, al inicio del bucle: si `grammar.forced_next_piece()` existe (y vocab/piece válido), obtener token_id, validar, llamar `grammar.advance(piece)`, añadir a `ids`, `iteration += 1`, `continue` sin llamar a modelo.
     - Asegurar que `ids` se actualice correctamente.
     - Mantener todas las validaciones existentes.
- **Verificación:**
  - `uv run flake8 . && uv run mypy . --strict`
  - Ejecutar con datos de ejemplo: `uv run python -m src` (comprobar que salida sigue siendo válida, misma estructura)
  - Comparar conteo aproximado de llamadas a modelo (opcional si hay forma de instrumentar) - al menos funcionalmente idéntico
- **Criterio de aceptación:** 
  - Tokens estructurales deterministas no pasan por `get_logits_from_input_ids` (verificable por inspección/callback)
  - Salida JSON idéntica en estructura a antes
  - Lint estricto pasa
- **Rollback:** Revertir cambios en grammar/generator

### Fase 3: Tokenizer recoding (evitar encode/decode directo en main code)
- **Objetivo:** Eliminar uso directo de `model.encode()`/`model.decode()` en `src/`. Usar `get_logits_from_input_ids` y `get_path_to_vocab_file`. Implementar tokenizer propio.
- **Archivos:** `src/tokenizer.py` (nuevo), `src/main.py`, `src/generator.py`, `src/vocabulary.py` (posibles ajustes), `src/models.py` (actualizar `ModelProtocol` si necesario)
- **Cambios:**
  1. **Crear `src/tokenizer.py`:**
     - Clase `SimpleTokenizer` que toma vocab path (desde `get_path_to_vocab_file()`) y opcionalmente logits_size.
     - Cargar vocab JSON, construir mapeo piece<->id (consistente con `Vocabulary`).
     - Implementar `encode(self, text: str) -> list[int]` - implementación propia basada en vocab (BPE-like simple o greedy longest-prefix matching usando vocab). Debe ser determinista.
     - Implementar `decode(self, ids: list[int]) -> str` opcional (bonus menciona "public implementation of tokenizer encode and optional decode methods").
     - Método para obtener vocab path vía modelo: `from_model(model)`.
     - Integrar con constrained decoding (usar mismo normalización `Ġ`->espacio si aplica).
  2. **Actualizar `src/main.py`:**
     - Eliminar `ids = model.encode(test).tolist()[0]` para obtener `logits_size` (`80`). Reemplazar por: obtener `vocab_path = model.get_path_to_vocab_file()`, cargar vocab para determinar tamaño (max id + 1 o len) o usar `len(model.get_logits_from_input_ids([0]))` si token 0 existe. Necesitar validar con API - `get_logits_from_input_ids` acepta lista de ids. Hay que encontrar forma compatible.
     - Usar `SimpleTokenizer.from_model(model).encode(...)` donde sea necesario.
  3. **Actualizar `src/generator.py`:**
     - Eliminar `ids = model.encode(inference_prompt).tolist()[0]` (`76`). Usar tokenizer propio.
  4. **Actualizar `src/models.py`:** `ModelProtocol` puede mantenerse (encode aún existe en llm_sdk pero no lo usaremos desde src). No usarlo.
- **Verificación:**
  - Grep: `grep -rn "\.encode\|\.decode" src/` debe mostrar 0 usos directos (excepto quizás en comentarios)
  - `uv run flake8 . && uv run mypy . --strict`
  - Ejecutar `uv run python -m src` correctamente
- **Criterio de aceptación:** No hay `model.encode()` ni `model.decode()` llamados directamente desde código en `src/`. Uso de `get_logits_from_input_ids` y `get_path_to_vocab_file` demostrado. Tokenizer propio funcional.
- **Rollback:** Restaurar usos de encode

### Fase 4: Caching y batching
- **Objetivo:** Añadir caching de resultados y batching.
- **Archivos:** `src/cache.py` (nuevo), `src/batch.py` (nuevo), `src/main.py`, `src/generator.py` (opcionales), `pyproject.toml` (si se añaden deps - evaluar)
- **Cambios:**
  1. **Cache** (`src/cache.py`):
     - `ResultCache` con clave hash (sha256) de functions def + prompt
     - Métodos `get`, `set`, `clear`
     - Persistencia opcional a JSON
  2. **Batching** (`src/batch.py`):
     - Lógica para agrupar test cases, procesar eficientemente
     - Compatible con API secuencial actual
  3. **Integración** (`src/main.py`):
     - Añadir flags `--cache/--no-cache`, `--batch-size` (con defaults compatibles)
     - Usar cache antes de generar
     - Guardar en cache tras generar
- **Verificación:** `uv run flake8 . && uv run mypy . --strict`, ejecutar con cache habilitado/deshabilitado
- **Criterio de aceptación:** Caching funciona, batching implementado, flags operativos
- **Rollback:** Deshabilitar features

### Fase 5: Visualización/cliente CLI con flag y `make cli`
- **Objetivo:** Cliente visualizable en tiempo real, activable/desactivable, `make cli` opcional.
- **Archivos:** `src/visualizer.py` (nuevo), `src/cli_viewer.py` (nuevo), `src/generator.py`, `src/main.py`, `Makefile`
- **Cambios:**
  1. **Visualizer** (`src/visualizer.py`): callback/event system para pasos de generación (token, estado, top-k opcional, forced vs model)
  2. **CLI viewer** (`src/cli_viewer.py`): interfaz en tiempo real (usar rich si se añade? O curses/stdlib). Intentar stdlib primero para evitar deps; si necesario añadir rich justificado.
  3. **Integración**: flag `--visualize`/`--cli-view` en CLI, pasar visualizer a `generate_prompt`
  4. **Makefile**: añadir `cli:` regla opcional (PHONY) que ejecute con visualización
- **Verificación:** `make cli` funciona si flag pasado correctamente, sin flag sin overhead
- **Criterio de aceptación:** Visualización muestra proceso en tiempo real, flag toggleable, `make cli` existe
- **Rollback:** Quitar integración

### Fase 6: Comprehensive test suite
- **Objetivo:** Añadir suite completa.
- **Archivos:** `tests/__init__.py`, `tests/test_vocabulary.py`, `tests/test_grammar.py`, `tests/test_generator.py`, `tests/test_tokenizer.py`, `tests/test_io_utils.py`, `tests/test_integration.py`, `pyproject.toml`, `Makefile` (opcional)
- **Cambios:**
  1. Crear tests cubriendo casos principales y edge cases
  2. Añadir pytest a dev dependencies en `pyproject.toml` + actualizar `uv.lock` (builder lo hará)
  3. Añadir target opcional
- **Verificación:** `uv run pytest -v` pasa
- **Criterio de aceptación:** Suite comprensiva implementada y pasando
- **Rollback:** Eliminar tests/

### Fase 7: Integración, verificación global y documentación mínima
- **Objetivo:** Asegurar todo integrado, lint estricto pasa, funcionalidad completa.
- **Archivos:** Todos modificados, `README.md` opcional (solo si necesario, pero no obligatorio tocar)
- **Cambios:** Ajustes finales, verificar flags, demos
- **Verificación:** `flake8 . && mypy . --strict`, `uv run pytest` (si existe), `uv run python -m src`, validar salida
- **Criterio de aceptación:** Todos los bonus funcionando, sin regresiones
- **Rollback:** Según fase problemática

## 6. Criterios de aceptación globales
- [ ] Fast-path: tokens estructurales deterministas no pasan por modelo
- [ ] Caching implementado y funcional (configurable)
- [ ] Batching implementado
- [ ] Cliente visualizable activable/desactivable con flag, `make cli` opcional existe
- [ ] Comprehensive test suite añadida y pasando
- [ ] Sin uso directo de `encode`/`decode` en `src/`, usando `get_logits_from_input_ids` y `get_path_to_vocab_file`
- [ ] Tokenizer propio implementado con encode (y decode opcional) integrado con constrained decoding
- [ ] `flake8 .` pasa
- [ ] `mypy . --strict` pasa
- [ ] `uv run python -m src` produce salida válida idéntica en formato
- [ ] Sin dependencias prohibidas añadidas, solo interfaz pública de `llm_sdk`
- [ ] Sin heurísticas para selección de función
- [ ] Pydantic usado donde corresponde, type hints completos

## 7. Comandos de verificación
```bash
# Estado
git status && git diff --stat

# Lint
uv run flake8 .
uv run mypy . --strict

# Tests (tras fase 6)
uv run pytest -v

# Ejecución normal
uv run python -m src

# Con flags nuevos (ejemplos)
uv run python -m src --visualize  # o flag equivalente
uv run python -m src --cache
make cli  # si implementado
```

Validación de salida: verificar JSON válido, un objeto por prompt, keys `prompt,name,parameters`, nombres válidos, tipos correctos según `functions_definition.json`.

## 8. Riesgos y mitigaciones
| Riesgo | Impacto | Probabilidad | Mitigación |
|---|---|---|---|
| Fast-path fuerza token incorrecto | Alto | Media | Solo forzar cuando máscara singleton inequívoco; validar con vocab; tests |
| Cambiar inicialización de logits_size rompe Vocabulary | Alto | Media | Probar múltiples métodos (cargar vocab, dummy call a get_logits_from_input_ids) |
| Tokenizer propio no coincide con tokenizer del modelo | Alto | Media | Usar mismo vocab file de `get_path_to_vocab_file()`, misma normalización, tests de encoding |
| Añadir deps (rich/pytest) rompe uv sync | Medio | Media | Justificar, actualizar pyproject+uv.lock coherentemente, intentar stdlib primero |
| Visualización afecta rendimiento | Medio | Media | Solo activo con flag, overhead mínimo, desactivado por defecto |
| Batching cambia comportamiento | Medio | Media | Mantener secuencial por defecto, flags opcionales |
| Romper API llm_sdk (uso privado) | Alto | Baja | Solo usar métodos públicos documentados |

## 9. Plan de rollback global
- `git log --oneline -5` para identificar commit
- `git diff` revisar cambios
- `git checkout -- src/ Makefile pyproject.toml tests/` según lo modificado
- Verificar `uv run flake8 . && uv run mypy . --strict` y ejecución
- Si hay archivos nuevos no commiteados, eliminarlos con precaución

## 10. Fuera de alcance
- Modificar `llm_sdk/` (usar solo interfaz pública)
- Añadir soporte para modelos distintos a Qwen/Qwen3-0.6B más allá de lo que permita arquitectura (bonus menciona soporte múltiple pero hay que ser compatible)
- Ejecutar tests/benchmarks (corresponde a builder)
- Tocar planes existentes
- Cambiar formato de salida obligatorio
- Usar heurísticas para selección de función

## Resumen
- **Ruta del plan:** `plans/01-optimizations-and-bonuses.md`
- **Fases:** 7 fases (preparación, fast-path estructural, tokenizer recoding, caching/batching, visualización+make cli, tests comprehensivos, integración)
- **Riesgos principales:** tokenización inconsistente, fast-path incorrecto, adición de dependencias
- **Decisión necesaria:** Confirmar enfoque para obtener `logits_size` sin `encode()` directo (cargar vocab desde `get_path_to_vocab_file()` vs dummy call a `get_logits_from_input_ids`). También confirmar si añadir `rich`/`pytest` está aceptable justificadamente.