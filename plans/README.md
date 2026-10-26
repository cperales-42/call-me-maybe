# Planes de implementación

Este directorio lo escribe el agente `planner` y lo lee el agente `builder`.
Cada plan debe ser autocontenido, estar en Markdown y dividirse en fases
pequeñas y verificables.

## Convención de nombres

```text
plans/NN-slug-corto.md
```

Ejemplo: `plans/01-reparar-fast-path.md`. `NN` es un número de dos dígitos y no
se reutiliza.

## Plantilla

Copia esta estructura y rellénala con evidencia real del repositorio:

```markdown
# Plan NN: Título

## 1. Objetivo
Resultado medible que se quiere conseguir.

## 2. Contexto y estado actual
Qué hace hoy el código, con rutas y líneas. Causa raíz del problema.

## 3. Restricciones
Requisitos del subject, prohibiciones y cosas que no se deben tocar.

## 4. Diseño propuesto
Solución técnica: estructuras, firmas, estados y flujo.

## 5. Fases

### Fase 1: Nombre
- **Objetivo:** ...
- **Archivos:** ...
- **Cambios:** pasos concretos, en orden.
- **Verificación:** comandos exactos y resultado esperado.
- **Criterio de aceptación:** condiciones medibles.
- **Rollback:** cómo revertir solo esta fase.

### Fase 2: Nombre
...

## 6. Criterios de aceptación globales
Checklist final verificable.

## 7. Comandos de verificación
Lint, tests, ejecución y validación de salida.

## 8. Riesgos y mitigaciones
Riesgos técnicos y de regresión, con detección y mitigación.

## 9. Plan de rollback global
Cómo volver al estado anterior si el plan falla.

## 10. Fuera de alcance
Lo que este plan no hará.
```

## Gate de calidad

Toda fase que toque código termina con:

```bash
uv run flake8 .
uv run mypy . --strict
```

El `builder` no puede dar una fase por terminada si alguno falla. Además debe
ejecutar los tests y la validación funcional indicados en el plan.
