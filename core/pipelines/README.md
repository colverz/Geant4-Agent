# Pipeline Selection

The project intentionally keeps legacy and v2 pipelines side by side while v2 is hardened.

## Defaults

- Core `select_pipelines()` compatibility default: legacy
- Web strict/main UI default: v2
- Industrial benchmark runners: v2

The product-facing main chain now defaults to the LLM + v2 geometry/source
route. The lower-level selector keeps its legacy default for tests and direct
compatibility callers that do not pass explicit pipeline arguments.

## Runtime Switches

`select_pipelines()` accepts explicit arguments and also reads environment variables:

- `GEOMETRY_PIPELINE=legacy|v2`
- `SOURCE_PIPELINE=legacy|v2`

Explicit function arguments win over environment variables. Unknown values fall back to the selector compatibility default.

## Expected Use

- Use `v2` for normal geometry/source development, runtime benchmark work, and UI testing.
- Use `legacy` for compatibility checks and fallback behavior.
- Do not delete legacy until v2 has equivalent workflow coverage and a documented removal window.
