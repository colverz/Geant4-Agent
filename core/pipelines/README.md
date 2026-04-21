# Pipeline Selection

The project intentionally keeps legacy and v2 pipelines side by side while v2 is hardened.

## Defaults

- `geometry`: legacy
- `source`: legacy

This keeps existing tests and UI flows stable unless a caller opts into v2.

## Runtime Switches

`select_pipelines()` accepts explicit arguments and also reads environment variables:

- `GEOMETRY_PIPELINE=legacy|v2`
- `SOURCE_PIPELINE=legacy|v2`

Explicit function arguments win over environment variables. Unknown values fall back to legacy.

## Expected Use

- Use `legacy` for compatibility checks and fallback behavior.
- Use `v2` for geometry/source development and regression testing.
- Do not delete legacy until v2 has equivalent workflow coverage and a documented removal window.
