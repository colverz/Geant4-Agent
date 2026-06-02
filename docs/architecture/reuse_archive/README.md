# v2 Reuse Archive

This directory records v2 assets that should be reused, adapted, frozen, or moved to legacy during the v3 rebuild.

It is an index archive only. No source code is moved in this step.

## Files

- `V2_REUSABLE_ASSETS_2026-05-21.md`: modules that should be reused directly or through adapters.
- `V2_LEGACY_CANDIDATES_2026-05-21.md`: modules that should be frozen, archived, or moved to `legacy/` after v3 replacement passes regression checks.

## Migration rule

Move code only after:

1. v3 has an equivalent entry point.
2. import shims are prepared when public imports may break.
3. smoke tests and casebank regression pass.
4. runtime behavior is either validated by Geant4 observation or explicitly marked `not_evaluable`.
