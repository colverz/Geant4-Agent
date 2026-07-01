# Architecture Docs

This directory keeps the current architecture docs small and navigable.

## Current Mainline

- `ARCHITECTURE.md`: current v3-first architecture overview.
- `GEANT4_AGENT_REBUILD_PROGRESS_2026-05-21.md`: detailed implementation log
  for the v3 rebuild.
- `GEANT4_AGENT_V3_INTELLIGENCE_DESIGN_2026-05-25.md`: next-step design for a
  smarter agent that avoids keyword/dictionary routing.
- `GEANT4_AGENT_V3_NEXT_IMPLEMENTATION_PLAN_2026-05-25.md`: phased execution
  plan based on architecture review, eval harness, and MCP tool design skills.
- `GEANT4_AGENT_V3_EVAL_AND_UPGRADE_PLAYBOOK_2026-06-04.md`: current operating
  manual for v3 architecture upgrades, eval gates, phased migration, and merge
  criteria.
- `V3_PHASE_REVIEW_2026-06-06.md`: latest phase review confirming the v3-first
  route and next upgrade priorities.
- `V3_STAGE_CONCLUSION_2026-06-11.md`: stage conclusion for completed v3
  functionality, verification, remaining risks, and next mainline.
- `V3_UPDATE_LOG.md`: plain-language running log for each v3 mainline update
  round.
- `reuse_archive/`: v2 asset inventory and legacy candidate list.

## Supporting References

- `GEANT4_SIMULATION_BRIDGE_2026-03-30.md`: runtime bridge and Geant4 adapter
  behavior.
- `GEANT4_MCP_DESIGN_2026-03-23.md`: original MCP boundary design.
- `PROJECT_LAYOUT_2026-03-25.md`: older layout map, still useful for migration
  context.
- `NLP_BERT_MAINLINE_BOUNDARY_2026-05-15.md`: BERT/NLU boundary notes.

## Historical Or Transitional

These are retained as context, not as the source of current product behavior:

- `CHROMIUM_UI_MIGRATION_2026-03-23.md`
- `geometry_upgrade_roadmap_2026-03-06.md`
- `MAINTENANCE_*_2026-05-05.md`
- `NLU_AGENT_*_2026-05-05.md`
- `REDUNDANCY_AND_LAYOUT_REFACTOR_PLAN_2026-05-05.md`
- Remaining HTML exports, if any, should be treated as temporary snapshots and
  converted to Markdown or deleted once their content is captured in current
  docs.

## Rule

New architecture decisions should update `ARCHITECTURE.md` or add a dated
Markdown note. Avoid adding new HTML exports as primary design documents.
