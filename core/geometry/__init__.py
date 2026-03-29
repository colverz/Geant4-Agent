from core.geometry.catalog import (
    GeometryCatalogEntry,
    GeometryParamDefinition,
    get_geometry_catalog_entry,
    iter_geometry_catalog,
    resolve_geometry_structure,
)
from core.geometry.adapters import (
    compare_slot_frame_geometry,
    diff_geometry_config_fragment,
    geometry_spec_to_config_fragment,
    geometry_spec_to_runtime_geometry,
    legacy_geometry_from_candidate,
)
from core.geometry.compiler import (
    GeometryCompileResult,
    build_geometry_intent_from_config,
    build_geometry_intent_from_semantic_frame,
    compile_geometry_intent,
    compile_geometry_spec_from_config,
    compile_geometry_spec_from_semantic_frame,
    compile_geometry_spec_from_slot_frame,
)
from core.geometry.resolver import (
    GeometryResolutionDraft,
    build_geometry_intent_from_resolved_draft,
    resolve_geometry_from_merged,
)
from core.geometry.spec import GeometryEvidence, GeometryIntent, GeometrySpec

__all__ = [
    "GeometryCatalogEntry",
    "GeometryCompileResult",
    "GeometryEvidence",
    "GeometryIntent",
    "GeometryResolutionDraft",
    "GeometryParamDefinition",
    "GeometrySpec",
    "build_geometry_intent_from_config",
    "build_geometry_intent_from_resolved_draft",
    "build_geometry_intent_from_semantic_frame",
    "compile_geometry_intent",
    "compile_geometry_spec_from_config",
    "compile_geometry_spec_from_semantic_frame",
    "compile_geometry_spec_from_slot_frame",
    "compare_slot_frame_geometry",
    "diff_geometry_config_fragment",
    "geometry_spec_to_config_fragment",
    "geometry_spec_to_runtime_geometry",
    "get_geometry_catalog_entry",
    "iter_geometry_catalog",
    "legacy_geometry_from_candidate",
    "resolve_geometry_from_merged",
    "resolve_geometry_structure",
]
