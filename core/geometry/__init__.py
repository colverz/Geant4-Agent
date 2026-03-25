from core.geometry.catalog import (
    GeometryCatalogEntry,
    GeometryParamDefinition,
    get_geometry_catalog_entry,
    iter_geometry_catalog,
    resolve_geometry_structure,
)
from core.geometry.adapters import (
    diff_geometry_config_fragment,
    geometry_spec_to_config_fragment,
    geometry_spec_to_runtime_geometry,
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
from core.geometry.spec import GeometryEvidence, GeometryIntent, GeometrySpec

__all__ = [
    "GeometryCatalogEntry",
    "GeometryCompileResult",
    "GeometryEvidence",
    "GeometryIntent",
    "GeometryParamDefinition",
    "GeometrySpec",
    "build_geometry_intent_from_config",
    "build_geometry_intent_from_semantic_frame",
    "compile_geometry_intent",
    "compile_geometry_spec_from_config",
    "compile_geometry_spec_from_semantic_frame",
    "compile_geometry_spec_from_slot_frame",
    "diff_geometry_config_fragment",
    "geometry_spec_to_config_fragment",
    "geometry_spec_to_runtime_geometry",
    "get_geometry_catalog_entry",
    "iter_geometry_catalog",
    "resolve_geometry_structure",
]
