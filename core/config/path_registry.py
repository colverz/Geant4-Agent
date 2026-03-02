from __future__ import annotations


_CANONICAL_ALIASES = {
    "physics_list": "physics.physics_list",
    "source.energy_MeV": "source.energy",
    "physics_list.name": "physics.physics_list",
}


def canonical_field_path(path: str) -> str:
    return _CANONICAL_ALIASES.get(path, path)


def field_matches_pattern(path: str, pattern: str) -> bool:
    canonical_path = canonical_field_path(path)
    if pattern.endswith("."):
        base = canonical_field_path(pattern[:-1])
        return canonical_path == base or canonical_path.startswith(base + ".")
    return canonical_path == canonical_field_path(pattern)
