from __future__ import annotations

from core.agent_v3.tools._config_builder import _materialize_depth_bins


def test_materialize_depth_bins_partitions_box_target() -> None:
    primary = {
        "name": "WaterPhantom",
        "shape": "box",
        "material": "G4_WATER",
        "dimensions": {"size_x_mm": 100.0, "size_y_mm": 80.0, "size_z_mm": 200.0},
    }
    volumes = [{"name": "WaterPhantom", "shape": "box", "material": "G4_WATER"}]

    names = _materialize_depth_bins(volumes, primary, count=10)

    assert len(names) == 10
    assert len(volumes) == 11
    assert volumes[1]["parent"] == "WaterPhantom"
    assert volumes[1]["role"] == "depth_bin"
    assert volumes[1]["position_mm"][2] == -90.0
    assert volumes[-1]["position_mm"][2] == 90.0
    assert all(volume["size_mm"] == [100.0, 80.0, 20.0] for volume in volumes[1:])


def test_materialize_depth_bins_does_not_fake_non_box_partition() -> None:
    primary = {
        "name": "Sphere",
        "shape": "sphere",
        "material": "G4_WATER",
        "dimensions": {"radius_mm": 50.0},
    }
    volumes = [{"name": "Sphere", "shape": "sphere", "material": "G4_WATER"}]

    assert _materialize_depth_bins(volumes, primary, count=10) == []
    assert len(volumes) == 1
