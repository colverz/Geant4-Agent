import json

from geometry.dsl import parse_graph_json
from geometry.feasibility import check_feasibility


def test_parse_and_feasibility_ok():
    text = json.dumps(
        {
            "nodes": [
                {"id": "parent", "type": "Box", "x": 20, "y": 20, "z": 20},
                {"id": "child", "type": "Tubs", "rmax": 4, "hz": 5},
                {"id": "nest", "type": "Nest", "parent": "parent", "child": "child", "clearance": 1},
            ],
            "root": "nest",
        }
    )
    graph = parse_graph_json(text)
    report = check_feasibility(graph)
    assert report.ok


def test_grid_infeasible():
    text = json.dumps(
        {
            "nodes": [
                {"id": "mod", "type": "Box", "x": 10, "y": 10, "z": 2},
                {"id": "grid", "type": "GridXY", "module": "mod", "nx": 2, "ny": 2, "pitch_x": 5, "pitch_y": 5, "clearance": 0},
            ],
            "root": "grid",
        }
    )
    graph = parse_graph_json(text)
    report = check_feasibility(graph)
    assert not report.ok
