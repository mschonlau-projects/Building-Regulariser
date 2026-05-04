import math
import warnings
from typing import Any

import geopandas as gpd
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Polygon

from buildingregulariser import regularize_geodataframe
from buildingregulariser.regularization import (
    flatten_to_polygons,
    regularize_single_polygon,
)

DEFAULT_KWARGS: dict[str, Any] = dict(
    parallel_threshold=1.0,
    allow_45_degree=False,
    diagonal_threshold_reduction=15,
    allow_circles=False,
    circle_threshold=0.9,
    simplify=True,
    simplify_tolerance=0.5,
)


def two_lobe_polygon() -> Polygon:
    """Self-intersecting polygon that buffer(0) splits into two equal squares."""
    return Polygon(
        [
            (0, 0),
            (4, 0),
            (4, 2),
            (0, 2),
            (0, 0),
            (0, 4),
            (4, 4),
            (4, 6),
            (0, 6),
            (0, 4),
            (0, 0),
        ]
    )


def figure_eight_polygon() -> Polygon:
    """Two squares meeting at a single vertex — buffer(0) returns MultiPolygon."""
    return Polygon(
        [
            (0, 0),
            (2, 0),
            (2, 2),
            (0, 2),
            (0, 0),
            (-2, 0),
            (-2, -2),
            (0, -2),
            (0, 0),
        ]
    )


def square() -> Polygon:
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_split_returns_one_dict_per_lobe():
    """A self-intersecting input that splits should produce one result dict
    per output Polygon — not a single dict with a MultiPolygon geometry."""
    results = regularize_single_polygon(polygon=two_lobe_polygon(), **DEFAULT_KWARGS)

    assert isinstance(results, list)
    assert len(results) >= 2
    for r in results:
        assert isinstance(r["geometry"], Polygon)
        assert r["geometry"].is_valid
        assert not r["geometry"].is_empty
        assert "iou" in r
        assert "main_direction" in r


def test_figure_eight_does_not_crash():
    """Vertex-touching lobes should not raise; each part should be a valid Polygon."""
    results = regularize_single_polygon(
        polygon=figure_eight_polygon(), **DEFAULT_KWARGS
    )
    assert isinstance(results, list)
    assert len(results) >= 2
    for r in results:
        assert isinstance(r["geometry"], Polygon)
        assert r["geometry"].is_valid


def test_split_iou_per_piece_meaningful():
    """Each piece's IoU should be a sensible number in (0, 1]."""
    results = regularize_single_polygon(polygon=two_lobe_polygon(), **DEFAULT_KWARGS)
    for r in results:
        assert 0 < r["iou"] <= 1


def test_split_preserves_both_lobes_area():
    """Two equal-sized real lobes — the union of all output Polygons should
    cover most of the cleaned input area."""
    poly = two_lobe_polygon()
    results = regularize_single_polygon(polygon=poly, **DEFAULT_KWARGS)
    total_output_area = sum(r["geometry"].area for r in results)
    cleaned_input_area = poly.buffer(0).area
    assert total_output_area > 0.7 * cleaned_input_area


def test_split_preserves_per_piece_main_direction():
    """Each piece must keep its own main_direction — not collapsed to a single
    value taken from the largest lobe (which was the previous lossy behavior)."""
    results = regularize_single_polygon(polygon=two_lobe_polygon(), **DEFAULT_KWARGS)
    assert all("main_direction" in r for r in results)
    assert all(r["main_direction"] is not None for r in results)


def test_single_polygon_returns_one_element_list():
    """The common-case (non-splitting) input must return a length-1 list with
    a Polygon geometry."""
    results = regularize_single_polygon(polygon=square(), **DEFAULT_KWARGS)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0]["geometry"], Polygon)


def test_geodataframe_through_split_path_produces_polygons():
    """End-to-end: a self-intersecting polygon should produce multiple output
    rows of Polygons (no MultiPolygons leaking through)."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[two_lobe_polygon()], crs="EPSG:32750")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = regularize_geodataframe(gdf, num_cores=1)

    assert len(out) >= 2
    assert all(g.geom_type == "Polygon" for g in out.geometry)
    assert out.geometry.is_valid.all()
    assert out.geometry.notnull().all()


def test_geodataframe_user_attributes_repeated_for_split_rows():
    """When a split creates multiple output rows from one input, user-provided
    attribute columns must be duplicated across the new rows."""
    gdf = gpd.GeoDataFrame(
        {"id": [42], "name": ["building_x"]},
        geometry=[two_lobe_polygon()],
        crs="EPSG:32750",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = regularize_geodataframe(gdf, num_cores=1)

    assert len(out) >= 2
    assert (out["id"] == 42).all()
    assert (out["name"] == "building_x").all()


def test_geodataframe_per_piece_metadata_preserved():
    """include_metadata=True must surface per-piece iou/main_direction columns,
    not collapsed values from a single representative lobe."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[two_lobe_polygon()], crs="EPSG:32750")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = regularize_geodataframe(gdf, num_cores=1, include_metadata=True)

    assert "iou" in out.columns
    assert "main_direction" in out.columns
    assert len(out) >= 2


def test_flatten_to_polygons_unwraps_multipolygons():
    """flatten_to_polygons should unwrap MultiPolygons into their Polygons."""
    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    b = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    c = Polygon([(4, 0), (5, 0), (5, 1), (4, 1)])
    nested = MultiPolygon([b, c])

    flat = flatten_to_polygons([a, nested])

    assert len(flat) == 3
    assert all(isinstance(p, Polygon) for p in flat)


def test_flatten_to_polygons_drops_empty_and_none():
    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    empty = Polygon()
    flat = flatten_to_polygons([a, empty, None])
    assert flat == [a]


def test_multipolygon_input_handled_directly():
    """A MultiPolygon passed directly (e.g. surviving make_valid + explode of
    a GeometryCollection upstream) should be handled by recursing on each part,
    not warned-about and dropped."""
    a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    b = Polygon([(10, 0), (15, 0), (15, 5), (10, 5)])
    multi = MultiPolygon([a, b])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into a failure
        results = regularize_single_polygon(polygon=multi, **DEFAULT_KWARGS)

    assert len(results) == 2
    assert all(isinstance(r["geometry"], Polygon) for r in results)
    assert all(r["geometry"].is_valid for r in results)


def test_multiple_splitting_inputs_repeat_attributes_correctly():
    """When several input rows each split, the coordinator must repeat each
    input's attributes the correct number of times — no cross-contamination
    between rows."""
    poly_a = two_lobe_polygon()
    poly_b = translate(two_lobe_polygon(), xoff=100)
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "name": ["A", "B"]},
        geometry=[poly_a, poly_b],
        crs="EPSG:32750",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = regularize_geodataframe(gdf, num_cores=1)

    assert len(out) >= 4
    # Every row labeled "A" must have id=1; every row labeled "B" must have id=2
    a_rows = out[out["name"] == "A"]
    b_rows = out[out["name"] == "B"]
    assert len(a_rows) >= 2
    assert len(b_rows) >= 2
    assert (a_rows["id"] == 1).all()
    assert (b_rows["id"] == 2).all()
    # The B-rows' geometries must be in the translated location, not mixed
    # with A-rows' geometries
    assert all(g.centroid.x > 50 for g in b_rows.geometry)
    assert all(g.centroid.x < 50 for g in a_rows.geometry)


def test_parallel_processing_handles_list_return():
    """Pool.map must round-trip the new List[dict] return type — pickling and
    list-of-lists assembly must work the same as the sequential path."""
    poly_a = two_lobe_polygon()
    poly_b = translate(two_lobe_polygon(), xoff=100)
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[poly_a, poly_b], crs="EPSG:32750")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequential = regularize_geodataframe(gdf, num_cores=1)
        parallel = regularize_geodataframe(gdf, num_cores=2)

    assert len(sequential) == len(parallel)
    assert sorted(g.area for g in sequential.geometry) == sorted(
        g.area for g in parallel.geometry
    )


def test_per_piece_main_direction_independent():
    """When two lobes have visibly different orientations, each piece should
    regularize to its own main_direction — not collapse to a single shared
    value (regression guard for the previous 'use largest lobe's direction'
    behavior)."""
    axis_aligned = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    rotated = rotate(
        Polygon([(50, 0), (60, 0), (60, 4), (50, 4)]), angle=30, origin="center"
    )
    multi = MultiPolygon([axis_aligned, rotated])

    results = regularize_single_polygon(polygon=multi, **DEFAULT_KWARGS)
    assert len(results) == 2

    directions = sorted(r["main_direction"] for r in results)
    # The two pieces should not have collapsed to identical directions —
    # they were oriented ~30° apart in input.
    assert not math.isclose(directions[0], directions[1], abs_tol=5.0), (
        f"Expected distinct main_directions, got {directions}"
    )
