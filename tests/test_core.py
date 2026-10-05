import numpy as np

from object_modeling.core import (
    _multi_view_filter,
    _oriented_hull_faces,
    _remove_board_and_background,
    _statistical_filter,
    _voxel_downsample,
)
from scipy.spatial import ConvexHull


def test_voxel_downsample_reduces_duplicate_points():
    points = np.array([[0.0, 0.0, 0.0], [0.0005, 0.0, 0.0], [0.01, 0.0, 0.0]])
    colors = np.ones((3, 3)) * 100
    reduced, reduced_colors = _voxel_downsample(points, colors, 0.002)
    assert len(reduced) == 2
    assert reduced_colors.shape == (2, 3)


def test_statistical_filter_removes_isolated_outlier():
    rng = np.random.default_rng(4)
    cluster = rng.normal(0.0, 0.001, size=(80, 3))
    points = np.vstack([cluster, [1.0, 1.0, 1.0]])
    colors = np.zeros_like(points)
    filtered, _ = _statistical_filter(points, colors, k=8, std_ratio=2.5)
    assert len(filtered) < len(points)
    assert np.max(np.linalg.norm(filtered, axis=1)) < 0.1


def test_multi_view_filter_removes_single_frame_voxel():
    points = np.array([[0.0, 0.0, 0.1], [0.0005, 0.0, 0.1], [0.01, 0.0, 0.1]])
    colors = np.ones((3, 3)) * 100
    frame_ids = np.array([0, 1, 2])
    kept, kept_colors = _multi_view_filter(points, colors, frame_ids, 0.002, min_views=2)
    assert len(kept) == 2
    assert kept_colors.shape == (2, 3)


def test_board_cleanup_removes_plane_outside_object_footprint():
    xs = np.arange(-0.018, 0.019, 0.003)
    ys = np.arange(-0.018, 0.019, 0.003)
    zs = np.arange(0.025, 0.061, 0.003)
    object_points = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    plane_points = np.array([[x, y, 0.012] for x in np.arange(-0.08, 0.081, 0.004)
                             for y in np.arange(-0.06, 0.061, 0.004)])
    points = np.vstack([object_points, plane_points])
    colors = np.ones((len(points), 3)) * 100
    kept, _, stats = _remove_board_and_background(
        points, colors, {"board_plane_clearance_m": 0.012,
                         "component_seed_z_m": 0.020,
                         "component_eps_m": 0.004,
                         "base_xy_margin_m": 0.004})
    assert stats["removed"] > 0
    assert kept[:, 0].min() > -0.03 and kept[:, 0].max() < 0.03


def test_convex_hull_faces_have_outward_consistent_winding():
    points = np.array([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ])
    hull = ConvexHull(points)
    faces = _oriented_hull_faces(points, hull)
    interior = points[hull.vertices].mean(axis=0)
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    centers = points[faces].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", normals, centers - interior) > 0.0)
