"""Offline RGB-D fusion and lightweight mesh export.

The public functions use only paths supplied by the caller and store metadata
relative to the model package, so the result can be copied to another machine.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import ConvexHull, cKDTree, QhullError
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BuildResult:
    """Paths and quality numbers produced by :func:`build_model`."""

    model_dir: Path
    point_count: int
    mesh_vertices: int
    mesh_faces: int
    extents_m: tuple[float, float, float]


def _load_pose_csv(path: Path):
    poses = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            q = np.array([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = Rotation.from_quat(q).as_matrix()
            transform[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            poses[row["frame"]] = (transform, int(row["charuco_corners"]))
    return poses


def _load_camera_info(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        np.asarray(data["k"], dtype=float).reshape(3, 3),
        float(data.get("depth_scale", 0.001)),
    )


def _fuse(dataset: Path, cfg: dict, min_corners: int, pixel_stride: int, voxel=None):
    K, depth_scale = _load_camera_info(dataset / "camera_info.yaml")
    model_cfg = cfg["model"]
    z_min = float(model_cfg.get("board_z_min_m", 0.008))
    z_max = float(model_cfg.get("board_z_max_m", 0.60))
    margin = float(model_cfg.get("board_xy_margin_m", 0.025))
    board_x = float(cfg["charuco"]["squares_x"]) * float(cfg["charuco"]["square_length_m"])
    board_y = float(cfg["charuco"]["squares_y"]) * float(cfg["charuco"]["square_length_m"])
    poses = _load_pose_csv(dataset / "poses.csv")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points, colors, frame_ids = [], [], []
    for frame_index, (frame, (T_board_camera, corners)) in enumerate(poses.items()):
        if corners < min_corners:
            continue
        color = cv2.imread(str(dataset / "color" / f"{frame}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(dataset / "depth" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or depth.ndim != 2 or color.shape[:2] != depth.shape[:2]:
            continue
        h, w = depth.shape
        vv, uu = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
        z = depth[vv, uu].astype(np.float64) * depth_scale
        valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
        if not np.any(valid):
            continue
        u, v, zc = uu[valid].astype(float), vv[valid].astype(float), z[valid]
        camera_points = np.column_stack(((u - cx) * zc / fx, (v - cy) * zc / fy, zc))
        board_points = (T_board_camera[:3, :3] @ camera_points.T).T + T_board_camera[:3, 3]
        keep = (
            (board_points[:, 0] >= -margin) & (board_points[:, 0] <= board_x + margin)
            & (board_points[:, 1] >= -margin) & (board_points[:, 1] <= board_y + margin)
            & (board_points[:, 2] >= z_min) & (board_points[:, 2] <= z_max)
        )
        if np.any(keep):
            frame_points = board_points[keep]
            frame_colors = color[vv[valid][keep], uu[valid][keep], ::-1].astype(np.float64)
            # Reduce each frame before the multi-view pass. This keeps memory
            # bounded for long handheld captures while retaining support data.
            if voxel is not None:
                frame_points, frame_colors = _voxel_downsample(frame_points, frame_colors, voxel)
            points.append(frame_points)
            colors.append(frame_colors)
            frame_ids.append(np.full(len(frame_points), frame_index, dtype=np.int64))
    if not points:
        raise RuntimeError("no object points survived; check ChArUco poses, depth scale, and crop limits")
    return np.vstack(points), np.vstack(colors), np.concatenate(frame_ids)


def _voxel_downsample(points, colors, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=float)
    color_sums = np.zeros((len(unique), 3), dtype=float)
    counts = np.bincount(inverse).astype(float)
    np.add.at(sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    return sums / counts[:, None], np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8)


def _statistical_filter(points, colors, k=12, std_ratio=2.5):
    if len(points) <= k + 1:
        return points, colors
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1, workers=-1)
    mean_distance = distances[:, 1:].mean(axis=1)
    median = float(np.median(mean_distance))
    mad = float(np.median(np.abs(mean_distance - median)))
    sigma = max(1.4826 * mad, 1e-6)
    keep = mean_distance <= median + std_ratio * sigma
    return points[keep], colors[keep]


def _multi_view_filter(points, colors, frame_ids, voxel, min_views=2):
    """Keep voxels observed from at least ``min_views`` different frames.

    After registration to the fixed board frame, real object surfaces recur
    across views while moving hands or stands do not. ``min_views=1`` disables
    this filter and preserves single-view details.
    """
    if min_views <= 1 or len(points) == 0:
        return points, colors
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    pair = np.column_stack((inverse, frame_ids))
    unique_pairs = np.unique(pair, axis=0)
    support = np.bincount(unique_pairs[:, 0], minlength=int(inverse.max()) + 1)
    keep = support[inverse] >= int(min_views)
    return points[keep], colors[keep]


def _largest_grid_component(points, cell_size=0.004):
    """Return a mask for the largest 26-connected occupied grid component."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    keys, inverse, counts = np.unique(
        np.floor(points / float(cell_size)).astype(np.int64),
        axis=0, return_inverse=True, return_counts=True,
    )
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    visited = np.zeros(len(keys), dtype=bool)
    best = []
    for start in range(len(keys)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            key = keys[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        neighbour = lookup.get((int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)))
                        if neighbour is not None and not visited[neighbour]:
                            visited[neighbour] = True
                            stack.append(neighbour)
        if not best or int(counts[component].sum()) > int(counts[best].sum()):
            best = component
    cell_mask = np.zeros(len(keys), dtype=bool)
    cell_mask[np.asarray(best, dtype=np.int64)] = True
    return cell_mask[inverse]


def _remove_board_and_background(points, colors, model_cfg):
    """Remove the board plane and non-object components in board coordinates.

    A dense component above the board seeds the object footprint. Points close
    to the board are restored only inside that footprint, preserving the real
    object base without restoring the surrounding board/table plane.
    """
    if len(points) == 0:
        return points, colors, {"input": 0, "output": 0, "removed": 0, "core": 0}
    clearance = float(model_cfg.get("board_plane_clearance_m", 0.012))
    seed_z = float(model_cfg.get("component_seed_z_m", 0.020))
    eps = float(model_cfg.get("component_eps_m", 0.004))
    base_margin = float(model_cfg.get("base_xy_margin_m", 0.004))
    above = points[:, 2] >= clearance
    seed_indices = np.flatnonzero(above & (points[:, 2] >= seed_z))
    if len(seed_indices) < 20:
        kept = np.flatnonzero(above)
        return points[kept], colors[kept], {
            "input": int(len(points)), "output": int(len(kept)),
            "removed": int(len(points) - len(kept)), "core": int(len(kept)),
        }
    core_local = _largest_grid_component(points[seed_indices], eps)
    core_indices = seed_indices[core_local]
    if len(core_indices) < 20:
        kept = np.flatnonzero(above)
        return points[kept], colors[kept], {
            "input": int(len(points)), "output": int(len(kept)),
            "removed": int(len(points) - len(kept)), "core": int(len(kept)),
        }
    core = points[core_indices]
    xy_lo = core[:, :2].min(axis=0) - base_margin
    xy_hi = core[:, :2].max(axis=0) + base_margin
    lower = (
        above & (points[:, 2] < seed_z)
        & np.all(points[:, :2] >= xy_lo, axis=1)
        & np.all(points[:, :2] <= xy_hi, axis=1)
    )
    keep = np.zeros(len(points), dtype=bool)
    keep[core_indices] = True
    keep |= lower
    kept = np.flatnonzero(keep)
    return points[kept], colors[kept], {
        "input": int(len(points)), "output": int(len(kept)),
        "removed": int(len(points) - len(kept)), "core": int(len(core_indices)),
    }


def _write_ply(path: Path, points, colors):
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def _write_hull(path: Path, points):
    if len(points) < 4:
        raise ValueError("At least four noncoplanar points are needed to build a mesh")
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ValueError("Cannot form a 3D hull: check point count and viewpoint coverage") from exc
    vertices = hull.vertices
    # ``scipy.spatial.ConvexHull.simplices`` lists the correct facets but does
    # not promise a consistent winding order.  Writing that array directly to
    # OBJ creates a watertight-looking mesh with roughly half its normals
    # pointing inward.  FoundationPose consumes ``mesh.vertex_normals`` during
    # rendering/refinement, so orient every facet explicitly away from an
    # interior point before exporting.
    faces = _oriented_hull_faces(points, hull)
    mapping = {int(index): i + 1 for i, index in enumerate(vertices)}
    with path.open("w", encoding="ascii") as stream:
        stream.write("# Portable mesh generated from RGB-D data; units are meters.\n")
        for point in points[vertices]:
            stream.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        for tri in faces:
            stream.write("f " + " ".join(str(mapping[int(i)]) for i in tri) + "\n")
    return len(vertices), len(faces)


def _oriented_hull_faces(points, hull=None):
    """Return convex-hull faces with a consistent outward winding.

    The returned face indices refer to ``points`` (the same convention as
    ``ConvexHull.simplices``).  The arithmetic uses only NumPy/SciPy, keeping
    the modeling package independent of Trimesh.  The mean of the hull
    vertices is an interior point for a non-degenerate convex hull; each facet
    is flipped when its normal points toward that point.
    """
    if hull is None:
        hull = ConvexHull(points)
    faces = np.asarray(hull.simplices, dtype=np.int64).copy()
    interior = np.asarray(points[np.asarray(hull.vertices, dtype=np.int64)], dtype=float).mean(axis=0)
    for index, tri in enumerate(faces):
        a, b, c = (np.asarray(points[int(i)], dtype=float) for i in tri)
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        if float(np.dot(normal, face_center - interior)) < 0.0:
            faces[index, 1], faces[index, 2] = faces[index, 2], faces[index, 1]
    return faces


def build_model(dataset, config, output=None, *, min_corners=12, pixel_stride=None,
                voxel_size=None, sor=True):
    """Build a portable model package from a captured dataset.

    ``dataset`` must contain ``color/``, ``depth/``, ``poses.csv`` and
    ``camera_info.yaml`` as generated by the capture command.  The returned
    model is centered at its bounding-box center and records its board-frame
    translation in ``model.yaml``.
    """
    dataset = Path(dataset).expanduser().resolve()
    cfg = yaml.safe_load(Path(config).expanduser().read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    output = Path(output).expanduser().resolve() if output else dataset / "model"
    output.mkdir(parents=True, exist_ok=True)
    stride = int(pixel_stride if pixel_stride is not None else model_cfg.get("pixel_stride", 2))
    voxel = float(voxel_size if voxel_size is not None else model_cfg.get("voxel_size_m", 0.002))
    if stride < 1 or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("pixel stride and voxel size must be positive")
    for key in ("support_voxel_size_m", "component_eps_m"):
        value = float(model_cfg.get(key, 0.004))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    points, colors, frame_ids = _fuse(dataset, cfg, int(min_corners), max(1, stride), voxel)
    min_views = int(model_cfg.get("min_voxel_views", 2))
    support_voxel = float(model_cfg.get("support_voxel_size_m", max(0.004, voxel * 2.0)))
    points, colors = _multi_view_filter(points, colors, frame_ids, support_voxel, min_views)
    points, colors, cleanup = _remove_board_and_background(points, colors, model_cfg)
    points, colors = _voxel_downsample(points, colors, voxel)
    before_filter = len(points)
    if sor and bool(model_cfg.get("statistical_outlier_filter", True)):
        points, colors = _statistical_filter(
            points, colors,
            int(model_cfg.get("sor_k", 12)),
            float(model_cfg.get("sor_std_ratio", 2.5)),
        )
    if len(points) < 4:
        raise ValueError("Too few points after filtering; check crop limits, multi-view support and board cleanup")
    board_min, board_max = points.min(axis=0), points.max(axis=0)
    center = (board_min + board_max) / 2.0
    centered = points - center
    ply_path = output / "object_cleaned_centered.ply"
    obj_path = output / "object_foundationpose.obj"
    vertices, faces = _write_hull(obj_path, centered)
    _write_ply(ply_path, centered, colors)
    rel_dataset = dataset.name
    summary = {
        "model_id": output.name,
        "units": "meters",
        "model_frame": "object_model_center",
        "mesh": obj_path.name,
        "point_cloud": ply_path.name,
        "source_dataset": rel_dataset,
        "axis_convention": "axes parallel to ChArUco board frame",
        "board_T_model_translation_m": [float(x) for x in center],
        "extents_m": [float(x) for x in (board_max - board_min)],
        "voxel_size_m": voxel,
        "charuco": cfg["charuco"],
        "build_parameters": {"min_corners": int(min_corners), "pixel_stride": stride, "sor": bool(sor), "model": model_cfg},
        "points_before_outlier_filter": int(before_filter),
        "points_after_filter": int(len(centered)),
        "board_cleanup": cleanup,
        "board_plane_clearance_m": float(model_cfg.get("board_plane_clearance_m", 0.012)),
        "component_seed_z_m": float(model_cfg.get("component_seed_z_m", 0.020)),
        "component_eps_m": float(model_cfg.get("component_eps_m", 0.004)),
        "mesh_vertices": int(vertices),
        "mesh_faces": int(faces),
        "mesh_quality": {
            "topology": "convex_hull",
            "face_winding": "consistent_outward",
            "normals_ready_for_foundationpose": True,
        },
        "warning": "OBJ is a coarse convex-hull mesh with repaired outward normals; use the cleaned PLY for inspection.",
    }
    (output / "model.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    (output / "quality_report.yaml").write_text(yaml.safe_dump({
        "point_count": int(len(centered)),
        "bbox_min_m": [float(x) for x in centered.min(axis=0)],
        "bbox_max_m": [float(x) for x in centered.max(axis=0)],
        "extents_m": summary["extents_m"],
        "outlier_filter_removed": int(before_filter - len(centered)),
    }, sort_keys=False), encoding="utf-8")
    return BuildResult(output, len(centered), vertices, faces, tuple(summary["extents_m"]))
