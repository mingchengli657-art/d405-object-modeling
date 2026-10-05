"""Portable dataset and model-package validation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    color_frames: int
    depth_frames: int
    pose_frames: int
    usable_frames: int
    min_corners: int
    median_corners: float
    median_depth_valid_ratio: float
    median_reprojection_px: float
    median_corner_area_ratio: float
    median_sharpness: float
    median_stamp_delta_ms: float
    covered_view_bins: int
    warnings: tuple[str, ...]


def validate_dataset(dataset) -> ValidationReport:
    root = Path(dataset).expanduser().resolve()
    colors = sorted((root / "color").glob("*.png"))
    depths = sorted((root / "depth").glob("*.png"))
    poses_path = root / "poses.csv"
    rows = list(csv.DictReader(poses_path.open(encoding="utf-8"))) if poses_path.exists() else []
    corners = [int(row.get("charuco_corners", 0)) for row in rows]
    usable = []
    invalid_frames = []
    for row in rows:
        frame = row["frame"]
        color = cv2.imread(str(root / "color" / f"{frame}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(root / "depth" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or depth.ndim != 2 or color.shape[:2] != depth.shape[:2]:
            invalid_frames.append(frame)
        elif int(row.get("charuco_corners", 0)) >= 12:
            usable.append(row)
    def numbers(key):
        values = []
        for row in rows:
            try:
                value = float(row[key])
                if np.isfinite(value):
                    values.append(value)
            except (KeyError, TypeError, ValueError):
                pass
        return values
    reprojection = numbers("reprojection_px")
    area = numbers("corner_area_ratio")
    sharpness = numbers("sharpness")
    stamp_delta = numbers("stamp_delta_ms")
    view_bins = {row.get("view_bin", "") for row in rows if row.get("view_bin")}
    ratios = []
    for path in depths[: min(len(depths), 30)]:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None and image.ndim == 2:
            ratios.append(float(np.count_nonzero(image)) / image.size)
    warnings = []
    if invalid_frames:
        warnings.append(f"{len(invalid_frames)} pose rows have missing, unreadable or mismatched RGB-D images")
    if not poses_path.exists():
        warnings.append("poses.csv is missing")
    if not (root / "camera_info.yaml").exists():
        warnings.append("camera_info.yaml is missing")
    if len(colors) != len(depths):
        warnings.append("color/depth frame counts differ")
    if len(usable) < 30:
        warnings.append("fewer than 30 usable ChArUco frames")
    if corners and min(corners) < 12:
        warnings.append("some saved pose rows have too few corners")
    if reprojection and float(np.median(reprojection)) > 3.0:
        warnings.append("median ChArUco reprojection error exceeds 3 px")
    if stamp_delta and float(np.median(stamp_delta)) > 50.0:
        warnings.append("median RGB-D timestamp gap exceeds 50 ms")
    return ValidationReport(
        dataset=root.name,
        color_frames=len(colors), depth_frames=len(depths), pose_frames=len(rows),
        usable_frames=len(usable), min_corners=min(corners) if corners else 0,
        median_corners=float(np.median(corners)) if corners else 0.0,
        median_depth_valid_ratio=float(np.median(ratios)) if ratios else 0.0,
        median_reprojection_px=float(np.median(reprojection)) if reprojection else 0.0,
        median_corner_area_ratio=float(np.median(area)) if area else 0.0,
        median_sharpness=float(np.median(sharpness)) if sharpness else 0.0,
        median_stamp_delta_ms=float(np.median(stamp_delta)) if stamp_delta else 0.0,
        covered_view_bins=len(view_bins),
        warnings=tuple(warnings),
    )


def report_dict(report):
    return asdict(report)
