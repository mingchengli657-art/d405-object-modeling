"""Command line interface for object_modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .core import build_model
from .validate import report_dict, validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(prog="object-modeling")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="capture D405 RGB-D and ChArUco poses around a fixed object")
    capture.add_argument("--config", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--color-topic", default=None)
    capture.add_argument("--depth-topic", default=None)
    capture.add_argument("--camera-info-topic", default=None)
    capture.add_argument("--depth-scale", type=float, default=None)
    capture.add_argument("--min-corners", type=int, default=None)
    capture.add_argument("--interval-ms", type=float, default=None)
    capture.add_argument("--sync-slop-ms", type=float, default=None)
    capture.add_argument("--max-reprojection-px", type=float, default=None)
    capture.add_argument("--min-corner-area-ratio", type=float, default=None)
    capture.add_argument("--min-sharpness", type=float, default=None)
    capture.add_argument("--min-depth-valid-ratio", type=float, default=None)
    capture.add_argument("--max-per-view-bin", type=int, default=None)
    capture.add_argument("--max-frames", type=int, default=0)
    capture.add_argument("--show", action="store_true")
    build = sub.add_parser("build", help="fuse a dataset into a portable model package")
    build.add_argument("--dataset", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--output", default=None)
    build.add_argument("--min-corners", type=int, default=12)
    build.add_argument("--pixel-stride", type=int, default=None)
    build.add_argument("--voxel-size", type=float, default=None)
    build.add_argument("--no-sor", action="store_true")
    validate = sub.add_parser("validate", help="validate captured data")
    validate.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        from .capture import capture_from_args
        capture_from_args(args)
    elif args.command == "build":
        result = build_model(args.dataset, args.config, args.output,
                             min_corners=args.min_corners,
                             pixel_stride=args.pixel_stride,
                             voxel_size=args.voxel_size, sor=not args.no_sor)
        print(yaml.safe_dump({
            "model_dir": result.model_dir.name,
            "point_count": result.point_count,
            "mesh_vertices": result.mesh_vertices,
            "mesh_faces": result.mesh_faces,
            "extents_m": list(result.extents_m),
        }, sort_keys=False).strip())
    else:
        print(yaml.safe_dump(report_dict(validate_dataset(args.dataset)), sort_keys=False).strip())


if __name__ == "__main__":
    main()
