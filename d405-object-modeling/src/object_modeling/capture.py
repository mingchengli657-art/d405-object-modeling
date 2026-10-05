"""ROS 2 RGB-D capture for a fixed object and a hand-held D405.

Every accepted frame is registered to the ChArUco board. The board and the
object must remain rigidly attached; the camera may be moved around them.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation


def _board(cfg):
    spec = cfg["charuco"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    args = (int(spec["squares_x"]), int(spec["squares_y"]),
            float(spec["square_length_m"]), float(spec["marker_length_m"]), dictionary)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(*args)
    else:
        board = cv2.aruco.CharucoBoard((args[0], args[1]), args[2], args[3], args[4])
    return dictionary, board, cv2.aruco.DetectorParameters_create()


def _board_points(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return np.asarray(board.chessboardCorners, dtype=np.float32)


def _pose(gray, dictionary, board, detector, K, D):
    """Return board<-camera pose and quality metrics for one RGB image."""
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector)
    empty = {"reprojection_px": float("inf"), "corner_area_ratio": 0.0,
             "corner_span_x": 0.0, "corner_span_y": 0.0}
    if ids is None or len(ids) == 0:
        return None, corners, ids, 0, empty
    try:
        count, cc, ci = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board, cameraMatrix=K, distCoeffs=D)
        count = int(count) if count is not None else 0
        if ci is None or cc is None or count < 4:
            return None, corners, ids, count, empty
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            cc, ci, board, K, D, np.zeros((3, 1)), np.zeros((3, 1)))
        if not ok:
            return None, corners, ids, count, empty

        charuco = np.asarray(cc, dtype=np.float32).reshape(-1, 2)
        indices = np.asarray(ci, dtype=np.int32).reshape(-1)
        object_points = _board_points(board)[indices]
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        projected = projected.reshape(-1, 2)
        reprojection = float(np.sqrt(np.mean(np.sum((projected - charuco) ** 2, axis=1))))
        hull = cv2.convexHull(charuco.reshape(-1, 1, 2))
        area_ratio = float(cv2.contourArea(hull) / max(1, gray.shape[0] * gray.shape[1]))
        span_x = float((charuco[:, 0].max() - charuco[:, 0].min()) / max(1, gray.shape[1]))
        span_y = float((charuco[:, 1].max() - charuco[:, 1].min()) / max(1, gray.shape[0]))

        Rm, _ = cv2.Rodrigues(rvec)
        T_camera_board = np.eye(4, dtype=float)
        T_camera_board[:3, :3] = Rm
        T_camera_board[:3, 3] = np.asarray(tvec).reshape(3)
        metrics = {"reprojection_px": reprojection,
                   "corner_area_ratio": area_ratio,
                   "corner_span_x": span_x,
                   "corner_span_y": span_y}
        # estimatePoseCharucoBoard returns camera<-board; invert it so the
        # returned transform maps camera points into board coordinates.
        return np.linalg.inv(T_camera_board), corners, ids, count, metrics
    except cv2.error:
        return None, corners, ids, 0, empty


def _stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _view_bin(camera_xyz, azimuth_bins, elevation_edges):
    x, y, z = [float(v) for v in camera_xyz]
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    elevation = math.degrees(math.atan2(z, math.hypot(x, y)))
    az_bin = int(math.floor(azimuth / 360.0 * azimuth_bins)) % azimuth_bins
    edges = np.asarray(elevation_edges, dtype=float)
    el_bin = int(np.searchsorted(edges, elevation, side="right") - 1)
    el_bin = max(0, min(el_bin, len(edges) - 2))
    return f"a{az_bin:02d}_e{el_bin:02d}", azimuth, elevation


class CaptureNode(Node):
    def __init__(self, args, cfg):
        super().__init__("object_modeling_capture")
        self.args, self.cfg = args, cfg
        self.bridge = CvBridge()
        self.dictionary, self.board, self.detector = _board(cfg)
        self.K = self.D = None
        self.last_stamp = 0
        self.saved = 0
        self.last_log = 0.0
        self.view_counts = {}
        self.out = Path(args.output).expanduser().resolve()
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"Capture directory is not empty: {self.out}. Choose a new directory.")
        for name in ("color", "depth", "overlays"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out / "manifest.yaml"
        self.manifest = {
            "format_version": 2,
            "charuco": cfg["charuco"],
            "capture_mode": "handheld_static_object",
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "interval_ms": float(args.interval_ms),
            "sync_slop_ms": float(args.sync_slop_ms),
            "min_charuco_corners": int(args.min_corners),
            "max_reprojection_error_px": float(args.max_reprojection_px),
            "min_corner_area_ratio": float(args.min_corner_area_ratio),
            "min_sharpness": float(args.min_sharpness),
            "min_depth_valid_ratio": float(args.min_depth_valid_ratio),
            "azimuth_bins": int(args.azimuth_bins),
            "elevation_edges_deg": [float(x) for x in args.elevation_edges_deg],
            "max_per_view_bin": int(args.max_per_view_bin),
            "depth_scale": float(args.depth_scale),
            "coordinate_frame": "charuco_board",
        }
        self._write_manifest()
        self.pose_file = (self.out / "poses.csv").open("w", newline="", encoding="utf-8")
        self.pose_writer = csv.writer(self.pose_file)
        self.pose_writer.writerow([
            "frame", "stamp_ns", "charuco_corners", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            "stamp_delta_ms", "reprojection_px", "corner_area_ratio", "corner_span_x", "corner_span_y",
            "sharpness", "depth_valid_ratio", "camera_x", "camera_y", "camera_z",
            "azimuth_deg", "elevation_deg", "view_bin",
        ])
        color = Subscriber(self, Image, args.color_topic)
        depth = Subscriber(self, Image, args.depth_topic)
        sync = ApproximateTimeSynchronizer(
            [color, depth], queue_size=50, slop=float(args.sync_slop_ms) / 1000.0)
        sync.registerCallback(self._rgbd)
        self.sync = sync
        self.create_subscription(CameraInfo, args.camera_info_topic, self._info, 10)
        self.get_logger().info(
            f"capturing handheld RGB-D at interval {args.interval_ms:.0f} ms; output={self.out}")

    def _write_manifest(self):
        self.manifest["accepted_frames"] = int(self.saved)
        self.manifest["covered_view_bins"] = sorted(self.view_counts)
        self.manifest_path.write_text(yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8")

    def _info(self, msg):
        self.K = np.asarray(msg.k, dtype=float).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=float)
        info_path = self.out / "camera_info.yaml"
        if not info_path.exists():
            info_path.write_text(yaml.safe_dump({
                "width": int(msg.width), "height": int(msg.height),
                "distortion_model": msg.distortion_model, "k": [float(x) for x in msg.k],
                "d": [float(x) for x in msg.d], "camera_frame": msg.header.frame_id,
                "depth_scale": float(self.args.depth_scale),
            }, sort_keys=False), encoding="utf-8")

    def _rgbd(self, color_msg, depth_msg):
        if self.K is None or (self.saved >= self.args.max_frames > 0):
            return
        color_stamp = _stamp_ns(color_msg)
        depth_stamp = _stamp_ns(depth_msg)
        if color_stamp <= 0 or depth_stamp <= 0:
            return
        stamp_delta_ms = abs(color_stamp - depth_stamp) / 1e6
        if color_stamp - self.last_stamp < int(self.args.interval_ms * 1_000_000):
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"image conversion failed: {exc}")
            return
        if depth.ndim != 2 or color.shape[:2] != depth.shape[:2]:
            return
        if depth.dtype != np.uint16:
            values = np.asarray(depth, dtype=float)
            depth = np.zeros(values.shape, dtype=np.uint16)
            valid = np.isfinite(values) & (values > 0)
            depth[valid] = np.clip(np.rint(values[valid] / self.args.depth_scale), 0, 65535).astype(np.uint16)

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pose, corners, ids, count, metrics = _pose(
            gray, self.dictionary, self.board, self.detector, self.K, self.D)
        overlay = color.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        depth_valid_ratio = float(np.count_nonzero(depth)) / max(1, depth.size)
        reasons = []
        if stamp_delta_ms > self.args.sync_slop_ms:
            reasons.append("rgbd_sync")
        if count < self.args.min_corners:
            reasons.append(f"corners<{self.args.min_corners}")
        if metrics["reprojection_px"] > self.args.max_reprojection_px:
            reasons.append("reprojection")
        if metrics["corner_area_ratio"] < self.args.min_corner_area_ratio:
            reasons.append("corner_spread")
        if sharpness < self.args.min_sharpness:
            reasons.append("blur")
        if depth_valid_ratio < self.args.min_depth_valid_ratio:
            reasons.append("depth")

        view_bin = ""
        azimuth = elevation = float("nan")
        camera_xyz = np.zeros(3, dtype=float)
        if pose is None:
            reasons.append("no_pose")
        else:
            camera_xyz = pose[:3, 3]
            view_bin, azimuth, elevation = _view_bin(
                camera_xyz, self.args.azimuth_bins, self.args.elevation_edges_deg)
            if self.args.max_per_view_bin > 0 and self.view_counts.get(view_bin, 0) >= self.args.max_per_view_bin:
                reasons.append("view_bin_full")

        accepted = pose is not None and not reasons
        if not accepted:
            if self.args.show:
                text = (f"REJECT corners={count} reproj={metrics['reprojection_px']:.1f}px "
                        f"sharp={sharpness:.0f} dt={stamp_delta_ms:.0f}ms "
                        + (";".join(reasons) or "unknown"))
                cv2.putText(overlay, text[:150], (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0, 0, 255), 2)
                cv2.imshow("object_modeling capture", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rclpy.shutdown()
            return

        frame = f"{self.saved:06d}"
        if not cv2.imwrite(str(self.out / "color" / f"{frame}.png"), color) or not cv2.imwrite(str(self.out / "depth" / f"{frame}.png"), depth):
            raise OSError("Failed to save RGB-D images")
        cv2.putText(overlay, (f"ACCEPT {view_bin} corners={count} "
                              f"reproj={metrics['reprojection_px']:.1f}px "
                              f"views={sum(self.view_counts.values()) + 1}"),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2)
        if not cv2.imwrite(str(self.out / "overlays" / f"{frame}.png"), overlay):
            raise OSError("Failed to save overlay")
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        self.pose_writer.writerow([
            frame, color_stamp, count, *pose[:3, 3], *q,
            stamp_delta_ms, metrics["reprojection_px"], metrics["corner_area_ratio"],
            metrics["corner_span_x"], metrics["corner_span_y"], sharpness, depth_valid_ratio,
            *camera_xyz, azimuth, elevation, view_bin,
        ])
        self.pose_file.flush()
        self.saved += 1
        self.last_stamp = color_stamp
        self.view_counts[view_bin] = self.view_counts.get(view_bin, 0) + 1
        if time.monotonic() - self.last_log > 1.0:
            self.get_logger().info(
                f"saved={self.saved}, corners={count}, view={view_bin}, "
                f"reproj={metrics['reprojection_px']:.2f}px")
            self.last_log = time.monotonic()
        if self.args.show:
            cv2.imshow("object_modeling capture", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()

        if self.saved >= self.args.max_frames > 0 and rclpy.ok():
            rclpy.shutdown()

    def close(self):
        self.pose_file.close()
        self._write_manifest()
        cv2.destroyAllWindows()


def run_capture(args, cfg):
    rclpy.init()
    node = CaptureNode(args, cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def add_capture_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--color-topic", default=None)
    parser.add_argument("--depth-topic", default=None)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--min-corners", type=int, default=None)
    parser.add_argument("--interval-ms", type=float, default=None)
    parser.add_argument("--sync-slop-ms", type=float, default=None)
    parser.add_argument("--max-reprojection-px", type=float, default=None)
    parser.add_argument("--min-corner-area-ratio", type=float, default=None)
    parser.add_argument("--min-sharpness", type=float, default=None)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=None)
    parser.add_argument("--max-per-view-bin", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")


def capture_from_args(args):
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8"))
    camera, cap = cfg["camera"], cfg["capture"]
    args.color_topic = args.color_topic or camera["color_topic"]
    args.depth_topic = args.depth_topic or camera["depth_topic"]
    args.camera_info_topic = args.camera_info_topic or camera["camera_info_topic"]
    args.depth_scale = float(args.depth_scale if args.depth_scale is not None else camera.get("depth_scale", 0.001))
    args.min_corners = int(args.min_corners if args.min_corners is not None else cap.get("min_charuco_corners", 12))
    args.interval_ms = float(args.interval_ms if args.interval_ms is not None else cap.get("interval_ms", 230))
    args.sync_slop_ms = float(args.sync_slop_ms if args.sync_slop_ms is not None else cap.get("sync_slop_ms", 50))
    args.max_reprojection_px = float(args.max_reprojection_px if args.max_reprojection_px is not None else cap.get("max_reprojection_error_px", 3.0))
    args.min_corner_area_ratio = float(args.min_corner_area_ratio if args.min_corner_area_ratio is not None else cap.get("min_corner_area_ratio", 0.015))
    args.min_sharpness = float(args.min_sharpness if args.min_sharpness is not None else cap.get("min_sharpness", 20.0))
    args.min_depth_valid_ratio = float(args.min_depth_valid_ratio if args.min_depth_valid_ratio is not None else cap.get("min_depth_valid_ratio", 0.20))
    args.azimuth_bins = int(cap.get("azimuth_bins", 24))
    args.elevation_edges_deg = [float(x) for x in cap.get("elevation_edges_deg", [0, 25, 45, 65, 90])]
    args.max_per_view_bin = int(args.max_per_view_bin if args.max_per_view_bin is not None else cap.get("max_per_view_bin", 4))
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    if args.interval_ms < 0 or args.sync_slop_ms <= 0 or args.azimuth_bins < 1:
        raise ValueError("invalid capture timing or view-bin configuration")
    run_capture(args, cfg)
