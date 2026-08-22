"""RGB-D perception helpers for non-privileged UniVTAC control."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as SciRotation


@dataclass(frozen=True)
class RgbdFrame:
    """One calibrated RGB-D observation in the ROS optical camera frame."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_position: np.ndarray
    camera_quaternion_wxyz: np.ndarray
    camera_name: str = "head"

    def validated(self) -> "RgbdFrame":
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth, dtype=np.float32).squeeze()
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64).reshape(3, 3)
        position = np.asarray(self.camera_position, dtype=np.float64).reshape(3)
        quaternion = np.asarray(self.camera_quaternion_wxyz, dtype=np.float64).reshape(4)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"RGB image must have shape (H, W, 3+), got {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8, copy=False))
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth shape {depth.shape} must match RGB shape {rgb.shape[:2]}"
            )
        if not np.all(np.isfinite(intrinsics)) or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("camera intrinsics must be finite with positive focal lengths")
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("camera quaternion must be finite and non-zero")
        return RgbdFrame(
            rgb=rgb,
            depth=np.ascontiguousarray(depth),
            intrinsics=intrinsics,
            camera_position=position,
            camera_quaternion_wxyz=quaternion / norm,
            camera_name=str(self.camera_name),
        )


@dataclass(frozen=True)
class ObjectEstimate:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    extent: np.ndarray
    mask: np.ndarray
    points_world: np.ndarray
    score: float
    prompt: str


@dataclass(frozen=True)
class GraspEstimate:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    mask: np.ndarray
    points_world: np.ndarray
    object_position: np.ndarray
    object_quaternion_wxyz: np.ndarray
    object_extent: np.ndarray
    scores: np.ndarray
    grasps_camera: np.ndarray
    selected_index: int
    prompt: str


class RgbdPerceptionError(RuntimeError):
    """Structured RGB-D perception failure for retry and artifact logging."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self.reason = str(reason)
        self.diagnostics = dict(diagnostics or {})
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        super().__init__(message)


class UniVTACRgbdPerception:
    """Original CaP-style SAM3 and Contact-GraspNet RGB-D pipeline."""

    def __init__(
        self,
        *,
        sam3_url: str = "http://127.0.0.1:8114",
        graspnet_url: str = "http://127.0.0.1:8115",
        request_timeout_seconds: float = 120.0,
        min_depth_points: int = 32,
        grasp_local_z_offset: float = 0.12,
        mask_dilation_radii: tuple[int, ...] | list[int] = (0, 2, 4, 8),
    ) -> None:
        self.sam3_url = sam3_url.rstrip("/")
        self.graspnet_url = graspnet_url.rstrip("/")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.min_depth_points = max(4, int(min_depth_points))
        self.grasp_local_z_offset = float(grasp_local_z_offset)
        radii = [max(0, int(radius)) for radius in mask_dilation_radii]
        self.mask_dilation_radii = tuple(dict.fromkeys(radii or [0]))

    def estimate_object(self, frame: RgbdFrame, prompt: str) -> ObjectEstimate:
        frame = frame.validated()
        mask, score = self._segment(frame.rgb, prompt)
        points_world, depth_mask, _diagnostics = self.masked_points_world_with_mask(frame, mask)
        try:
            position, quaternion, extent = _oriented_bounding_box(points_world)
        except RuntimeError as exc:
            raise RgbdPerceptionError(
                "rgbd_obb_failed",
                str(exc),
                diagnostics={"points_world": int(len(points_world))},
                mask=depth_mask,
            ) from exc
        return ObjectEstimate(
            position=position.astype(np.float32),
            quaternion_wxyz=quaternion.astype(np.float32),
            extent=extent.astype(np.float32),
            mask=depth_mask,
            points_world=points_world.astype(np.float32),
            score=float(score),
            prompt=str(prompt),
        )

    def estimate_grasp(self, frame: RgbdFrame, prompt: str) -> GraspEstimate:
        frame = frame.validated()
        mask, _score = self._segment(frame.rgb, prompt)
        # Validate masked depth before invoking the heavier grasp service.
        points_world, depth_mask, _diagnostics = self.masked_points_world_with_mask(frame, mask)
        try:
            object_position, object_quaternion, object_extent = _oriented_bounding_box(
                points_world
            )
        except RuntimeError as exc:
            raise RgbdPerceptionError(
                "rgbd_obb_failed",
                str(exc),
                diagnostics={"points_world": int(len(points_world))},
                mask=depth_mask,
            ) from exc
        grasps, scores = self._request_grasps(frame.depth, frame.intrinsics, depth_mask)
        grasps = np.asarray(grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) == 0:
            raise RgbdPerceptionError(
                "graspnet_invalid_grasps",
                f"Contact-GraspNet returned invalid grasps shape {grasps.shape}",
                diagnostics={"grasps_shape": tuple(int(v) for v in grasps.shape)},
                mask=depth_mask,
            )
        if scores.size != len(grasps) or not np.any(np.isfinite(scores)):
            raise RgbdPerceptionError(
                "graspnet_invalid_scores",
                "Contact-GraspNet returned invalid grasp scores",
                diagnostics={
                    "scores_shape": tuple(int(v) for v in scores.shape),
                    "grasps_count": int(len(grasps)),
                },
                mask=depth_mask,
            )
        selected_index = int(np.nanargmax(scores))
        grasp_camera = grasps[selected_index].copy()
        offset = np.eye(4, dtype=np.float64)
        offset[2, 3] = self.grasp_local_z_offset
        grasp_camera = grasp_camera @ offset
        grasp_world = _camera_to_world_matrix(frame) @ grasp_camera
        quat_xyzw = SciRotation.from_matrix(grasp_world[:3, :3]).as_quat()
        quaternion = quat_xyzw[[3, 0, 1, 2]]
        return GraspEstimate(
            position=grasp_world[:3, 3].astype(np.float32),
            quaternion_wxyz=quaternion.astype(np.float32),
            mask=depth_mask,
            points_world=points_world.astype(np.float32),
            object_position=object_position.astype(np.float32),
            object_quaternion_wxyz=object_quaternion.astype(np.float32),
            object_extent=object_extent.astype(np.float32),
            scores=scores.astype(np.float32),
            grasps_camera=grasps.astype(np.float32),
            selected_index=selected_index,
            prompt=str(prompt),
        )

    def masked_points_world(self, frame: RgbdFrame, mask: np.ndarray) -> np.ndarray:
        points_world, _mask, _diagnostics = self.masked_points_world_with_mask(frame, mask)
        return points_world

    def masked_points_world_with_mask(
        self,
        frame: RgbdFrame,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        frame = frame.validated()
        mask = np.asarray(mask, dtype=bool).squeeze()
        if mask.shape != frame.depth.shape:
            raise RgbdPerceptionError(
                "mask_depth_shape_mismatch",
                f"SAM3 mask shape {mask.shape} does not match depth shape {frame.depth.shape}",
                diagnostics={
                    "mask_shape": tuple(int(v) for v in mask.shape),
                    "depth_shape": tuple(int(v) for v in frame.depth.shape),
                },
                mask=mask,
            )
        base_valid_depth = np.isfinite(frame.depth) & (frame.depth > 0.01) & (frame.depth < 20.0)
        best_mask = mask
        best_valid_count = 0
        best_radius = 0
        for radius in self.mask_dilation_radii:
            candidate = mask if radius <= 0 else binary_dilation(mask, iterations=radius)
            valid = candidate & base_valid_depth
            valid_count = int(np.count_nonzero(valid))
            if valid_count > best_valid_count:
                best_mask = np.asarray(candidate, dtype=bool)
                best_valid_count = valid_count
                best_radius = int(radius)
            if valid_count >= self.min_depth_points:
                best_mask = np.asarray(candidate, dtype=bool)
                best_valid_count = valid_count
                best_radius = int(radius)
                break
        valid = best_mask & base_valid_depth
        ys, xs = np.nonzero(valid)
        if len(xs) < self.min_depth_points:
            diagnostics = {
                "valid_depth_points": int(len(xs)),
                "min_depth_points": int(self.min_depth_points),
                "mask_area_px": int(np.count_nonzero(mask)),
                "dilated_mask_area_px": int(np.count_nonzero(best_mask)),
                "dilation_radius": int(best_radius),
                "finite_depth_px": int(np.count_nonzero(np.isfinite(frame.depth))),
                "positive_depth_px": int(np.count_nonzero(base_valid_depth)),
            }
            raise RgbdPerceptionError(
                "insufficient_depth_points",
                f"RGB-D detection has only {len(xs)} valid depth points; "
                f"requires at least {self.min_depth_points}",
                diagnostics=diagnostics,
                mask=best_mask,
            )
        z = frame.depth[ys, xs].astype(np.float64)
        k = frame.intrinsics
        points_camera = np.column_stack(
            (
                (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0],
                (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1],
                z,
            )
        )
        world_from_camera = _camera_to_world_matrix(frame)
        points_world = (
            points_camera @ world_from_camera[:3, :3].T
            + world_from_camera[:3, 3]
        )
        diagnostics = {
            "valid_depth_points": int(len(xs)),
            "min_depth_points": int(self.min_depth_points),
            "mask_area_px": int(np.count_nonzero(mask)),
            "dilated_mask_area_px": int(np.count_nonzero(best_mask)),
            "dilation_radius": int(best_radius),
            "finite_depth_px": int(np.count_nonzero(np.isfinite(frame.depth))),
            "positive_depth_px": int(np.count_nonzero(base_valid_depth)),
        }
        return points_world, best_mask, diagnostics

    def _segment(self, rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
        payload = {
            "image_base64": _encode_png(rgb),
            "text_prompt": str(prompt),
        }
        try:
            response = requests.post(
                f"{self.sam3_url}/segment",
                json=payload,
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            raise RgbdPerceptionError(
                "sam3_request_failed",
                f"SAM3 request failed: {exc}",
            ) from exc
        if not results:
            raise RgbdPerceptionError(
                "sam3_no_detection",
                f"SAM3 returned no detection for {prompt!r}",
            )
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        try:
            shape = tuple(int(v) for v in best["shape"])
            mask = np.frombuffer(
                base64.b64decode(best["mask_base64"]),
                dtype=np.uint8,
            ).reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise RgbdPerceptionError(
                "sam3_invalid_mask",
                f"SAM3 returned an invalid mask: {exc}",
            ) from exc
        return np.asarray(mask, dtype=bool).squeeze(), float(best.get("score", 0.0))

    def _request_grasps(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        segmap = np.asarray(mask, dtype=np.uint8)
        payload: dict[str, Any] = {
            "depth_base64": _encode_npy(np.asarray(depth, dtype=np.float32)),
            "cam_K_base64": _encode_npy(np.asarray(intrinsics, dtype=np.float32)),
            "segmap_base64": _encode_npy(segmap),
            "segmap_id": 1,
            "local_regions": True,
            "filter_grasps": True,
            "skip_border_objects": False,
            "z_range": [0.2, 2.0],
            "forward_passes": 2,
            "max_retries": 10,
        }
        try:
            response = requests.post(
                f"{self.graspnet_url}/plan",
                json=payload,
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            grasps = _decode_npy(data["grasps_base64"])
            scores = _decode_npy(data["scores_base64"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise RgbdPerceptionError(
                "graspnet_request_failed",
                f"Contact-GraspNet request failed: {exc}",
            ) from exc
        return grasps, scores


def _camera_to_world_matrix(frame: RgbdFrame) -> np.ndarray:
    frame = frame.validated()
    quat_xyzw = frame.camera_quaternion_wxyz[[1, 2, 3, 0]]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = SciRotation.from_quat(quat_xyzw).as_matrix()
    matrix[:3, 3] = frame.camera_position
    return matrix


def _oriented_bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise RuntimeError(f"cannot fit RGB-D OBB from points shape {points.shape}")
    mean = points.mean(axis=0)
    covariance = np.cov(points - mean, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    for column in range(3):
        dominant = int(np.argmax(np.abs(axes[:, column])))
        if axes[dominant, column] < 0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    local = (points - mean) @ axes
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    center = mean + axes @ ((lower + upper) * 0.5)
    extent = upper - lower
    quat_xyzw = SciRotation.from_matrix(axes).as_quat()
    quaternion = quat_xyzw[[3, 0, 1, 2]]
    return center, quaternion, extent


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_npy(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_npy(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)
