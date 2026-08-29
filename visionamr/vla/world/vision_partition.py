"""Cached semantic vision partition used once per geometry instance."""  # Describe the non-iterative VLA perception boundary.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import dataclass  # Import immutable semantic-region contracts.
import json  # Import the cached-vision interchange format.
from pathlib import Path  # Import portable cache paths.
from typing import Any  # Import generic repository problem and mesh types.
import numpy as np  # Import normalized geometric distance operations.

@dataclass(frozen=True)
class VisionRegion:  # Store one semantic region proposed by a VLM or deterministic fixture.
    name: str  # Store the mechanism identifier.
    center: tuple[float, float, float]  # Store the mechanism centre in physical coordinates.
    radius: tuple[float, float, float]  # Store an anisotropic influence radius.
    priority: float  # Store overlap priority without prescribing mesh size.

class CachedVisionPartition:  # Assign remeshed elements to one fixed semantic vision output.
    def __init__(self, regions: list[VisionRegion], bbox: tuple[float, ...], *, field_name: str = "field_remainder") -> None:  # Initialize a stable geometry-level partition.
        if not regions:  # Reject an empty vision result.
            raise ValueError("cached vision partition requires at least one semantic region")  # Explain the partition contract.
        self.regions = tuple(regions)  # Freeze semantic regions for the whole adaptive trajectory.
        self.bbox = tuple(float(value) for value in bbox)  # Freeze the physical bounding box.
        self.field_name = str(field_name)  # Store the generic background-region name.
        self.names = tuple(region.name for region in self.regions) + (self.field_name,)  # Expose stable names including the generic field region.
    @classmethod
    def from_problem(cls, problem: Any) -> "CachedVisionPartition":  # Build a deterministic scripted-VLM fixture from declared bridge features.
        bbox = tuple(float(value) for value in problem.bbox)  # Read the exact problem bounding box.
        lower = np.asarray(bbox[:3], dtype=float)  # Read the lower three-dimensional corner.
        upper = np.asarray(bbox[3:6], dtype=float)  # Read the upper three-dimensional corner.
        extent = np.maximum(upper - lower, 1.0e-9)  # Compute non-zero physical extents.
        groups: dict[str, list[np.ndarray]] = {}  # Group feature anchors into interpretable bridge mechanisms.
        for feature in problem.features:  # Traverse all explicitly declared geometry and boundary features.
            name = str(getattr(feature, "name", "feature")).lower()  # Normalize the feature name.
            kind = str(getattr(feature, "kind", "feature")).lower()  # Normalize the feature kind.
            point = np.asarray([float(feature.x), float(feature.y), float(getattr(feature, "z", 0.0))], dtype=float)  # Read the physical anchor point.
            if any(token in name for token in ("opening", "hole", "rim")) or kind == "hole":  # Group the inspection-opening mechanism.
                group = "inspection_opening"  # Assign the persistent opening-rim mechanism name.
            elif any(token in name for token in ("wheel", "load")) or kind == "load":  # Group the eccentric wheel-load mechanism.
                group = "wheel_load"  # Assign the load-footprint mechanism name.
            elif "left" in name and any(token in name for token in ("bearing", "support")):  # Separate the first bearing mechanism.
                group = "left_bearing"  # Assign the first bearing-region name.
            elif "right" in name and any(token in name for token in ("bearing", "support")):  # Separate the second bearing mechanism.
                group = "right_bearing"  # Assign the second bearing-region name.
            elif any(token in name for token in ("joint", "diaphragm", "web", "flange")) or kind == "corner":  # Group diaphragm-to-box load-path joints.
                group = "diaphragm_joints"  # Assign the structural transition-region name.
            else:  # Preserve any additional explicit feature as its own semantic mechanism.
                group = str(getattr(feature, "name", "feature"))  # Use the declared feature name without inventing a mesh parameter.
            groups.setdefault(group, []).append(point)  # Append the feature point to its mechanism group.
        regions: list[VisionRegion] = []  # Collect stable semantic vision regions.
        for name in sorted(groups):  # Build regions in deterministic name order.
            points = np.stack(groups[name], axis=0)  # Materialize all anchors belonging to this mechanism.
            center = np.mean(points, axis=0)  # Use the anchor centroid as the cached visual centre.
            if name == "inspection_opening":  # Allocate a broad anisotropic region around the opening rim.
                radius = np.asarray([0.15, 0.25, 0.28], dtype=float) * extent  # Cover the diaphragm thickness and in-plane rim field.
                priority = 1.25  # Give the persistent geometric singularity strong overlap priority.
            elif name == "wheel_load":  # Allocate a local top-flange load-path region.
                radius = np.asarray([0.24, 0.22, 0.16], dtype=float) * extent  # Cover the wheel patch and immediate flange spread.
                priority = 1.10  # Give the applied-load mechanism high overlap priority.
            elif "bearing" in name:  # Allocate local support-transfer regions.
                radius = np.asarray([0.22, 0.17, 0.16], dtype=float) * extent  # Cover the bearing pad and adjacent web-foot path.
                priority = 1.05  # Give support footprints high overlap priority.
            elif name == "diaphragm_joints":  # Allocate a distributed box-to-diaphragm transition region.
                radius = np.asarray([0.16, 0.34, 0.42], dtype=float) * extent  # Cover web and flange joint lines without swallowing the whole box.
                priority = 0.95  # Retain the load-path region behind sharper local mechanisms.
            else:  # Allocate a neutral region for additional declared features.
                radius = np.asarray([0.16, 0.16, 0.16], dtype=float) * extent  # Use a bounded isotropic influence radius.
                priority = 0.75  # Give unspecified mechanisms neutral overlap priority.
            regions.append(VisionRegion(name=name, center=tuple(float(value) for value in center), radius=tuple(float(max(value, 1.0e-9)) for value in radius), priority=float(priority)))  # Store the region without any mesh-size decision.
        return cls(regions, bbox)  # Return the cached one-shot semantic vision partition.
    def assign(self, mesh: Any) -> np.ndarray:  # Assign current element centroids to cached semantic regions.
        points = np.asarray(getattr(mesh, "points", getattr(mesh, "nodes", None)), dtype=float)  # Read current nodal coordinates.
        cells = np.asarray(getattr(mesh, "cells", getattr(mesh, "elements", None)), dtype=int)  # Read current element connectivity.
        if points.ndim != 2 or cells.ndim != 2:  # Reject unsupported mesh objects.
            raise ValueError("mesh must expose points and cells arrays")  # Explain the assignment contract.
        centroids = np.mean(points[cells, :3], axis=1)  # Compute current element centroids in physical coordinates.
        scores = np.full((cells.shape[0], len(self.regions)), np.inf, dtype=float)  # Allocate normalized region-distance scores.
        for index, region in enumerate(self.regions):  # Evaluate each cached semantic region.
            center = np.asarray(region.center, dtype=float)  # Read the region centre.
            radius = np.asarray(region.radius, dtype=float)  # Read anisotropic influence radii.
            normalized = np.linalg.norm((centroids - center) / radius, axis=1)  # Measure anisotropic normalized distance.
            scores[:, index] = normalized / max(region.priority, 1.0e-9)  # Resolve overlaps by distance and semantic priority.
        best = np.argmin(scores, axis=1)  # Identify the closest active semantic mechanism.
        best_score = scores[np.arange(cells.shape[0]), best]  # Read the winning normalized score.
        labels = np.where(best_score <= 1.0, best, len(self.regions)).astype(int)  # Assign uncovered elements to the generic field remainder.
        return labels  # Return one stable region index per current element.
    def to_dict(self) -> dict[str, Any]:  # Serialize the cached vision output without mesh parameters.
        return {"bbox": list(self.bbox), "field_name": self.field_name, "regions": [{"name": region.name, "center": list(region.center), "radius": list(region.radius), "priority": region.priority} for region in self.regions]}  # Return a JSON-compatible semantic partition.
    def save(self, path: str | Path) -> None:  # Persist the one-shot vision result for replay and audit.
        target = Path(path)  # Normalize the cache path.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create the cache directory when needed.
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")  # Write a human-auditable vision cache.
    @classmethod
    def load(cls, path: str | Path) -> "CachedVisionPartition":  # Restore a cached VLM or scripted-VLM region output.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Parse the cached semantic partition.
        regions = [VisionRegion(name=str(item["name"]), center=tuple(float(value) for value in item["center"]), radius=tuple(float(value) for value in item["radius"]), priority=float(item["priority"])) for item in payload["regions"]]  # Reconstruct immutable region contracts.
        return cls(regions, tuple(float(value) for value in payload["bbox"]), field_name=str(payload.get("field_name", "field_remainder")))  # Return the restored stable partition.
