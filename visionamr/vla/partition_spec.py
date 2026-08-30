"""Frozen per-geometry semantic partitions shared by world-model VLA and RL."""  # Describe the protocol-level fairness boundary implemented here.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from collections.abc import Mapping  # Import the read-only mapping contract used by JSON validation.
import hashlib  # Import SHA-256 for geometry, probe-mesh, and specification identities.
import json  # Import the transparent frozen-partition interchange format.
from pathlib import Path  # Import portable partition artifact paths.
import re  # Import strict lowercase SHA-256 syntax validation.
from typing import Any  # Import generic repository problem and mesh annotations.
import numpy as np  # Import numerical validation and fixed-graph operations.
from .regions import Partition, Seed  # Reuse the legacy regional feature and size-field implementation for RL only.
from .world.vision_partition import CachedVisionPartition, VisionRegion  # Reuse the new-stack semantic assignment contract exactly.

SPEC_SCHEMA = "wmvla.partition-spec.v1"  # Freeze the top-level partition specification schema.
PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every specification to the four-way frozen experiment.
ASSIGNMENT_RULE = "anisotropic-ellipsoid-centroid-min-score-with-background"  # Name the exact cached-vision element assignment rule.
ASSIGNMENT_VERSION = "visionamr.cached-vision-partition.assign.v1"  # Version the numerical assignment implementation independently.
BACKGROUND_RULE = "outside-all-active-ellipsoids"  # Define the generic remainder without a learned size prior.
ADJACENCY_RULE = "shared-face-cross-label-on-common-uniform-probe"  # Define one topology graph from the common probe mesh.
ADJACENCY_VERSION = "visionamr.partition-adjacency.v1"  # Version the fixed binary graph construction.
PROBE_SIZE_RULE = "global-constant-problem-h0"  # Prohibit semantic-region sizes in the common probe.
COMMON_NODAL_GRADATION = 1.0  # Freeze the PR-40 V0 nodal size-field gradation shared by WM, RL, LP, supervised, and Dörfler.
FACTORY_PATH = "visionamr.bridge_cases.make_box_girder_diaphragm"  # Bind the main protocol implementation to the canonical bridge factory.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")  # Accept only complete lowercase SHA-256 strings.

def _canonical_json_bytes(payload: object) -> bytes:  # Serialize one JSON-compatible value for stable semantic hashing.
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Remove formatting, locale, key-order, and NaN ambiguity.

def _sha256_payload(payload: object) -> str:  # Hash one canonical JSON-compatible value.
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()  # Return the complete lowercase hexadecimal digest.

def _canonical_value(value: Any) -> Any:  # Normalize repository metadata to deterministic JSON primitives.
    if isinstance(value, Mapping):  # Normalize mappings in sorted string-key order.
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=lambda item: str(item))}  # Recursively normalize every mapping value.
    if isinstance(value, (list, tuple)):  # Normalize ordered containers without losing their order.
        return [_canonical_value(item) for item in value]  # Recursively normalize every sequence entry.
    if isinstance(value, np.ndarray):  # Normalize numerical arrays to ordered JSON lists.
        return _canonical_value(value.tolist())  # Reuse recursive scalar normalization after conversion.
    if isinstance(value, (np.integer, np.floating)):  # Normalize NumPy scalar wrappers.
        return value.item()  # Return the corresponding built-in numerical scalar.
    if isinstance(value, (str, int, float, bool)) or value is None:  # Admit deterministic JSON scalar values.
        return value  # Preserve the scalar exactly.
    return repr(value)  # Preserve otherwise unsupported metadata through an explicit stable textual representation.

def problem_geometry_fingerprint(problem: Any) -> str:  # Bind a loaded specification to the exact geometry metadata without load magnitude.
    parameters = {str(key): value for key, value in dict(getattr(problem, "params", {})).items() if str(key) != "pressure"}  # Exclude only the non-geometric pressure parameter.
    features = [{"name": str(getattr(feature, "name", "feature")), "kind": str(getattr(feature, "kind", "feature")), "xyz": [float(feature.x), float(feature.y), float(getattr(feature, "z", 0.0))], "radius": float(getattr(feature, "r", 0.0))} for feature in getattr(problem, "features", [])]  # Preserve ordered semantic geometry anchors.
    payload = {"family": str(getattr(problem, "name", "unknown")), "dimension": int(getattr(problem, "dim", 0)), "bbox": [float(value) for value in getattr(problem, "bbox", ())], "h0": float(getattr(problem, "h0", 0.0)), "h_ref": float(getattr(problem, "h_ref", 0.0)), "h_min": float(getattr(problem, "h_min", 0.0)), "parameters_without_pressure": _canonical_value(parameters), "features": features, "singular_points": _canonical_value(getattr(problem, "singular_points", [])), "singular_segments": _canonical_value(getattr(problem, "singular_segments", []))}  # Assemble all assignment-relevant geometry metadata.
    return _sha256_payload(payload)  # Return a collision-resistant local geometry identity.

def probe_mesh_sha256(mesh: Any) -> str:  # Hash the exact common uniform probe nodes and connectivity.
    points = np.asarray(getattr(mesh, "points", getattr(mesh, "nodes", None)), dtype="<f8")  # Normalize coordinates to portable little-endian float64.
    cells = np.asarray(getattr(mesh, "cells", getattr(mesh, "elements", None)), dtype="<i8")  # Normalize connectivity to portable little-endian int64.
    if points.ndim != 2 or cells.ndim != 2:  # Reject unsupported probe mesh objects.
        raise ValueError("probe mesh must expose two-dimensional points and cells arrays")  # Explain the probe contract.
    digest = hashlib.sha256()  # Initialize the full probe-mesh digest.
    digest.update(_canonical_json_bytes({"points_shape": list(points.shape), "cells_shape": list(cells.shape)}))  # Bind array dimensions before raw data.
    digest.update(np.ascontiguousarray(points).tobytes())  # Bind every coordinate in canonical storage order.
    digest.update(np.ascontiguousarray(cells).tobytes())  # Bind every connectivity index in canonical storage order.
    return digest.hexdigest()  # Return the complete probe-mesh SHA-256.

def _fixed_face_adjacency(mesh: Any, labels: np.ndarray, count: int) -> np.ndarray:  # Build one binary region graph from face-adjacent probe elements.
    if hasattr(mesh, "cell_adjacency"):  # Prefer the repository's exact simplex face-adjacency implementation.
        pairs = np.asarray(mesh.cell_adjacency[0], dtype=int)  # Read the deterministic cell-pair list.
    else:  # Support simple mesh fixtures through explicit shared-face reconstruction.
        cells = np.asarray(getattr(mesh, "cells", getattr(mesh, "elements", None)), dtype=int)  # Read simplex connectivity from the fixture.
        if cells.ndim != 2 or cells.shape[1] < 3:  # Reject malformed or non-simplex fixtures.
            raise ValueError("probe mesh cells must be a two-dimensional simplex array")  # Explain the fallback adjacency contract.
        facets = [np.delete(cells, local, axis=1) for local in range(cells.shape[1])]  # Enumerate every codimension-one simplex face.
        stacked = np.sort(np.vstack(facets), axis=1)  # Canonicalize local face-node order.
        owners = np.tile(np.arange(cells.shape[0], dtype=int), cells.shape[1])  # Associate each stacked face with its owning cell.
        order = np.lexsort(stacked.T[::-1])  # Sort faces lexicographically for duplicate detection.
        sorted_facets = stacked[order]  # Apply the canonical face ordering.
        sorted_owners = owners[order]  # Keep owners aligned with their faces.
        shared = np.all(sorted_facets[1:] == sorted_facets[:-1], axis=1)  # Locate faces shared by two simplex cells.
        pairs = np.column_stack((sorted_owners[:-1][shared], sorted_owners[1:][shared]))  # Recover adjacent owner pairs.
    graph = np.zeros((count, count), dtype=np.int8)  # Allocate the symmetric binary regional graph.
    if pairs.size:  # Process the graph only when the probe contains adjacent cells.
        left = labels[pairs[:, 0]]  # Read labels on the first side of every shared face.
        right = labels[pairs[:, 1]]  # Read labels on the second side of every shared face.
        crossing = left != right  # Retain only cross-region interfaces.
        graph[left[crossing], right[crossing]] = 1  # Record forward regional adjacency.
        graph[right[crossing], left[crossing]] = 1  # Record reverse regional adjacency.
    np.fill_diagonal(graph, 0)  # Forbid self-edges explicitly.
    return graph  # Return the fixed graph in semantic region order.

def _row_normalized(graph: np.ndarray) -> np.ndarray:  # Convert a binary graph into the new world model's coupling convention.
    values = np.asarray(graph, dtype=float)  # Copy the graph into floating-point storage.
    totals = np.sum(values, axis=1, keepdims=True)  # Measure each region's fixed neighbor count.
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0.0)  # Preserve isolated regions while normalizing connected rows.

class FrozenPartitionSpec(CachedVisionPartition):  # Extend the new-stack partition with complete frozen protocol evidence.
    def __init__(self, regions: list[VisionRegion], bbox: tuple[float, ...], *, field_name: str, geometry_hash: str, geometry_fingerprint: str, probe_sha256: str, probe_node_count: int, probe_cell_count: int, probe_uniform_size: float, fixed_adjacency: np.ndarray) -> None:  # Initialize one per-geometry immutable semantic specification.
        super().__init__(regions, bbox, field_name=field_name)  # Reuse the exact new-stack assignment implementation.
        self.geometry_hash = str(geometry_hash)  # Store the manifest geometry identity.
        self.geometry_fingerprint = str(geometry_fingerprint)  # Store the independently reproducible problem fingerprint.
        self.probe_sha256 = str(probe_sha256)  # Store the exact common-probe mesh identity.
        self.probe_node_count = int(probe_node_count)  # Store the common-probe node count.
        self.probe_cell_count = int(probe_cell_count)  # Store the common-probe element count.
        self.probe_uniform_size = float(probe_uniform_size)  # Store the sole size used to create the common probe.
        self.fixed_adjacency = np.asarray(fixed_adjacency, dtype=np.int8).copy()  # Freeze the binary graph independently from later remeshes.
        self._validate_invariants()  # Reject malformed in-memory specifications before hashing or use.
        self.spec_sha256 = _sha256_payload(self._body())  # Hash the complete semantic body without recursive self-reference.
    def _validate_invariants(self) -> None:  # Validate geometry, assignment, probe, and graph invariants.
        if _SHA256_RE.fullmatch(self.geometry_hash) is None or _SHA256_RE.fullmatch(self.geometry_fingerprint) is None or _SHA256_RE.fullmatch(self.probe_sha256) is None:  # Require complete lowercase identities.
            raise ValueError("geometry_hash, geometry_fingerprint, and probe_sha256 must be lowercase SHA-256 values")  # Explain the identity contract.
        if len(self.bbox) != 6 or not np.all(np.isfinite(np.asarray(self.bbox, dtype=float))):  # Require a finite three-dimensional bridge bounding box.
            raise ValueError("partition bbox must contain six finite values")  # Explain the three-dimensional geometry contract.
        if not np.all(np.asarray(self.bbox[3:], dtype=float) > np.asarray(self.bbox[:3], dtype=float)):  # Require strictly positive physical extents.
            raise ValueError("partition bbox upper bounds must exceed lower bounds")  # Explain the invalid geometry envelope.
        if len(set(self.names)) != len(self.names):  # Require unique semantic identifiers including the background.
            raise ValueError("partition region names must be unique")  # Reject action-vector ambiguity.
        if any(not np.all(np.isfinite(np.asarray(region.center, dtype=float))) or not np.all(np.asarray(region.radius, dtype=float) > 0.0) or not np.isfinite(region.priority) or region.priority <= 0.0 for region in self.regions):  # Validate every ellipsoid definition.
            raise ValueError("partition regions require finite centers and positive radii and priorities")  # Explain invalid semantic geometry.
        count = len(self.names)  # Read the complete semantic-region count once.
        if self.fixed_adjacency.shape != (count, count):  # Require exact graph ordering and dimensions.
            raise ValueError("fixed adjacency shape must match partition names")  # Explain the graph-order contract.
        if np.any((self.fixed_adjacency != 0) & (self.fixed_adjacency != 1)) or not np.array_equal(self.fixed_adjacency, self.fixed_adjacency.T) or np.any(np.diag(self.fixed_adjacency) != 0):  # Require a simple undirected binary graph.
            raise ValueError("fixed adjacency must be symmetric, binary, and diagonal-free")  # Explain the fixed topology contract.
        if self.probe_node_count <= 0 or self.probe_cell_count <= 0 or not np.isfinite(self.probe_uniform_size) or self.probe_uniform_size <= 0.0:  # Require auditable positive probe metadata.
            raise ValueError("probe counts and uniform size must be positive")  # Explain the common-probe contract.
    def _body(self) -> dict[str, Any]:  # Serialize all hash-covered specification fields.
        regions = [{"name": region.name, "center": [float(value) for value in region.center], "radius": [float(value) for value in region.radius], "priority": float(region.priority)} for region in self.regions]  # Preserve semantic ellipsoids in frozen action-vector order.
        return {"schema": SPEC_SCHEMA, "protocol_id": PROTOCOL_ID, "factory": FACTORY_PATH, "geometry_hash": self.geometry_hash, "problem_geometry_fingerprint": self.geometry_fingerprint, "bbox": [float(value) for value in self.bbox], "region_order": list(self.names), "regions": regions, "assignment": {"rule": ASSIGNMENT_RULE, "version": ASSIGNMENT_VERSION, "centroid": "arithmetic-mean-of-element-nodes", "score": "l2((centroid-center)/radius)/priority", "overlap": "minimum-score-then-region-order", "active_threshold": 1.0}, "background": {"name": self.field_name, "rule": BACKGROUND_RULE, "mesh_size_prior": None}, "probe": {"size_rule": PROBE_SIZE_RULE, "uniform_size": self.probe_uniform_size, "semantic_region_sizes_forbidden": True, "mesh_sha256": self.probe_sha256, "node_count": self.probe_node_count, "cell_count": self.probe_cell_count}, "size_field": {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}, "adjacency": {"rule": ADJACENCY_RULE, "version": ADJACENCY_VERSION, "matrix": self.fixed_adjacency.astype(int).tolist()}}  # Return one transparent, deterministic schema body.
    def to_dict(self) -> dict[str, Any]:  # Serialize the specification including its semantic-body digest.
        payload = self._body()  # Build the complete hash-covered body.
        payload["spec_sha256"] = self.spec_sha256  # Attach the non-recursive integrity digest.
        return payload  # Return the complete persisted representation.
    def save(self, path: str | Path) -> None:  # Persist the unique per-geometry partition specification.
        target = Path(path)  # Normalize the output artifact path.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create the per-case partition directory when needed.
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Write one deterministic, human-auditable JSON document.
    @classmethod  # Restore specifications through the validated constructor.
    def load(cls, path: str | Path, *, expected_sha256: str | None = None, problem: Any | None = None, expected_geometry_hash: str | None = None) -> "FrozenPartitionSpec":  # Load and verify one persisted per-geometry specification.
        encoded = Path(path).read_bytes()  # Read exact persisted bytes before decoding or semantic validation.
        file_sha256 = hashlib.sha256(encoded).hexdigest()  # Compute the conventional artifact digest used by freeze records.
        payload = json.loads(encoded.decode("utf-8"))  # Parse the transparent UTF-8 specification.
        if not isinstance(payload, Mapping):  # Require one top-level JSON object.
            raise ValueError("partition specification root must be an object")  # Reject structurally invalid artifacts.
        if payload.get("schema") != SPEC_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID or payload.get("factory") != FACTORY_PATH:  # Require exact schema, protocol, and factory identities.
            raise ValueError("partition specification schema, protocol, or factory is not frozen WMVLA-4WAY-P1")  # Reject unrelated or stale specifications.
        assignment = payload.get("assignment", {})  # Read the assignment contract for exact validation.
        background = payload.get("background", {})  # Read the background contract for exact validation.
        probe = payload.get("probe", {})  # Read the common-probe evidence for exact validation.
        size_field = payload.get("size_field", {})  # Read the shared nodal-field gradation contract for exact validation.
        adjacency = payload.get("adjacency", {})  # Read the fixed graph contract for exact validation.
        if assignment.get("rule") != ASSIGNMENT_RULE or assignment.get("version") != ASSIGNMENT_VERSION or assignment.get("active_threshold") != 1.0:  # Forbid assignment-rule drift.
            raise ValueError("partition assignment contract does not match the frozen implementation")  # Explain the incompatible semantic definition.
        if background.get("rule") != BACKGROUND_RULE or background.get("mesh_size_prior", "missing") is not None:  # Require an explicit prior-free background.
            raise ValueError("partition background contract is not the frozen prior-free remainder")  # Reject hidden background sizing.
        if probe.get("size_rule") != PROBE_SIZE_RULE or probe.get("semantic_region_sizes_forbidden") is not True:  # Require a uniformly sized semantic-prior-free probe.
            raise ValueError("partition probe is not declared as global uniform problem.h0")  # Reject hidden visual size priors.
        if float(size_field.get("nodal_gradation", float("nan"))) != COMMON_NODAL_GRADATION or size_field.get("source") != "PR40_V0_default_tool_behavior":  # Require the exact common V0 size-field smoothing behavior.
            raise ValueError("partition nodal gradation does not match the frozen PR-40 V0 contract")  # Reject method-specific or post-result gradation drift.
        if adjacency.get("rule") != ADJACENCY_RULE or adjacency.get("version") != ADJACENCY_VERSION:  # Require the fixed shared-face graph contract.
            raise ValueError("partition adjacency contract does not match the frozen implementation")  # Reject graph construction drift.
        region_values = payload.get("regions", [])  # Read ordered active semantic region definitions.
        if not isinstance(region_values, list) or not region_values:  # Require at least one active semantic mechanism.
            raise ValueError("partition specification requires active regions")  # Reject an empty visual partition.
        regions = [VisionRegion(name=str(item["name"]), center=tuple(float(value) for value in item["center"]), radius=tuple(float(value) for value in item["radius"]), priority=float(item["priority"])) for item in region_values]  # Reconstruct immutable cached-vision regions in persisted order.
        instance = cls(regions, tuple(float(value) for value in payload["bbox"]), field_name=str(background["name"]), geometry_hash=str(payload["geometry_hash"]), geometry_fingerprint=str(payload["problem_geometry_fingerprint"]), probe_sha256=str(probe["mesh_sha256"]), probe_node_count=int(probe["node_count"]), probe_cell_count=int(probe["cell_count"]), probe_uniform_size=float(probe["uniform_size"]), fixed_adjacency=np.asarray(adjacency["matrix"], dtype=np.int8))  # Reconstruct and validate the complete in-memory specification.
        if list(instance.names) != payload.get("region_order"):  # Require region definitions and action-vector order to agree exactly.
            raise ValueError("partition region_order does not match region definitions and background")  # Reject reordered or duplicated action semantics.
        recorded_sha = str(payload.get("spec_sha256", ""))  # Read the persisted semantic-body digest.
        if recorded_sha != instance.spec_sha256:  # Recompute integrity independently after reconstruction.
            raise ValueError("partition spec_sha256 does not match its canonical semantic body")  # Reject any tampering or partial update.
        if expected_sha256 is not None and str(expected_sha256) not in (instance.spec_sha256, file_sha256):  # Accept either the embedded semantic digest or conventional exact-file digest.
            raise ValueError("partition specification does not match the expected frozen SHA-256")  # Reject the wrong frozen partition artifact.
        instance.file_sha256 = file_sha256  # Expose the exact loaded artifact digest for freeze-record reporting.
        instance.verify(problem=problem, expected_geometry_hash=expected_geometry_hash)  # Bind the specification to optional runtime case evidence.
        return instance  # Return the fully validated new-stack partition object.
    def verify(self, *, problem: Any | None = None, expected_geometry_hash: str | None = None, probe_mesh: Any | None = None) -> None:  # Verify runtime case and probe identities without mutating the specification.
        if expected_geometry_hash is not None and self.geometry_hash != str(expected_geometry_hash):  # Compare against the case manifest's geometry identity.
            raise ValueError("partition geometry_hash does not match the requested case")  # Reject cross-case partition reuse.
        if problem is not None and problem_geometry_fingerprint(problem) != self.geometry_fingerprint:  # Recompute assignment-relevant geometry metadata.
            raise ValueError("partition problem_geometry_fingerprint does not match the runtime problem")  # Reject geometry or feature drift.
        if problem is not None and not np.isclose(float(problem.h0), self.probe_uniform_size, rtol=0.0, atol=1.0e-12):  # Require the exact global uniform probe size.
            raise ValueError("partition probe uniform size does not match problem.h0")  # Reject common-probe drift.
        if probe_mesh is not None and probe_mesh_sha256(probe_mesh) != self.probe_sha256:  # Compare the exact regenerated common probe when available.
            raise ValueError("partition probe mesh hash does not match the runtime common probe")  # Reject mesh-version or geometry drift.
    def adjacency_matrix(self, mesh: Any | None = None, labels: np.ndarray | None = None) -> np.ndarray:  # Expose the frozen row-normalized graph to the new world-model gateway.
        del mesh, labels  # Ignore later remesh topology and labels by protocol design.
        return _row_normalized(self.fixed_adjacency)  # Return the immutable probe-derived coupling graph.
    def as_world_partition(self, problem: Any, *, expected_geometry_hash: str | None = None) -> "FrozenPartitionSpec":  # Return the direct new-stack partition after runtime binding checks.
        self.verify(problem=problem, expected_geometry_hash=expected_geometry_hash)  # Verify the case before any world-model solve.
        return self  # Reuse this exact CachedVisionPartition subclass without reconstruction.
    def as_rl_partition(self, problem: Any, *, probe_mesh: Any | None = None, expected_geometry_hash: str | None = None) -> "FrozenRLPartition":  # Build the legacy RL thin adapter with the same labels and graph.
        self.verify(problem=problem, expected_geometry_hash=expected_geometry_hash, probe_mesh=probe_mesh)  # Verify geometry and the regenerated common probe.
        return FrozenRLPartition(self, problem, gradation=COMMON_NODAL_GRADATION)  # Initialize all regional mesh sizes uniformly at problem.h0 with the explicit common V0 gradation.

class FrozenRLPartition(Partition):  # Adapt one frozen semantic spec to the existing regional Double-DQN environment.
    def __init__(self, spec: FrozenPartitionSpec, problem: Any, sizes: np.ndarray | None = None, gradation: float = COMMON_NODAL_GRADATION) -> None:  # Initialize same-label RL state with the explicit common PR-40 V0 nodal gradation.
        self.spec = spec  # Retain the exact shared assignment and graph source.
        if float(gradation) != COMMON_NODAL_GRADATION:  # Forbid an RL-only smoothing choice under the shared finite-element contract.
            raise ValueError("RL partition nodal gradation must equal the frozen common value 1.0")  # Reject hidden method-specific gradation drift.
        centers = [tuple(float(value) for value in region.center) for region in spec.regions] + [tuple(float((spec.bbox[index] + spec.bbox[index + 3]) * 0.5) for index in range(3))]  # Give the background a neutral bounding-box center used only by legacy feature contracts.
        values = np.full(len(spec.names), float(problem.h0), dtype=float) if sizes is None else np.asarray(sizes, dtype=float).copy()  # Use one common initial size or preserve explicit RL actions.
        if values.shape != (len(spec.names),) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):  # Validate action-vector-aligned positive sizes.
            raise ValueError("RL partition sizes must be positive and match frozen region_order")  # Explain the RL adapter contract.
        seeds = [Seed(name=name, xyz=center, h=float(value), origin="coarse" if name == spec.field_name else "vision") for name, center, value in zip(spec.names, centers, values, strict=True)]  # Build legacy size carriers in exact shared order.
        super().__init__(seeds=seeds, problem=problem, gradation=float(gradation), assign_mode="geodesic", drawings=[])  # Reuse legacy feature and nodal-field methods while overriding assignment.
    @property  # Expose exact semantic names independently from legacy seeds.
    def names(self) -> tuple[str, ...]:  # Return action-vector names in frozen order.
        return self.spec.names  # Delegate to the single persisted source.
    def assign(self, mesh: Any) -> np.ndarray:  # Assign every remesh through the exact new-stack cached semantic rule.
        return self.spec.assign(mesh)  # Prevent drawn or geodesic substitution.
    def adjacency_matrix(self, mesh: Any | None = None, labels: np.ndarray | None = None) -> np.ndarray:  # Expose the fixed binary graph expected by the existing DQN normalizer.
        del mesh, labels  # Ignore later remesh topology and labels by protocol design.
        return self.spec.fixed_adjacency.astype(float).copy()  # Return an independent binary graph in frozen order.
    def with_sizes(self, sizes: np.ndarray) -> "FrozenRLPartition":  # Apply an RL refine action without changing assignment or graph.
        return FrozenRLPartition(self.spec, self.problem, np.asarray(sizes, dtype=float), self.gradation)  # Rebuild only legacy size carriers.

class SharedPartitioner:  # Supply one frozen specification to the new world stack and the existing RL environment.
    def __init__(self, spec: FrozenPartitionSpec, *, expected_geometry_hash: str | None = None) -> None:  # Initialize a case-bound partition provider.
        self.spec = spec  # Retain the unique loaded per-geometry specification.
        self.expected_geometry_hash = expected_geometry_hash  # Retain the optional manifest identity for every use.
    def partition_for_world(self, problem: Any, probe_mesh: Any | None = None) -> FrozenPartitionSpec:  # Return the direct new-stack partition object.
        self.spec.verify(problem=problem, expected_geometry_hash=self.expected_geometry_hash, probe_mesh=probe_mesh)  # Bind world execution to case and optional probe evidence.
        return self.spec  # Return the exact shared object.
    def partition_for_rl(self, problem: Any, probe_mesh: Any | None = None) -> FrozenRLPartition:  # Return the thin existing-RL adapter.
        return self.spec.as_rl_partition(problem, probe_mesh=probe_mesh, expected_geometry_hash=self.expected_geometry_hash)  # Preserve labels, order, and fixed graph exactly.

class PartitionSpecRegistry:  # Resolve deterministic per-case partition artifacts for the benchmark harness.
    def __init__(self, root: str | Path, *, expected_sha256: Mapping[str, str] | None = None) -> None:  # Initialize a read-only per-case specification registry.
        self.root = Path(root)  # Normalize the configured protocol partition root.
        self.expected_sha256 = {} if expected_sha256 is None else {str(key): str(value) for key, value in expected_sha256.items()}  # Copy optional freeze-record digests by case ID.
    def path_for(self, case_id: str) -> Path:  # Resolve the unique artifact path for one case.
        identifier = str(case_id)  # Normalize the manifest case identifier.
        if not identifier or Path(identifier).name != identifier or identifier in (".", ".."):  # Forbid path traversal and empty identifiers.
            raise ValueError("case_id must be one safe path component")  # Explain the registry path contract.
        return self.root / identifier / "partition_spec.json"  # Return the protocol-required per-instance filename.
    def partitioner_for(self, case_id: str, problem: Any, geometry_hash: str) -> SharedPartitioner:  # Load one case and return a shared WM/RL provider.
        spec = load_partition_spec(self.path_for(case_id), expected_sha256=self.expected_sha256.get(str(case_id)), problem=problem, expected_geometry_hash=geometry_hash)  # Perform all integrity and geometry checks before use.
        return SharedPartitioner(spec, expected_geometry_hash=geometry_hash)  # Return one case-bound provider for both methods.
    def partition_for(self, case_id: str, problem: Any, geometry_hash: str) -> FrozenPartitionSpec:  # Load one case for direct injection into the new world-model pipeline.
        shared = self.partitioner_for(case_id, problem, geometry_hash)  # Reuse the single complete registry validation path.
        return shared.partition_for_world(problem)  # Return a CachedVisionPartition-compatible object with the frozen graph.

def build_partition_spec(problem: Any, geometry_hash: str, probe_mesh: Any) -> FrozenPartitionSpec:  # Build one specification from the exact common uniform probe without solving.
    base = CachedVisionPartition.from_problem(problem)  # Generate semantic regions once from geometry and boundary-condition anchors only.
    labels = np.asarray(base.assign(probe_mesh), dtype=int).reshape(-1)  # Assign the common-probe elements using the same new-stack rule.
    cells = np.asarray(getattr(probe_mesh, "cells", getattr(probe_mesh, "elements", None)), dtype=int)  # Read the probe element count independently.
    points = np.asarray(getattr(probe_mesh, "points", getattr(probe_mesh, "nodes", None)), dtype=float)  # Read the probe node count independently.
    if labels.shape != (cells.shape[0],) or np.any(labels < 0) or np.any(labels >= len(base.names)):  # Require one valid label per probe element.
        raise ValueError("cached partition produced invalid common-probe labels")  # Reject an unusable semantic specification.
    graph = _fixed_face_adjacency(probe_mesh, labels, len(base.names))  # Freeze adjacency exactly once on the common uniform probe.
    return FrozenPartitionSpec(list(base.regions), base.bbox, field_name=base.field_name, geometry_hash=str(geometry_hash), geometry_fingerprint=problem_geometry_fingerprint(problem), probe_sha256=probe_mesh_sha256(probe_mesh), probe_node_count=int(points.shape[0]), probe_cell_count=int(cells.shape[0]), probe_uniform_size=float(problem.h0), fixed_adjacency=graph)  # Return the complete hash-ready specification.

def generate_partition_spec(problem: Any, geometry_hash: str) -> FrozenPartitionSpec:  # Generate the common uniform probe and freeze one per-geometry specification.
    from ..experiment import initial_mesh  # Import the canonical common-probe generator lazily to avoid native initialization during module import.
    probe_mesh = initial_mesh(problem)  # Generate exactly one global problem.h0 probe with no semantic size field.
    return build_partition_spec(problem, geometry_hash, probe_mesh)  # Freeze assignment and adjacency from that single unsolved mesh.

def load_partition_spec(path: str | Path, *, expected_sha256: str | None = None, problem: Any | None = None, expected_geometry_hash: str | None = None) -> FrozenPartitionSpec:  # Provide the benchmark harness's stable loading interface.
    return FrozenPartitionSpec.load(path, expected_sha256=expected_sha256, problem=problem, expected_geometry_hash=expected_geometry_hash)  # Delegate to the complete schema and integrity validator.
