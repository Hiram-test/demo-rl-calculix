"""Spatial image-conditioned AMR actions and visual-WM features; no general VLA claim."""  # Describe the actual implemented scope.
from __future__ import annotations  # Allow forward-compatible annotations.
from dataclasses import dataclass  # Store inspectable raster states.
from itertools import product  # Enumerate spatial blocks and simplex bounding boxes.
import numpy as np  # Implement deterministic numerical operations.
from scipy.ndimage import gaussian_filter, map_coordinates  # Apply genuine local image operations.
from scipy.sparse import csr_matrix  # Keep conservative cell-to-image transport sparse.
#
CHANNEL_NAMES = ("occupancy", "h", "eta2_density", "von_mises", "energy_density", "load", "constraint")  # Define the public channel order.
#
@dataclass  # Expose all physical scaling and raster correspondence information.
class VisualState:  # Represent one fixed-coordinate physical observation.
    image: np.ndarray  # Store seven channels followed by two or three spatial axes.
    channel_names: tuple[str, ...]  # Name every image channel explicitly.
    centers: np.ndarray  # Store flattened pixel or voxel centers in physical XYZ coordinates.
    cell_to_pixel: np.ndarray  # Map each cell centroid to its flattened raster location.
    pixel_to_cell: np.ndarray  # Store one contributing cell per pixel, or minus one outside material.
    cell_pixel_weights: csr_matrix  # Transport extensive cell quantities with column sums equal to one.
    material_measure: np.ndarray  # Store the integrated material area or volume deposited per pixel.
    bbox: tuple[float, ...]  # Preserve the fixed problem box as xmin,ymin,zmin,xmax,ymax,zmax.
    dim: int  # Distinguish physical triangles from tetrahedra.
    resolution: int  # Record the number of samples on each active axis.
    h_min: float  # Preserve the problem's minimum permitted mesh size.
    h0: float  # Preserve the problem's background mesh size.
    total_eta2: float  # Preserve the original solver-side total squared estimator.
    n_nodes: int  # Preserve the actual mesh node count.
    n_cells: int  # Preserve the actual mesh element count.
    pixel_measure: float  # Convert image densities back to integrated quantities.
    @property  # Retain the alternative total-error spelling for downstream callers.
    def eta2_total(self) -> float:  # Expose the same total without duplicating state.
        return self.total_eta2  # Return the independently retained original sum.
    @property  # Retain the explicit upper-bound spelling for downstream callers.
    def h_max(self) -> float:  # Expose the background size as the current upper bound.
        return self.h0  # Match existing size-field callers.
#
def _indices(points, bbox, dim, resolution):  # Convert physical points to clipped nearest raster bins.
    lower = np.asarray(bbox[:3])[:dim]  # Extract the physical box origin.
    width = np.asarray(bbox[3:])[:dim] - lower  # Extract active side lengths.
    index = np.floor((np.asarray(points)[:, :dim] - lower) / width * resolution).astype(int)  # Quantize without changing the physical box.
    return np.clip(index, 0, resolution - 1)  # Include boundary nodes in their nearest interior bin.
#
def _transport(mesh, bbox, resolution):  # Rasterize simplex interiors while retaining extensive-quantity conservation.
    dim, shape = mesh.dim, (resolution,) * mesh.dim  # Establish the active raster shape.
    lower, upper = np.asarray(bbox[:3])[:dim], np.asarray(bbox[3:])[:dim]  # Read physical box endpoints.
    spacing = (upper - lower) / resolution  # Keep voxel dimensions tied to geometry rather than mesh density.
    centers = np.zeros((resolution ** dim, 3))  # Retain XYZ coordinates for both dimensions.
    centers[:, :dim] = np.stack(np.meshgrid(*[lower[k] + (np.arange(resolution) + 0.5) * spacing[k] for k in range(dim)], indexing="ij"), axis=-1).reshape(-1, dim)  # Generate fixed pixel-center coordinates.
    rows, cols, weights = [], [], []  # Accumulate one sparse conservative scatter operator.
    pixel_to_cell = np.full(resolution ** dim, -1, dtype=int)  # Mark initially empty raster bins.
    cell_to_pixel = np.ravel_multi_index(_indices(mesh.centroids, bbox, dim, resolution).T, shape)  # Retain a direct centroid mapping.
    for cell, vertices in enumerate(mesh.nodes[mesh.cells, :dim]):  # Resolve each physical simplex independently.
        lo = np.maximum(np.ceil((vertices.min(axis=0) - lower) / spacing - 0.5).astype(int), 0)  # Find the first candidate pixel center.
        hi = np.minimum(np.floor((vertices.max(axis=0) - lower) / spacing - 0.5).astype(int), resolution - 1)  # Find the last candidate pixel center.
        candidate = np.asarray(list(product(*[range(lo[k], hi[k] + 1) for k in range(dim)])), dtype=int).reshape(-1, dim)  # Enumerate only the simplex's raster bounding box.
        flat = np.ravel_multi_index(candidate.T, shape) if len(candidate) else np.empty(0, dtype=int)  # Convert candidate bins to flat identifiers.
        if len(flat):  # Evaluate containment only when voxel centers can intersect the simplex.
            bary = (centers[flat, :dim] - vertices[0]) @ np.linalg.inv((vertices[1:] - vertices[0]).T).T  # Compute barycentric coordinates from the actual simplex.
            flat = flat[(bary.min(axis=1) >= -1e-10) & (bary.sum(axis=1) <= 1.0 + 1e-10)]  # Keep centers inside the physical cell.
        if not len(flat):  # Preserve cells smaller than one pixel instead of discarding their error.
            flat = np.asarray([cell_to_pixel[cell]])  # Deposit their extensive quantities in the quantized centroid bin.
        rows.extend(flat.tolist())  # Record every receiving pixel.
        cols.extend([cell] * len(flat))  # Record the source cell for each receiving pixel.
        weights.extend([1.0 / len(flat)] * len(flat))  # Normalize each cell's transported mass exactly.
        pixel_to_cell[flat] = cell  # Retain one contributing owner for inspection.
    transport = csr_matrix((weights, (rows, cols)), shape=(resolution ** dim, mesh.n_cells))  # Build the public sparse correspondence matrix.
    return centers, cell_to_pixel, pixel_to_cell, transport, float(np.prod(spacing))  # Return geometry and physical density scaling.
#
def rasterize(problem, post, eta2, resolution=32) -> VisualState:  # Convert current solved fields into a conservative fixed-coordinate observation.
    mesh, dim, resolution = post.mesh, post.mesh.dim, int(resolution)  # Read the actual mesh and raster dimension.
    eta2 = np.asarray(eta2, dtype=float)  # Preserve the supplied estimator values without ranking cells.
    centers, cell_to_pixel, pixel_to_cell, transport, measure = _transport(mesh, problem.bbox, resolution)  # Compute mesh-to-image transport.
    volume = np.asarray(transport @ mesh.measures).ravel()  # Integrate material measure in each pixel.
    safe_volume = np.maximum(volume, np.finfo(float).tiny)  # Avoid division by zero outside the object.
    image = np.zeros((len(CHANNEL_NAMES), resolution ** dim))  # Allocate all seven physical channels.
    image[0] = np.minimum(volume / measure, 1.0)  # Paint quantized material occupancy including thin cells.
    image[1] = (transport @ (mesh.cell_sizes * mesh.measures)) / safe_volume  # Paint volume-weighted current mesh size.
    image[2] = (transport @ eta2) / measure  # Preserve total estimator mass through density integration.
    image[3] = (transport @ (post.vm_elem * mesh.measures)) / safe_volume  # Paint volume-weighted von Mises stress.
    image[4] = (transport @ post.energy_elem) / measure  # Paint integrated strain energy per raster area or volume.
    for traction in problem.tractions:  # Obtain load locations from executable physical boundary conditions.
        selected = np.asarray(traction.facet_predicate(mesh.facet_centroids), dtype=bool)  # Select the actual loaded boundary facets.
        ids = np.ravel_multi_index(_indices(mesh.facet_centroids[selected], problem.bbox, dim, resolution).T, (resolution,) * dim)  # Quantize their physical centroids.
        intensity = np.linalg.norm(traction.value) * mesh.facet_measures[selected]  # Weight the load mark by the physical facet resultant scale.
        intensity *= float(problem.material.thickness) if dim == 2 else 1.0  # Include the plane-stress thickness used by the solver.
        np.add.at(image[5], ids, intensity)  # Aggregate collocated load contributions without feature-anchor heuristics.
    for constraint in problem.constraints:  # Obtain support locations from actual constrained nodes.
        selected = np.asarray(constraint.node_predicate(mesh.nodes), dtype=bool)  # Evaluate the solver's geometric constraint predicate.
        ids = np.ravel_multi_index(_indices(mesh.nodes[selected], problem.bbox, dim, resolution).T, (resolution,) * dim)  # Quantize support coordinates.
        np.maximum.at(image[6], ids, len(constraint.dofs) / dim)  # Paint the fraction of constrained physical translations.
    return VisualState(image.reshape((len(CHANNEL_NAMES),) + (resolution,) * dim), CHANNEL_NAMES, centers, cell_to_pixel, pixel_to_cell, transport, volume, tuple(problem.bbox), dim, resolution, float(problem.h_min), float(problem.h0), float(eta2.sum()), mesh.n_nodes, mesh.n_cells, measure)  # Keep all public metadata with the observation.
#
def _view(state, mode):  # Apply causal image ablations with an otherwise identical action architecture.
    image = np.asarray(state.image, dtype=float).copy()  # Avoid mutating the recorded physical observation.
    occupied = image[0].ravel() > 0  # Preserve the material support for every ablation.
    flat = image.reshape(len(CHANNEL_NAMES), -1)  # Address all spatial channels in the same coordinate order.
    if mode == "shuffled":  # Destroy spatial correspondence while preserving joint channel samples.
        order = np.random.default_rng(913).permutation(np.flatnonzero(occupied))  # Use a deterministic independent spatial permutation.
        flat[1:, occupied] = flat[1:, order]  # Move fields together while retaining the original object mask.
    elif mode == "constant":  # Remove spatial information while retaining one mean per channel.
        flat[1:, occupied] = flat[1:, occupied].mean(axis=1, keepdims=True)  # Broadcast global means through the same image-processing path.
    elif mode != "visual":  # Reject misspelled experimental conditions instead of silently changing the experiment.
        raise ValueError("mode must be visual, shuffled, or constant")  # Report the supported scientific ablation labels.
    return image  # Return the image used to generate actions.
#
def _smooth(field, occupancy, sigma):  # Perform normalized image convolution without averaging exterior zero values into material.
    support = (occupancy > 0).astype(float)  # Use material support as the convolution mask.
    numerator = gaussian_filter(np.asarray(field) * support, sigma=sigma, mode="nearest")  # Convolve local physical evidence.
    denominator = gaussian_filter(support, sigma=sigma, mode="nearest")  # Convolve the matching local support.
    return numerator / np.maximum(denominator, 1e-12)  # Return support-normalized spatial evidence.
#
def _sample(state, field, points):  # Interpolate a physical raster field at current mesh nodes.
    lower = np.asarray(state.bbox[:3])[:state.dim]  # Recover the fixed physical origin.
    width = np.asarray(state.bbox[3:])[:state.dim] - lower  # Recover active physical side lengths.
    coordinates = ((np.asarray(points)[:, :state.dim] - lower) / width * state.resolution - 0.5).T  # Convert node positions to image-index coordinates.
    support = state.image[0] > 0  # Preserve geometric support independently of the physics ablation.
    sampled = map_coordinates(np.asarray(field) * support, coordinates, order=1, mode="nearest")  # Sample the numerator using multilinear image interpolation.
    denominator = map_coordinates(support.astype(float), coordinates, order=1, mode="nearest")  # Sample the corresponding material support.
    fallback = float(np.median(np.asarray(field)[support]))  # Define a neutral interior value for unresolved boundary gaps.
    return np.where(denominator > 1e-12, sampled / np.maximum(denominator, 1e-12), fallback)  # Avoid artificial zero-size targets at object boundaries.
#
def _budget(mesh, raw, h_min, h0, growth):  # Normalize each spatial candidate to the same predicted isotropic cell-growth budget.
    raw = np.maximum(np.asarray(raw, dtype=float), 1e-12)  # Keep the scalar rescaling well defined.
    base = np.maximum(mesh.cell_sizes, 1e-12)  # Use actual current element sizes in the resource estimate.
    lo, hi = h_min / raw.max() * 0.5, h0 / raw.min() * 2.0  # Bracket every clipped target configuration.
    for _ in range(55):  # Resolve only a one-dimensional normalization of the visually selected shape.
        scale = np.sqrt(lo * hi)  # Search in mesh-size logarithms.
        target = np.clip(scale * raw, h_min, h0)  # Respect the same problem bounds for every candidate.
        ratio = np.mean((base / target[mesh.cells].mean(axis=1)) ** mesh.dim)  # Estimate new cells per old cell from local isotropic scaling.
        if ratio > growth:  # Reduce predicted resources by increasing the global mesh-size scale.
            lo = scale  # Move the lower bracket toward larger sizes.
        else:  # Increase predicted resources by reducing the global mesh-size scale.
            hi = scale  # Move the upper bracket toward smaller sizes.
    return np.clip(np.sqrt(lo * hi) * raw, h_min, h0)  # Return the original visual shape with budget normalization only.
#
def make_targets(state, mesh, growth=2.0, mode="visual", goal="energy") -> dict[str, np.ndarray]:  # Produce competing genuinely spatial nodal mesh-size actions.
    image = _view(state, mode)  # Apply the requested observation or image ablation before any decision.
    occupancy, current_h, eta_density, vm = image[0], image[1], image[2], image[3]  # Read exclusively image-derived local decision evidence.
    coefficient = eta_density / np.maximum(current_h, state.h_min) ** 2  # Remove the current linear-element h-squared bias before equidistribution.
    result = {}  # Collect multiple policies with comparable resource estimates.
    variants = (("err_s0", 0.0, 1.0), ("err_s1", 1.0, 1.0), ("err_s2", 2.0, 1.0), ("err_focused", 1.0, 1.35), ("err_diffuse", 1.0, 0.65))  # Span sharp, smoothed, focused, and diffuse visual allocations.
    for name, sigma, power in variants:  # Build each candidate through the same visual action architecture.
        density = _smooth(coefficient, occupancy, sigma)  # Gather neighboring image evidence at the selected scale.
        if goal in ("stress", "vm"):  # Let a requested stress objective alter where resources are allocated.
            density *= 0.2 + _smooth(vm, occupancy, sigma) / max(float(vm.max()), 1e-12)  # Weight the error image by the observed stress field.
        positive = density[occupancy > 0]  # Estimate a numerical floor only from material pixels.
        floor = max(float(positive.mean()) * 1e-3, 1e-30)  # Keep smooth or error-free regions representable with finite sizes.
        raw_image = np.maximum(density, floor) ** (-power / (state.dim + 2))  # Apply continuous error-resource equidistribution directly in image space.
        raw_nodes = _sample(state, raw_image, mesh.nodes)  # Convert the spatial image action to current mesh nodes.
        result[name] = _budget(mesh, raw_nodes, state.h_min, state.h0, float(growth))  # Normalize resource scale without replacing the image-selected support.
    stress = _smooth(vm ** 2, occupancy, 1.0)  # Add an independent stress-focused image alternative.
    stress_floor = max(float(stress[occupancy > 0].mean()) * 0.02, 1e-30)  # Prevent zero-stress pixels from becoming infinite sizes.
    stress_nodes = _sample(state, np.maximum(stress, stress_floor) ** (-1.0 / (state.dim + 2)), mesh.nodes)  # Convert stress image evidence to a nodal size proposal.
    result["vm_s1"] = _budget(mesh, stress_nodes, state.h_min, state.h0, float(growth))  # Keep the same resource estimator for the stress alternative.
    return result  # Return actions that retain causal dependence on image content.
#
def _pool(state, field, bins=4):  # Integrate spatial blocks in a fixed three-axis encoding.
    field = np.asarray(field).reshape((state.resolution,) * state.dim)  # Align the requested field with the observation grid.
    pooled = np.zeros((bins, bins, bins))  # Reserve the same feature dimension for 2D and 3D.
    groups = np.array_split(np.arange(state.resolution), bins)  # Partition physical image axes into equal-index blocks.
    for block in product(range(bins), repeat=state.dim):  # Preserve the location of each occupied spatial patch.
        destination = block if state.dim == 3 else block + (0,)  # Put 2D patches in the first Z plane and leave the rest zero.
        pooled[destination] = field[np.ix_(*[groups[k] for k in block])].sum()  # Integrate each patch without sorting away its position.
    return pooled.ravel()  # Expose one fixed-length vector in XYZ order.
#
def spatial_error(state, bins=4):  # Return absolute spatial estimator masses for visual-world-model transition targets.
    return _pool(state, state.image[2] * state.pixel_measure, bins=bins)  # Preserve the full original total across the spatial bins.
#
def features(state, target, mesh) -> np.ndarray:  # Encode a solved image jointly with a proposed spatial action.
    occupancy = state.image[0] > 0  # Retain the material mask for normalization.
    support_mass = _pool(state, occupancy.astype(float))  # Count samples in each physical patch.
    vectors, summaries = [], []  # Assemble fixed-length local and global descriptors.
    for index, channel in enumerate(state.image):  # Encode each available physical image channel.
        scale = state.h0 if index == 1 else max(float(np.abs(channel[occupancy]).mean()), 1e-30)  # Normalize physical magnitudes without losing spatial contrast.
        normalized = channel / scale  # Place heterogeneous field channels on comparable numerical scales.
        vectors.append(_pool(state, normalized) / np.maximum(support_mass, 1.0))  # Retain coarse patch contents rather than only scalar summaries.
        smooth = _smooth(normalized, occupancy, 1.0)  # Extract local convolution evidence at one-pixel scale.
        gradient = sum(component ** 2 for component in np.gradient(smooth)) ** 0.5  # Capture edges and field concentration in physical raster adjacency.
        summaries.extend([float(np.mean(normalized[occupancy] ** 2)), float(np.mean(gradient[occupancy])), float(np.max(gradient[occupancy]))])  # Retain compact spatial contrast statistics.
    node_ratio = np.asarray(target) / np.maximum(mesh.node_sizes, 1e-12)  # Record the actual proposed node-size changes.
    cell_ratio = np.asarray(target)[mesh.cells].mean(axis=1) / np.maximum(mesh.cell_sizes, 1e-12)  # Estimate action scale on actual solved cells.
    pixel_ratio = (state.cell_pixel_weights @ (mesh.measures * cell_ratio)) / np.maximum(state.material_measure, np.finfo(float).tiny)  # Transport the action into the same physical image frame.
    ratio_image = pixel_ratio.reshape(occupancy.shape)  # Recover spatial action geometry.
    vectors.append(_pool(state, ratio_image) / np.maximum(support_mass, 1.0))  # Preserve the action's regional size distribution.
    predicted_mass = state.image[2] * state.pixel_measure * ratio_image ** 2  # Couple image error and action using the linear-element local error law.
    vectors.append(_pool(state, predicted_mass) / max(state.total_eta2, 1e-30))  # Encode spatially resolved counterfactual error redistribution.
    summaries.extend(np.quantile(node_ratio, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist())  # Preserve actual action-strength quantiles.
    summaries.extend([state.dim, np.log1p(state.n_nodes), np.log1p(state.n_cells), np.log1p(state.total_eta2), float(np.mean(cell_ratio ** (-state.dim))), float(predicted_mass.sum() / max(state.total_eta2, 1e-30)), float(occupancy.mean())])  # Add physical dimension, resources, error scale, and predicted action effects.
    return np.nan_to_num(np.concatenate(vectors + [np.asarray(summaries)]), nan=0.0, posinf=1e12, neginf=-1e12)  # Return a finite fixed-length numerical encoder for a learned visual WM.
