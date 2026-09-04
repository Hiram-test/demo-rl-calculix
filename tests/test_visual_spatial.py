"""Causal spatial-image checks using actual simplex geometry without a solver substitute."""  # Describe the tests' limited purpose.
from types import SimpleNamespace  # Construct compact physical fixtures.
import numpy as np  # Evaluate geometry and numerical invariants.
from visionamr.mesher import Mesh  # Use the repository's real mesh representation.
from visionamr.vla.visual_spatial import rasterize, make_targets, features, spatial_error  # Exercise the public visual-WM interfaces.
#
def _fixture(dim=2):  # Build a small physical mesh and fields for deterministic causal tests.
    if dim == 2:  # Use a connected triangular square with enough spatial detail for translated hotspots.
        grid = np.linspace(0.0, 1.0, 9)  # Define a regular physical coordinate grid.
        xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)  # Arrange Cartesian nodes in a stable order.
        nodes = np.column_stack((xy, np.zeros(len(xy))))  # Match the repository's XYZ coordinate convention.
        cells = np.asarray([[i * 9 + j, (i + 1) * 9 + j, i * 9 + j + 1] for i in range(8) for j in range(8)] + [[(i + 1) * 9 + j, (i + 1) * 9 + j + 1, i * 9 + j + 1] for i in range(8) for j in range(8)])  # Triangulate the square without hidden remeshing.
    else:  # Use a true tetrahedron to verify three-dimensional transport and padded feature shape.
        nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # Define a nondegenerate tetrahedron in the unit box.
        cells = np.asarray([[0, 1, 2, 3]])  # Preserve the actual tetrahedral connectivity.
    mesh = Mesh(nodes, cells, dim)  # Instantiate the production geometry interface.
    constraint = SimpleNamespace(node_predicate=lambda xyz: xyz[:, 0] < 1e-9, dofs=tuple(range(1, dim + 1)))  # Define a physical fixed boundary by its solver predicate.
    traction = SimpleNamespace(facet_predicate=lambda xyz: xyz[:, 0] > 0.99, value=(1.0, 0.0, 0.0))  # Define a physical unit traction on the opposite boundary.
    problem = SimpleNamespace(dim=dim, bbox=(0, 0, 0, 1, 1, 1 if dim == 3 else 0), h_min=0.012, h0=2.0, constraints=[constraint], tractions=[traction], material=SimpleNamespace(thickness=1.0))  # Supply only real rasterization inputs.
    post = SimpleNamespace(mesh=mesh, vm_elem=np.ones(mesh.n_cells), energy_elem=mesh.measures.copy())  # Keep non-error fields fixed in the causal experiment.
    return problem, post  # Return the reusable physical fixture.
#
def test_conservative_raster_and_spatial_targets():  # Check physical error mass and field metadata in both supported dimensions.
    for dim in (2, 3):  # Apply identical invariants to pixels and voxels.
        problem, post = _fixture(dim)  # Obtain a real simplex geometry fixture.
        eta = np.linspace(0.1, 2.0, post.mesh.n_cells)  # Define independent extensive error contributions.
        state = rasterize(problem, post, eta, resolution=16)  # Rasterize through the public path.
        assert state.image.shape == (7,) + (16,) * dim  # Confirm the physical dimensional layout.
        assert np.allclose(np.asarray(state.cell_pixel_weights.sum(axis=0)), 1.0)  # Every cell conserves its deposited extensive quantities.
        assert np.isclose(np.sum(state.image[2]) * state.pixel_measure, eta.sum())  # Raster integration must reproduce the supplied solver total.
        assert np.isclose(spatial_error(state).sum(), eta.sum())  # Spatial WM targets must conserve that same total.
        assert spatial_error(state).shape == (64,)  # Preserve one fixed 2D/3D learning interface.
        assert np.any(state.image[6] > 0)  # Confirm the actual support predicate is represented.
        action = make_targets(state, post.mesh)["err_s1"]  # Generate a bounded visual size field.
        assert np.all((action >= problem.h_min) & (action <= problem.h0))  # Verify physical size limits independently of raster dimension.
        assert features(state, action, post.mesh).shape == (611,)  # Keep the joint state-action encoder fixed across dimensions.
#
def test_translated_hotspot_moves_action_and_constant_image_erases_it():  # Demonstrate that image position causally controls the generated mesh action.
    problem, post = _fixture()  # Keep the same physical mesh and all scalar resource metadata.
    x, y = post.mesh.centroids[:, 0], post.mesh.centroids[:, 1]  # Read physical cell locations.
    eta_left = post.mesh.measures * (0.01 + np.exp(-((x - 0.22) ** 2 + (y - 0.5) ** 2) / 0.01))  # Place an error hotspot on the left side.
    eta_right = post.mesh.measures * (0.01 + np.exp(-((x - 0.78) ** 2 + (y - 0.5) ** 2) / 0.01))  # Translate the same hotspot to the right side.
    left, right = [rasterize(problem, post, eta, resolution=24) for eta in (eta_left, eta_right)]  # Produce images with matched global error but different locations.
    assert np.isclose(left.total_eta2, right.total_eta2)  # Rule out a total-error explanation for the changed action.
    action_left, action_right = [make_targets(state, post.mesh)["err_s1"] for state in (left, right)]  # Generate actions without reading raw element-error arrays.
    left_nodes = (post.mesh.nodes[:, 0] < 0.4) & (np.abs(post.mesh.nodes[:, 1] - 0.5) < 0.2)  # Identify the left physical hotspot neighborhood.
    right_nodes = (post.mesh.nodes[:, 0] > 0.6) & (np.abs(post.mesh.nodes[:, 1] - 0.5) < 0.2)  # Identify the mirrored right neighborhood.
    assert action_left[left_nodes].mean() < action_left[right_nodes].mean()  # The first image must refine its own observed hotspot.
    assert action_right[right_nodes].mean() < action_right[left_nodes].mean()  # Moving only the image hotspot must move the refinement support.
    constant_left, constant_right = [make_targets(state, post.mesh, mode="constant")["err_s1"] for state in (left, right)]  # Run the same architecture without image location information.
    assert np.allclose(constant_left, constant_right)  # Removing spatial information must remove the translated action difference.
    shuffled = make_targets(left, post.mesh, mode="shuffled")["err_s1"]  # Destroy spatial registration while retaining channel statistics.
    assert not np.allclose(shuffled, action_left)  # The original physical image must matter beyond aggregate statistics.
    assert np.allclose(shuffled, make_targets(left, post.mesh, mode="shuffled")["err_s1"])  # Keep the ablation reproducible across benchmark branches.
    assert not np.allclose(features(left, action_left, post.mesh), features(right, action_right, post.mesh))  # Preserve location changes in the WM encoder as well.
