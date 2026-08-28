import numpy as np

from visionamr.fem_post import compute_post
from visionamr.geometry import make_plate_holes
from visionamr.mesher import Mesh
from visionamr.vla.regions import Partition, Seed


def grid_mesh(nx=12, ny=4) -> Mesh:
    """Structured triangle grid on [0,nx] x [0,ny]."""

    nodes, cells = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i, j, 0.0])
    idx = lambda i, j: j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            cells.append([a, b, c])
            cells.append([a, c, d])
    return Mesh(nodes=np.array(nodes, float), cells=np.array(cells), dim=2)


def make_partition(mesh):
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    seeds = [
        Seed("left", (1.0, 2.0, 0.0), h=0.5),
        Seed("right", (11.0, 2.0, 0.0), h=1.0),
    ]
    return Partition(seeds, problem), problem


def test_geodesic_assignment_covers_domain():
    mesh = grid_mesh()
    part, _ = make_partition(mesh)
    part.assign_mode = "geodesic"
    labels = part.assign(mesh)
    assert labels.min() >= 0            # no background: a true partition
    assert set(np.unique(labels)) == {0, 1}
    cen = mesh.centroids
    # far-left cells belong to the left seed, far-right to the right seed
    assert np.all(labels[cen[:, 0] < 1.0] == 0)
    assert np.all(labels[cen[:, 0] > 11.0] == 1)


def test_partition_adjacency_and_features():
    mesh = grid_mesh()
    part, problem = make_partition(mesh)
    labels = part.assign(mesh)
    adj = part.adjacency(mesh, labels)
    assert 1 in adj[0] and 0 in adj[1]

    u = np.zeros((mesh.n_nodes, 3))
    u[:, 0] = 1e-3 * mesh.nodes[:, 0]
    post = compute_post(mesh, problem, u)
    eta2 = np.ones(mesh.n_cells)
    feats = part.features(post, eta2, labels)
    assert feats.elems.sum() == mesh.n_cells
    assert np.isclose(feats.err_sum.sum(), eta2.sum())
    assert np.isclose(feats.volume.sum(), mesh.measures.sum())


def test_region_shapes_are_not_boxes():
    """Geodesic growth follows the mesh graph, so the interface between
    two unevenly placed seeds is a distance bisector, not an axis box."""

    mesh = grid_mesh(12, 4)
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    part = Partition(
        [Seed("a", (2.0, 0.5, 0.0), h=1.0), Seed("b", (8.0, 3.5, 0.0), h=1.0)],
        problem,
    )
    labels = part.assign(mesh)
    cen = mesh.centroids
    # pick the boundary cells of region a: their x-extent varies with y,
    # which an axis-aligned box cannot represent
    xa_top = cen[(labels == 0) & (cen[:, 1] > 3.0), 0].max()
    xa_bot = cen[(labels == 0) & (cen[:, 1] < 1.0), 0].max()
    assert abs(xa_top - xa_bot) > 1.0


def test_linf_box_assignment_is_chebyshev():
    mesh = grid_mesh(12, 4)
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    part = Partition(
        [Seed("a", (1.0, 2.0, 0.0), h=1.0), Seed("b", (11.0, 2.0, 0.0), h=1.0)],
        problem,
        assign_mode="linf_box",
    )
    labels = part.assign(mesh)
    assert labels.min() >= 0
    # a vertical interface (axis-aligned), unlike the geodesic slanted case
    cen = mesh.centroids
    xs_left = cen[labels == 0, 0]
    xs_right = cen[labels == 1, 0]
    assert xs_left.max() < xs_right.min() + 1.5


def test_drawn_regions_are_not_boxes_and_cover_domain():
    """Eye-drawn polygons assign labels; leftover is the field; not an AABB split."""

    from visionamr.vla.drawing import DrawnRegion

    mesh = grid_mesh(12, 4)
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    left = DrawnRegion(
        "hot", 0.3,
        "top",
        ((0.0, 0.0), (3.0, 0.2), (8.5, 3.8), (0.0, 4.0)),
    )
    part = Partition(
        [Seed("hot", (2.0, 2.0, 0.0), h=0.3), Seed("field", (10.0, 2.0, 0.0), h=1.0, origin="coarse")],
        problem,
        drawings=[left],
    )
    labels = part.assign(mesh)
    assert labels.min() >= 0
    assert set(np.unique(labels)) == {0, 1}
    cen = mesh.centroids
    xa_top = cen[(labels == 0) & (cen[:, 1] > 3.0), 0]
    xa_bot = cen[(labels == 0) & (cen[:, 1] < 1.0), 0]
    assert len(xa_top) and len(xa_bot)
    assert abs(xa_top.max() - xa_bot.max()) > 0.8


def test_section_drawing_only_paints_near_the_cut():
    from visionamr.vla.drawing import DrawnRegion

    mesh = grid_mesh(12, 4)
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    # 2-D plate lives in z=0; a yz section at x=2 should only tag a slab.
    column = DrawnRegion(
        "column", 0.25, "section",
        ((0.0, -1.0), (4.0, -1.0), (4.0, 1.0), (0.5, 0.8)),
        plane="yz", cut=2.0, slab=1.2,
    )
    part = Partition(
        [Seed("column", (2.0, 2.0, 0.0), h=0.25),
         Seed("field", (10.0, 2.0, 0.0), h=0.8, origin="coarse")],
        problem,
        drawings=[column],
    )
    labels = part.assign(mesh)
    cen = mesh.centroids
    tagged = labels == 0
    assert tagged.any()
    assert np.all(np.abs(cen[tagged, 0] - 2.0) <= 1.2 + 1e-9)
    assert (labels == 1).any()


def test_scripted_head_draws_then_assigns_eye_sizes():
    from visionamr.vla.partition import ScriptedVisionPartitioner
    from visionamr.vla.pipeline import vision_assigned_sizes

    mesh = grid_mesh(12, 4)
    problem = make_plate_holes(width=12.0, height=4.0, holes=(), tension=1.0)
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 0] = 1e-3 * mesh.nodes[:, 0]
    post = compute_post(mesh, problem, u)
    head = ScriptedVisionPartitioner()
    seeds = head.propose(problem, post, np.ones(mesh.n_cells))
    assert head.last_drawings
    for d in head.last_drawings:
        assert len(d.polygon) >= 3
        # a box has two unique x and two unique y at the corners only;
        # an irregular halo/hull has more distinct vertices than 4, or
        # an edge that is not axis-aligned.
        axis_edges = 0
        poly = d.polygon
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            if abs(p[0] - q[0]) < 1e-9 or abs(p[1] - q[1]) < 1e-9:
                axis_edges += 1
        assert axis_edges < len(poly) or len(poly) > 4
    part = Partition(seeds, problem, drawings=list(head.last_drawings))
    labels = part.assign(mesh)
    assert labels.min() >= 0
    h = vision_assigned_sizes(part, problem)
    assert np.allclose(h, [s.h for s in seeds])


def test_split_concentrated_adds_child_seed():
    mesh = grid_mesh(20, 4)
    problem = make_plate_holes(width=20.0, height=4.0, holes=(), tension=1.0)
    part = Partition([Seed("all", (10.0, 2.0, 0.0), h=1.0)], problem)
    labels = part.assign(mesh)
    u = np.zeros((mesh.n_nodes, 3))
    post = compute_post(mesh, problem, u)
    eta2 = np.full(mesh.n_cells, 1e-8)
    hot = np.argmin(np.linalg.norm(mesh.centroids - [2.0, 0.5, 0.0], axis=1))
    eta2[hot] = 1.0
    grown = part.split_concentrated(post, eta2, labels, min_elems=10)
    assert len(grown.seeds) == 2
    assert grown.seeds[1].origin == "split"
    assert grown.seeds[1].h < part.seeds[0].h
    d = np.linalg.norm(np.array(grown.seeds[1].xyz) - mesh.centroids[hot])
    assert d < 1e-9
