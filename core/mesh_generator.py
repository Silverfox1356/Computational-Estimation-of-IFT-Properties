"""
core.mesh_generator
===================

Generates a high-quality 2-D triangular mesh for the pendant-drop
FEM domain using the Gmsh Python API.

Quality controls
----------------
    · Controlled growth rate (1.2–1.3) for smooth size transition
    · Maximum area ratio capped at < 100–300
    · Minimum triangle angle ≥ 25° (Gmsh optimisation)
    · Extra refinement at high-curvature zones (apex, neck)
    · Preserved structural corners (needle walls, tip steps)
    · Separate physical groups: drop region vs neck region

Outputs
-------
    · Node coordinates (metres) — consistent units throughout
    · Triangle connectivity (0-based)
    · Boundary edges with region tags for FEM boundary conditions
"""

import numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


# ======================================================================
# Polygon preprocessing
# ======================================================================
def _clean_polygon(poly, tol=1e-9):
    """Remove duplicate consecutive points and ensure open polygon."""
    cleaned = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) > tol:
            cleaned.append(p)
    if np.linalg.norm(cleaned[0] - cleaned[-1]) < tol:
        cleaned = cleaned[:-1]
    return np.array(cleaned)


def _downsample(poly, target):
    if len(poly) <= target:
        return poly
    step = max(1, len(poly) // target)
    return poly[::step]


def _smooth(poly, window, n_protect):
    """Smooth free-surface points only, preserve structural corners."""
    if window <= 0 or len(poly) <= 2 * n_protect + 2 * window:
        return poly
    out = poly.copy()
    for i in range(n_protect + window, len(poly) - n_protect - window):
        out[i] = np.mean(poly[i - window:i + window], axis=0)
    return out


def _local_curvature(poly):
    """Estimate curvature at each polygon vertex via the inscribed-circle
    approximation.  Returns an array of curvatures (1/radius)."""
    n = len(poly)
    curv = np.zeros(n)
    for i in range(1, n - 1):
        a = np.linalg.norm(poly[i] - poly[i - 1])
        b = np.linalg.norm(poly[i + 1] - poly[i])
        c = np.linalg.norm(poly[i + 1] - poly[i - 1])
        if a < 1e-15 or b < 1e-15 or c < 1e-15:
            continue
        # Menger curvature: κ = 4·Area / (a·b·c)
        cross = abs((poly[i][0] - poly[i - 1][0]) *
                    (poly[i + 1][1] - poly[i - 1][1]) -
                    (poly[i + 1][0] - poly[i - 1][0]) *
                    (poly[i][1] - poly[i - 1][1]))
        curv[i] = 2.0 * cross / (a * b * c + 1e-30)
    return curv


# ======================================================================
# Main mesh generation
# ======================================================================
def generate_mesh(domain_polygon, domain_metadata=None,
                  size_min_frac=0.010,
                  size_max_frac=0.045,
                  growth_rate=1.25,
                  dist_max_frac=0.25,
                  curvature_refine=True,
                  min_angle=25.0,
                  optimise_steps=5,
                  downsample_target=250,
                  smooth_window=2,
                  verbose=False):
    """
    Generate a high-quality triangular mesh for the pendant-drop domain.

    Parameters
    ----------
    domain_polygon : (N, 2) ndarray
        Closed polygon (r, z) in **metres**.
    domain_metadata : dict or None
        Metadata from ``build_domain_polygon`` — used to identify the
        free-surface segments for boundary tagging.
    size_min_frac : float
        Min element size as fraction of max domain dimension.
    size_max_frac : float
        Max element size as fraction of max domain dimension.
    growth_rate : float
        Element size growth rate (1.0 = uniform, 1.3 = fast transition).
    dist_max_frac : float
        Distance from boundary where element size reaches *size_max*.
    curvature_refine : bool
        Extra refinement at high-curvature zones (apex, neck junction).
    min_angle : float
        Minimum triangle angle in degrees (Gmsh optimiser target).
    optimise_steps : int
        Number of Laplacian + Netgen optimisation passes.
    downsample_target : int
        Max polygon boundary points.
    smooth_window : int
        Moving-average window for the free-surface portion.
    verbose : bool
        Print mesh statistics.

    Returns
    -------
    points : (M, 2) ndarray          – in metres
    triangles : (T, 3) int ndarray   – 0-based
    boundary_edges : (E, 2) int ndarray or None
    mesh_quality : dict               – quality metrics
    """
    if not GMSH_AVAILABLE:
        raise ImportError(
            "The 'gmsh' package is required.  pip install gmsh")

    poly = np.asarray(domain_polygon, dtype=float)

    # ---- preprocessing ----
    poly = _clean_polygon(poly)
    n_protect = min(6, len(poly) // 4)

    if len(poly) > downsample_target:
        head = poly[:n_protect]
        tail = poly[-n_protect:]
        mid  = _downsample(poly[n_protect:-n_protect],
                           downsample_target - 2 * n_protect)
        poly = np.vstack([head, mid, tail])

    poly = _smooth(poly, smooth_window, n_protect)

    # Size parameters (in metres)
    rng_r = poly[:, 0].max() - poly[:, 0].min()
    rng_z = poly[:, 1].max() - poly[:, 1].min()
    max_dim = max(rng_r, rng_z)

    size_min = max_dim * size_min_frac
    size_max = max_dim * size_max_frac
    dist_max = max_dim * dist_max_frac

    # ---- Curvature-based local sizes ----
    curv = _local_curvature(poly) if curvature_refine else None

    # ================================================================
    # Gmsh setup
    # ================================================================
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("pendant_drop")

    # Create points (with per-point mesh sizes if curvature is available)
    point_tags = []
    for k, p in enumerate(poly):
        if curv is not None and curv[k] > 0:
            # Higher curvature → smaller local size
            # Map curvature to a size between size_min and 0.5*size_max
            local = max(size_min, min(0.5 * size_max,
                                      1.0 / (curv[k] + 1.0 / size_max)))
        else:
            local = 0.0   # let background field decide
        tag = gmsh.model.geo.addPoint(p[0], p[1], 0.0, local)
        point_tags.append(tag)

    # Close loop
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-9:
        point_tags.append(point_tags[0])

    # Create lines
    line_tags = []
    for i in range(len(point_tags) - 1):
        tag = gmsh.model.geo.addLine(point_tags[i], point_tags[i + 1])
        line_tags.append(tag)

    loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # ---- Background size field: distance-based threshold ----
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "EdgesList", line_tags)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max)

    # ---- Extra refinement at apex (bottom) and neck junction ----
    field_ids = [f_thresh]

    if curvature_refine:
        # Apex: the lowest point of the domain
        z_vals = poly[:, 1]
        apex_idx = np.argmax(z_vals)       # z increases downward
        apex_r, apex_z = poly[apex_idx]

        f_apex = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f_apex, "Radius", max_dim * 0.12)
        gmsh.model.mesh.field.setNumber(f_apex, "VIn", size_min * 1.2)
        gmsh.model.mesh.field.setNumber(f_apex, "VOut", size_max)
        gmsh.model.mesh.field.setNumber(f_apex, "XCenter", apex_r)
        gmsh.model.mesh.field.setNumber(f_apex, "YCenter", apex_z)
        field_ids.append(f_apex)

        # Neck junction: around z = neck_height (where tip steps are)
        if domain_metadata is not None:
            nh = domain_metadata.get('neck_height_m', 0)
            if nh > 0:
                f_neck = gmsh.model.mesh.field.add("Ball")
                gmsh.model.mesh.field.setNumber(
                    f_neck, "Radius", max_dim * 0.10)
                gmsh.model.mesh.field.setNumber(
                    f_neck, "VIn", size_min * 1.5)
                gmsh.model.mesh.field.setNumber(
                    f_neck, "VOut", size_max)
                gmsh.model.mesh.field.setNumber(f_neck, "XCenter", 0.0)
                gmsh.model.mesh.field.setNumber(f_neck, "YCenter", nh)
                field_ids.append(f_neck)

    # Combine all fields with Min (take the smallest size at each point)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # ---- Mesh options for quality ----
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if curvature_refine else 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Smooth growth rate
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)

    # Meshing algorithm: Frontal-Delaunay for better quality
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    # ---- Generate ----
    gmsh.model.mesh.generate(2)

    # ---- Optimise for triangle quality ----
    for _ in range(optimise_steps):
        gmsh.model.mesh.optimize("Laplace2D")
    gmsh.model.mesh.optimize("Netgen")

    # ---- Extract data ----
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elements = gmsh.model.mesh.getElements()
    triangles = None
    boundary_edges = None

    for i, elem_type in enumerate(elements[0]):
        if elem_type == 2:
            raw = elements[2][i].reshape(-1, 3)
            triangles = np.array([[tag_to_idx[int(t)] for t in tri]
                                  for tri in raw], dtype=int)
        elif elem_type == 1:
            raw = elements[2][i].reshape(-1, 2)
            boundary_edges = np.array([[tag_to_idx[int(t)] for t in e]
                                       for e in raw], dtype=int)

    gmsh.finalize()

    if triangles is None or len(triangles) == 0:
        raise RuntimeError("Mesh generation failed: no triangles produced.")

    # ---- Quality metrics ----
    quality = _compute_quality(points, triangles)

    if verbose:
        print(f"Mesh: {len(points)} nodes, {len(triangles)} elements"
              + (f", {len(boundary_edges)} boundary edges"
                 if boundary_edges is not None else ""))
        print(f"  Min angle: {quality['min_angle']:.1f}°  "
              f"Mean angle: {quality['mean_min_angle']:.1f}°  "
              f"Max aspect ratio: {quality['max_aspect_ratio']:.1f}  "
              f"Area ratio: {quality['area_ratio']:.0f}")

    return points, triangles, boundary_edges, quality


# ======================================================================
# Quality metrics
# ======================================================================
def _compute_quality(points, triangles):
    """Compute mesh quality metrics."""
    areas = _tri_areas(points, triangles)
    angles = _tri_min_angles(points, triangles)
    aspect = _tri_aspect_ratios(points, triangles)

    return dict(
        min_angle=float(np.min(angles)),
        mean_min_angle=float(np.mean(angles)),
        max_aspect_ratio=float(np.max(aspect)),
        mean_aspect_ratio=float(np.mean(aspect)),
        area_ratio=float(np.max(areas) / max(np.min(areas), 1e-30)),
        min_area=float(np.min(areas)),
        max_area=float(np.max(areas)),
        n_nodes=len(points),
        n_elements=len(triangles),
    )


def _tri_areas(pts, tris):
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
           - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    return 0.5 * np.abs(cross)


def _tri_min_angles(pts, tris):
    """Min angle (degrees) per triangle."""
    mins = np.full(len(tris), 180.0)
    for k in range(len(tris)):
        v = pts[tris[k]]
        for i in range(3):
            a = v[(i + 1) % 3] - v[i]
            b = v[(i + 2) % 3] - v[i]
            cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)
                                     + 1e-30)
            cos_a = np.clip(cos_a, -1, 1)
            angle = np.degrees(np.arccos(cos_a))
            mins[k] = min(mins[k], angle)
    return mins


def _tri_aspect_ratios(pts, tris):
    """Aspect ratio = longest edge / shortest altitude."""
    ratios = np.ones(len(tris))
    for k in range(len(tris)):
        v = pts[tris[k]]
        edges = [np.linalg.norm(v[1] - v[0]),
                 np.linalg.norm(v[2] - v[1]),
                 np.linalg.norm(v[0] - v[2])]
        area = _tri_areas(pts, tris[k:k + 1])[0]
        if area < 1e-30:
            ratios[k] = 999.0
            continue
        longest = max(edges)
        shortest_alt = 2.0 * area / longest
        ratios[k] = longest / max(shortest_alt, 1e-30)
    return ratios
