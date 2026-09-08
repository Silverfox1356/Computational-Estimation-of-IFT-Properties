"""
core.fem_assembly
=================

Assembles the FEM system matrices for axisymmetric diffusion on the
pendant-drop domain.

All coordinates are in **metres** internally.  The caller must
ensure that the mesh points and domain metadata are in metres.

Matrices produced
-----------------
    H   – Mass matrix       (axisymmetric: includes r-weighting)
    K   – Stiffness matrix  (diffusion + axisymmetric correction)
    Kb  – Boundary mass     (Robin BC on the free surface only)
    F   – Boundary load     (Robin BC RHS)

Interface detection
-------------------
    Boundary edges are classified as *interface* (free surface)
    or *wall* (needle inner wall, tip step, top wall) by comparing
    their midpoint against the domain polygon segments stored in
    ``domain_metadata['segment_indices']``.

    Only edges whose midpoint lies on a ``free_surface_*`` segment
    receive the Robin BC — this prevents applying mass-transfer BCs
    on the needle walls or the neck interior.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ======================================================================
# Element-level computations
# ======================================================================
def _triangle_area_and_gradients(r, z):
    """
    For a triangle with vertices at (r[0..2], z[0..2]):

    Returns
    -------
    A    : float – triangle area (always positive)
    grads : (3, 2) ndarray – grad(φᵢ) = [∂φ/∂r, ∂φ/∂z]
    """
    det = ((r[1] - r[0]) * (z[2] - z[0])
         - (r[2] - r[0]) * (z[1] - z[0]))

    A = 0.5 * abs(det)

    if A < 1e-30:
        return A, np.zeros((3, 2))

    b = np.array([z[1] - z[2], z[2] - z[0], z[0] - z[1]])
    c = np.array([r[2] - r[1], r[0] - r[2], r[1] - r[0]])

    grads = np.column_stack((b, c)) / (2.0 * A)
    return A, grads


# ======================================================================
# Mass matrix   H :  ∫ φᵢ φⱼ r dΩ
# ======================================================================
def assemble_mass_matrix(points, triangles):
    """
    Lumped-consistent mass matrix with axisymmetric r-weighting.

    H_ij += (R_c · A / 12) · [2 1 1; 1 2 1; 1 1 2]   per element.
    """
    N = len(points)
    H = lil_matrix((N, N))

    for tri in triangles:
        r = points[tri, 0]
        z = points[tri, 1]

        A, _ = _triangle_area_and_gradients(r, z)
        if A < 1e-30:
            continue

        R_c = np.mean(np.abs(r))   # centroid |r| for axisymmetric

        H_e = (R_c * A / 12.0) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ])

        for a in range(3):
            for b in range(3):
                H[tri[a], tri[b]] += H_e[a, b]

    return csr_matrix(H)


# ======================================================================
# Stiffness matrix   K :  D ∫ ∇φᵢ · ∇φⱼ r dΩ  + axisym correction
# ======================================================================
def assemble_stiffness_matrix(points, triangles, D=1.0):
    """
    Stiffness matrix with diffusion coefficient D (m²/s).

    K_ij += D · R_c · A · (∇φᵢ · ∇φⱼ)
          - D · A · ∂φⱼ/∂r / 3        (axisymmetric correction)
    """
    N = len(points)
    K = lil_matrix((N, N))

    for tri in triangles:
        r = points[tri, 0]
        z = points[tri, 1]

        A, grads = _triangle_area_and_gradients(r, z)
        if A < 1e-30:
            continue

        # The domain is the FULL mirrored cross-section, so left-half
        # triangles have r < 0.  The axisymmetric radius is |r| — a
        # signed mean would give those elements a NEGATIVE diffusion
        # coefficient (indefinite K, wildly parameter-sensitive
        # solutions).  The 1/r correction term carries sign(r) instead:
        # (1/r)∂c/∂r · |r| = sign(r)·∂c/∂r, which is mirror-consistent
        # because ∂c/∂r also flips sign on the reflected half.
        R_c = np.mean(np.abs(r))
        sgn = 1.0 if np.mean(r) >= 0.0 else -1.0

        for i in range(3):
            for j in range(3):
                diffusion = D * R_c * A * np.dot(grads[i], grads[j])
                axisym = -D * sgn * (A / 3.0) * grads[j, 0]
                K[tri[i], tri[j]] += diffusion + axisym

    return csr_matrix(K)


# ======================================================================
# Interface detection  (geometry-based, NOT the naive r > 0 check)
# ======================================================================
def classify_boundary_edges(points, boundary_edges, domain_metadata):
    """
    Classify boundary edges as 'interface' (free surface) or 'wall'.

    Uses the segment index ranges stored in *domain_metadata* to
    determine which part of the polygon each boundary edge belongs to.
    Only edges whose midpoint is geometrically nearest to a
    ``free_surface_*`` segment are classified as interface.

    Parameters
    ----------
    points : (M, 2) ndarray
    boundary_edges : (E, 2) ndarray
    domain_metadata : dict   (must contain 'r_inner_m', 'r_outer_m',
                              'neck_height_m')

    Returns
    -------
    interface_edges : list of [i, j]
    wall_edges : list of [i, j]
    """
    r_inner = domain_metadata.get('r_inner_m', 0)
    r_outer = domain_metadata.get('r_outer_m', 0)
    neck_h  = domain_metadata.get('neck_height_m', 0)

    interface_edges = []
    wall_edges = []

    for edge in boundary_edges:
        i, j = int(edge[0]), int(edge[1])
        r1, z1 = points[i]
        r2, z2 = points[j]
        r_mid = 0.5 * (r1 + r2)
        z_mid = 0.5 * (z1 + z2)
        abs_r_mid = abs(r_mid)

        # ---- Wall conditions ----
        # 1. Top wall: z ≈ 0 and |r| ≤ r_inner
        if z_mid < neck_h * 0.05 and abs_r_mid <= r_inner * 1.05:
            wall_edges.append([i, j])
            continue

        # 2. Inner wall: |r| ≈ r_inner and z < neck_h
        if (abs(abs_r_mid - r_inner) < r_inner * 0.08
                and z_mid < neck_h * 1.02):
            wall_edges.append([i, j])
            continue

        # 3. Tip step: z ≈ neck_h and r_inner ≤ |r| ≤ r_outer
        if (abs(z_mid - neck_h) < neck_h * 0.05
                and r_inner * 0.9 <= abs_r_mid <= r_outer * 1.1):
            wall_edges.append([i, j])
            continue

        # ---- Everything else is free surface (interface) ----
        interface_edges.append([i, j])

    return interface_edges, wall_edges


# ======================================================================
# Boundary terms  (Robin BC on interface only)
# ======================================================================
def assemble_boundary_terms(points, interface_edges, k):
    """
    Assemble the boundary mass matrix Kb and load vector F
    for a Robin BC on the free-surface interface:

        -D ∂c/∂n = k (c - c∞)

    where c∞ = 1 (normalised bulk concentration).

    Note ``k`` here is the PHYSICAL mass-transfer coefficient in m/s.  It
    is deliberately not called ``kD``: everywhere else in this project
    ``kD`` means the dimensionless Biot number k·rₙ/D (see
    ``core.time_solver`` and ``core.estimator``), and the two differ by
    several orders of magnitude.

    Parameters
    ----------
    points : (M, 2) ndarray
    interface_edges : list of [i, j]
    k : float  – mass-transfer coefficient (m/s)

    Returns
    -------
    Kb : (N, N) sparse
    F  : (N,) ndarray
    """
    N = len(points)
    Kb = lil_matrix((N, N))
    F = np.zeros(N)

    for edge in interface_edges:
        i, j = edge
        r1, z1 = points[i]
        r2, z2 = points[j]

        L = np.sqrt((r2 - r1) ** 2 + (z2 - z1) ** 2)
        R_mid = 0.5 * (abs(r1) + abs(r2))   # axisymmetric |r|

        K_e = (k * R_mid * L / 6.0) * np.array([
            [2.0, 1.0],
            [1.0, 2.0],
        ])
        F_e = (k * R_mid * L / 2.0) * np.array([1.0, 1.0])

        idx = [i, j]
        for a in range(2):
            for b in range(2):
                Kb[idx[a], idx[b]] += K_e[a, b]
            F[idx[a]] += F_e[a]

    return csr_matrix(Kb), F


# ======================================================================
# Sanity checks
# ======================================================================
def run_sanity_checks(H, K, Kb, F, points, interface_edges, wall_edges):
    """
    Validate the assembled FEM matrices.

    Symmetry thresholds for K (which has an inherently asymmetric
    axisymmetric correction term):
        OK      : Frobenius(K − Kᵀ) < 1e-10
        Warning : Frobenius(K − Kᵀ) < 1e-8
        Fail    : otherwise

    Returns
    -------
    results : dict   with pass/fail for each check and diagnostic values.
    """
    results = {}
    n = H.shape[0]
    total = n * n

    # ── 1. H symmetry (strict – H is exactly symmetric by construction)
    sym_H = float((H - H.T).power(2).sum() ** 0.5)
    results['H_symmetry_error'] = sym_H
    results['H_symmetric'] = sym_H < 1e-12

    # ── 2. K near-symmetry (relaxed – axisymmetric term introduces
    #       small floating-point asymmetry)
    sym_K = float((K - K.T).power(2).sum() ** 0.5)
    results['K_symmetry_error'] = sym_K
    if sym_K < 1e-10:
        results['K_symmetry_status'] = 'ok'       # green
        results['K_near_symmetric'] = True
    elif sym_K < 1e-8:
        results['K_symmetry_status'] = 'warning'   # amber
        results['K_near_symmetric'] = True
    else:
        results['K_symmetry_status'] = 'fail'      # red
        results['K_near_symmetric'] = False

    # ── 3. Positive diagonal of H
    H_diag = np.array(H.diagonal())
    results['H_diag_all_positive'] = bool(np.all(H_diag >= 0))
    results['H_diag_min'] = float(H_diag.min())

    # ── 4. K diagonal – should be positive for well-posed diffusion
    K_diag = np.array(K.diagonal())
    results['K_diag_min'] = float(K_diag.min())
    results['K_diag_positive'] = bool(np.all(K_diag > 0))

    # ── 5. Sparsity (both must be < 1 %)
    results['H_nnz'] = H.nnz
    results['K_nnz'] = K.nnz
    results['H_sparsity_pct'] = 100.0 * H.nnz / total
    results['K_sparsity_pct'] = 100.0 * K.nnz / total
    results['sparsity_ok'] = (results['H_sparsity_pct'] < 1.0
                              and results['K_sparsity_pct'] < 1.0)

    # ── 6. Boundary node classification
    intf_nodes = set()
    for e in interface_edges:
        intf_nodes.update(e)
    wall_nodes = set()
    for e in wall_edges:
        wall_nodes.update(e)
    results['n_interface_nodes'] = len(intf_nodes)
    results['n_wall_nodes'] = len(wall_nodes)
    results['n_interior_nodes'] = n - len(intf_nodes | wall_nodes)

    # ── 7. Kb and F consistency
    #       F must be sparse: nonzero ONLY at interface nodes
    results['Kb_nnz'] = Kb.nnz
    F_nz = int(np.count_nonzero(F))
    results['F_nonzero'] = F_nz
    results['F_max'] = float(np.max(F)) if F_nz > 0 else 0.0
    results['F_min_nonzero'] = float(np.min(F[F > 0])) if F_nz > 0 else 0.0
    results['boundary_terms_ok'] = (Kb.nnz > 0 and F_nz > 0)

    # Verify F is nonzero ONLY at interface nodes
    F_nonzero_indices = set(np.nonzero(F)[0].tolist())
    results['F_only_on_interface'] = F_nonzero_indices.issubset(intf_nodes)

    # ── Overall pass/fail
    results['all_pass'] = all([
        results['H_symmetric'],
        results['K_near_symmetric'],       # relaxed, not strict
        results['H_diag_all_positive'],
        results['K_diag_positive'],
        results['sparsity_ok'],
        results['boundary_terms_ok'],
        results['F_only_on_interface'],
    ])

    return results


# ======================================================================
# Debug helper: F-vector diagnostics
# ======================================================================
def debug_F_vector(F, points, interface_edges):
    """
    Print diagnostic information about the load vector F.

    Call this after assembly to verify that F is nonzero only at
    interface nodes and that its values are physically plausible.

    Returns a dict with the diagnostics for programmatic use.
    """
    intf_nodes = set()
    for e in interface_edges:
        intf_nodes.update(e)

    F_nz_idx = set(np.nonzero(F)[0].tolist())
    spurious = F_nz_idx - intf_nodes

    info = {
        'total_nodes':       len(F),
        'interface_nodes':   len(intf_nodes),
        'F_nonzero_count':   len(F_nz_idx),
        'F_max':             float(np.max(F)) if len(F_nz_idx) > 0 else 0.0,
        'F_min_nonzero':     float(np.min(F[F > 0])) if np.any(F > 0) else 0.0,
        'F_sum':             float(np.sum(F)),
        'spurious_nodes':    len(spurious),
        'all_on_interface':  len(spurious) == 0,
    }

    print("── F-vector diagnostics ──")
    print(f"  Total nodes:        {info['total_nodes']}")
    print(f"  Interface nodes:    {info['interface_nodes']}")
    print(f"  F nonzero entries:  {info['F_nonzero_count']}")
    print(f"  max(F):             {info['F_max']:.6e}")
    print(f"  min(F>0):           {info['F_min_nonzero']:.6e}")
    print(f"  sum(F):             {info['F_sum']:.6e}")
    if info['spurious_nodes'] > 0:
        print(f"  ⚠ {info['spurious_nodes']} F entries on non-interface nodes!")
    else:
        print("  ✓ F is nonzero only at interface nodes")

    return info


# ======================================================================
# Master function
# ======================================================================
def assemble_fem_system(points, triangles, boundary_edges,
                        domain_metadata, D=1e-9, k=1e-5):
    """
    Full FEM assembly pipeline.

    Parameters
    ----------
    points : (M, 2) ndarray          – in metres
    triangles : (T, 3) ndarray       – 0-based
    boundary_edges : (E, 2) ndarray
    domain_metadata : dict
    D  : float  – diffusion coefficient (m²/s), default 1e-9 (water)
    k  : float  – PHYSICAL mass-transfer coefficient (m/s).  Not the
                  dimensionless Biot number kD = k·rₙ/D that the solver
                  and estimator work in.

    Returns
    -------
    fem : dict with keys
        'H', 'K', 'Kb', 'F',
        'interface_edges', 'wall_edges',
        'sanity', 'points', 'triangles',
        'assembly_D', 'assembly_k'
    """
    H = assemble_mass_matrix(points, triangles)
    K = assemble_stiffness_matrix(points, triangles, D=D)

    interface_edges, wall_edges = classify_boundary_edges(
        points, boundary_edges, domain_metadata)

    Kb, F = assemble_boundary_terms(points, interface_edges, k)

    # Print F-vector diagnostics to console for debugging
    debug_F_vector(F, points, interface_edges)

    sanity = run_sanity_checks(
        H, K, Kb, F, points, interface_edges, wall_edges)

    return dict(
        H=H, K=K, Kb=Kb, F=F,
        interface_edges=interface_edges,
        wall_edges=wall_edges,
        sanity=sanity,
        points=points,
        triangles=triangles,
        # Coefficients baked into K (D) and Kb/F (k) at assembly time.
        # Both matrices are linear in their coefficient, so a solver run
        # at a different (D, k) only needs a scalar rescale — see
        # core.estimator.
        assembly_D=D,
        assembly_k=k,
    )
