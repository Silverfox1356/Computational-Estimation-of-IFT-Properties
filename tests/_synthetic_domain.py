"""
tests._synthetic_domain
=======================

Shared test fixture: a synthetic pendant-drop FEM system built from an
analytic (spherical-cap) contour, so tests need no camera data.

Import ``get_fem_system()`` — the mesh + assembly runs once per process
and is cached.
"""

import sys
from pathlib import Path

import numpy as np

# Tests are run as plain scripts from anywhere; make `core` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domain_builder import build_domain_polygon
from core.mesh_generator import generate_mesh
from core.fem_assembly import assemble_fem_system

# Geometry: 1.5 mm-radius spherical drop hanging from a 1.6 mm-OD needle.
R_DROP = 1.5e-3
R_OUTER = 0.8e-3
R_INNER = 0.5e-3

_cache = None


def get_fem_system():
    """(fem_results, domain_metadata) for the synthetic drop, cached."""
    global _cache
    if _cache is not None:
        return _cache

    # Spherical-cap contour: sphere of radius R_DROP whose surface
    # passes through (r=R_OUTER, z=0) at the needle tip.
    z_center = np.sqrt(R_DROP ** 2 - R_OUTER ** 2)
    z_apex = z_center + R_DROP
    z = np.linspace(0.0, 0.985 * z_apex, 80)
    r = np.sqrt(np.maximum(R_DROP ** 2 - (z - z_center) ** 2, 0.0))

    polygon, metadata = build_domain_polygon(
        r, z, r_inner_m=R_INNER, r_outer_m=R_OUTER)

    # Coarse mesh on purpose: the estimator tests run dozens of forward
    # simulations, and Phase A only needs a consistent forward model.
    points, triangles, boundary_edges, _quality = generate_mesh(
        polygon, metadata,
        size_min_frac=0.02, size_max_frac=0.08)

    fem_results = assemble_fem_system(
        points, triangles, boundary_edges, metadata)

    _cache = (fem_results, metadata)
    return _cache
