"""Polar structured Elmer mesh writer for the radial PMSM cross-section.

Writes mesh.header/nodes/elements/boundary directly in ElmerSolver's
native mesh-directory format (no ElmerGrid, no Gmsh) — same approach as
scripts/elmer_axial/mesh_writer.py, in polar coordinates.

Layout (inrunner): center triangle fan -> shaft -> pm -> airgap ->
stator, on a tensor-product (r, theta) grid. A node ring lands exactly
at r_ag and carries boundary 2 for SaveLine sampling; boundary 1 is the
outer Dirichlet circle. Magnet arcs (alpha_p < 1) and slots are pure
theta-sector body assignment — no boolean geometry, so none of the OCC
sliver failure modes, but arc and slot edges snap to element boundaries:
alpha_p and slot_width_ratio are quantized by n_theta (see README).
"""
from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

BODY = {
    "shaft": 1, "pm": 2, "airgap": 3, "stator": 4,
    "pm_gap": 5, "slot": 6,
}


def write_radial_mesh(mesh_dir, *, r_outer, r_stator, r_ag, r_magnet,
                      r_rotor, n_p, alpha_p=1.0, n_theta=720,
                      nr=None, r0_frac=0.02,
                      n_slots=0, slot_depth=0.0, slot_width_ratio=0.6):
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    r0 = r0_frac * r_outer
    # radial layer boundaries and the body each annulus belongs to
    layers = [
        (r0, r_rotor, "shaft", 6),
        (r_rotor, r_magnet, "pm", 10),
        (r_magnet, r_ag, "airgap", 4),
        (r_ag, r_stator, "airgap", 4),
    ]
    if n_slots > 0 and slot_depth > 0:
        layers += [(r_stator, r_stator + slot_depth, "stator", 5),
                   (r_stator + slot_depth, r_outer, "stator", 5)]
    else:
        layers += [(r_stator, r_outer, "stator", 8)]
    if nr:
        layers = [(a, b, n, nr) for (a, b, n, _) in layers]

    radii = [0.0, r0]
    ring_body = []            # body name for annulus between radii[i], radii[i+1]
    for a, b, name, n in layers:
        for k in range(1, n + 1):
            radii.append(a + (b - a) * k / n)
            ring_body.append(name)
    ring_body = ["shaft", *ring_body]   # innermost fan is shaft

    i_ag = radii.index(min(radii, key=lambda r: abs(r - r_ag)))
    thetas = [2 * pi * j / n_theta for j in range(n_theta)]

    # node ids: 1 = center; ring i (i>=1) node j -> 1 + (i-1)*n_theta + j + 1
    def nid(i, j):
        if i == 0:
            return 1
        return 1 + (i - 1) * n_theta + (j % n_theta) + 1

    n_rings = len(radii) - 1          # rings 1..n_rings at radii[1..]
    n_nodes = 1 + n_rings * n_theta

    def body_of(name, r_mid, th_mid):
        if name == "pm" and alpha_p < 1.0:
            # magnet arcs centered on pole axes th_k = k*pi/n_p
            phase = (th_mid * n_p / pi) % 1.0     # 0 at a pole axis
            d = min(phase, 1.0 - phase)           # distance to axis, in half-pitches
            return "pm" if d <= alpha_p / 2 else "pm_gap"
        if (name == "stator" and n_slots > 0 and slot_depth > 0
                and r_mid < r_stator + slot_depth):
            phase = (th_mid * n_slots / (2 * pi)) % 1.0
            d = min(phase, 1.0 - phase)
            return "slot" if d <= slot_width_ratio / 2 else "stator"
        return name

    elements = []   # (body_id, [nodes])
    for j in range(n_theta):
        elements.append((BODY["shaft"], [nid(0, 0), nid(1, j), nid(1, j + 1)]))
    for i in range(1, n_rings):
        name = ring_body[i]
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        for j in range(n_theta):
            th_mid = 2 * pi * (j + 0.5) / n_theta
            b = BODY[body_of(name, r_mid, th_mid)]
            elements.append((b, [nid(i, j), nid(i, j + 1),
                                 nid(i + 1, j + 1), nid(i + 1, j)]))

    # element index lookup for boundary parents
    def quad_elem_id(i, j):     # 1-based
        return n_theta + (i - 1) * n_theta + j + 1

    boundary = []
    for j in range(n_theta):    # BC 1: outer Dirichlet
        boundary.append((1, quad_elem_id(n_rings - 1, j),
                         nid(n_rings, j), nid(n_rings, j + 1)))
    for j in range(n_theta):    # BC 2: mid-gap sampling circle
        boundary.append((2, quad_elem_id(i_ag - 1, j),
                         nid(i_ag, j), nid(i_ag, j + 1)))

    n_tri = n_theta
    n_quad = len(elements) - n_tri
    (mesh_dir / "mesh.header").write_text(
        f"{n_nodes} {len(elements)} {len(boundary)}\n"
        f"3\n303 {n_tri}\n404 {n_quad}\n202 {len(boundary)}\n")
    with open(mesh_dir / "mesh.nodes", "w") as f:
        f.write("1 -1 0.0 0.0 0.0\n")
        n = 2
        for i in range(1, n_rings + 1):
            r = radii[i]
            for th in thetas:
                f.write(f"{n} -1 {r*cos(th):.12e} {r*sin(th):.12e} 0.0\n")
                n += 1
    with open(mesh_dir / "mesh.elements", "w") as f:
        for eid, (b, ns) in enumerate(elements, start=1):
            code = 303 if len(ns) == 3 else 404
            f.write(f"{eid} {b} {code} " + " ".join(str(k) for k in ns) + "\n")
    with open(mesh_dir / "mesh.boundary", "w") as f:
        for bid, (bc, parent, n1, n2) in enumerate(boundary, start=1):
            f.write(f"{bid} {bc} {parent} 0 202 {n1} {n2}\n")
    return {"n_nodes": n_nodes, "n_elements": len(elements),
            "r_ag_actual": radii[i_ag], "n_boundary": len(boundary)}
