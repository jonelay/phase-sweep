"""Structured quad mesh writer for the Elmer axial-flux pole problem.

Writes the one-pole rectangle stack of the shared physics contract
(tests/crossval/axial_fem_contract.py) directly in
ElmerSolver's native mesh-directory format (mesh.header / mesh.nodes /
mesh.elements / mesh.boundary), bypassing ElmerGrid.

Geometry (unrolled linear machine at radius r, SI units, metres):
  x = circumferential, one pole pitch [0, tau_p]; magnet centers on the
      two vertical edges (mirror-symmetry Dirichlet planes).
  y = axial, stack bottom-to-top:
      rotor_yoke / (pm_n + interpole air + pm_s) / airgap / stator_yoke
  Slotless only (v1 of this study): the stator is solid iron above the
  gap, matching the analytical layer series with g' = g.

Body ids (referenced by the SIF template):
  1 rotor_yoke  2 pm_n  3 air_interpole  4 pm_s  5 airgap  6 stator_yoke
Boundary ids:
  1 = entire outer boundary (both magnet-center vertical edges and both
      outer horizontal edges): Dirichlet A_z = 0 per the contract.

At alpha_p = 1 the interpole segment has zero width and body 3 gets no
elements; the SIF keeps the body block (Elmer accepts an element-less
body).
"""

from __future__ import annotations

from pathlib import Path


def _segment_divisions(length: float, dx_target: float, minimum: int = 2) -> int:
    """Number of uniform divisions for a segment aiming at dx_target [m]."""
    if length <= 0.0:
        return 0
    return max(minimum, round(length / dx_target))


def build_grid(
    tau_p: float,
    alpha_p: float,
    layers: list[tuple[str, float, int]],
    nx: int,
) -> tuple[list[float], list[float], list[int], list[int]]:
    """Build the tensor-product grid.

    Args:
      tau_p: pole pitch [m].
      alpha_p: magnet fraction of the pole pitch (0 < alpha_p <= 1).
      layers: bottom-to-top (name, thickness [m], n_divisions).
      nx: target number of x divisions across tau_p.

    Returns (xs, ys, xseg, yseg) where xs/ys are node coordinates [m]
    and xseg/yseg give, per interval, the x-segment index (0 = pm_n,
    1 = interpole air, 2 = pm_s) and the layer index into `layers`.
    """
    if not (0.0 < alpha_p <= 1.0):
        raise ValueError(f"alpha_p must be in (0, 1], got {alpha_p}")
    w_half = alpha_p * tau_p / 2.0  # half-magnet width at each edge [m]
    dx = tau_p / nx
    xs: list[float] = [0.0]
    xseg: list[int] = []
    segments = [(w_half, 0), (tau_p - 2.0 * w_half, 1), (w_half, 2)]
    x0 = 0.0
    for length, seg_id in segments:
        if length <= 1e-12 * tau_p:
            continue
        n = _segment_divisions(length, dx)
        for i in range(1, n + 1):
            xs.append(x0 + length * i / n)
            xseg.append(seg_id)
        x0 += length
    xs[-1] = tau_p  # exact, no float accumulation

    ys: list[float] = [0.0]
    yseg: list[int] = []
    y0 = 0.0
    for li, (_name, t, ny) in enumerate(layers):
        if t <= 0.0:
            raise ValueError(f"layer {li} thickness must be > 0, got {t}")
        for i in range(1, ny + 1):
            ys.append(y0 + t * i / ny)
            yseg.append(li)
        y0 += t
    return xs, ys, xseg, yseg


def write_pole_mesh(
    mesh_dir: str | Path,
    *,
    tau_p: float,
    alpha_p: float,
    t_rotor_yoke: float,
    t_pm: float,
    t_gap: float,
    t_stator_yoke: float,
    nx: int = 192,
    ny_yoke: int = 8,
    ny_pm: int = 8,
    ny_gap: int = 6,
) -> dict:
    """Write the Elmer mesh directory. All thicknesses in metres.

    Returns a small dict of derived values the runner needs:
    {y_mid_gap [m], n_nodes, n_elements, n_boundary}.
    """
    layers = [
        ("rotor_yoke", t_rotor_yoke, ny_yoke),
        ("pm", t_pm, ny_pm),
        ("gap", t_gap, ny_gap),
        ("stator_yoke", t_stator_yoke, ny_yoke),
    ]
    xs, ys, xseg, yseg = build_grid(tau_p, alpha_p, layers, nx)
    nxn, nyn = len(xs), len(ys)

    def node_id(i: int, j: int) -> int:  # 1-based Elmer node numbering
        return j * nxn + i + 1

    # Body id per element from (x-segment, layer) — see module docstring.
    pm_layer_body = {0: 2, 1: 3, 2: 4}
    layer_body = {"rotor_yoke": 1, "gap": 5, "stator_yoke": 6}

    elements: list[tuple[int, int, int, int, int]] = []  # (body, n1..n4 CCW)
    elem_index: dict[tuple[int, int], int] = {}  # (i, j) -> 1-based elem id
    for j in range(nyn - 1):
        lname = layers[yseg[j]][0]
        for i in range(nxn - 1):
            body = (
                pm_layer_body[xseg[i]] if lname == "pm" else layer_body[lname]
            )
            elements.append((
                body,
                node_id(i, j), node_id(i + 1, j),
                node_id(i + 1, j + 1), node_id(i, j + 1),
            ))
            elem_index[(i, j)] = len(elements)

    # Boundary 1: whole outer rectangle (Dirichlet A_z = 0).
    boundary: list[tuple[int, int, int, int]] = []  # (bc, parent, n1, n2)
    for i in range(nxn - 1):  # bottom, top
        boundary.append((1, elem_index[(i, 0)], node_id(i, 0),
                         node_id(i + 1, 0)))
        boundary.append((1, elem_index[(i, nyn - 2)], node_id(i, nyn - 1),
                         node_id(i + 1, nyn - 1)))
    for j in range(nyn - 1):  # left, right
        boundary.append((1, elem_index[(0, j)], node_id(0, j),
                         node_id(0, j + 1)))
        boundary.append((1, elem_index[(nxn - 2, j)], node_id(nxn - 1, j),
                         node_id(nxn - 1, j + 1)))

    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "mesh.header").write_text(
        f"{nxn * nyn} {len(elements)} {len(boundary)}\n"
        f"2\n404 {len(elements)}\n202 {len(boundary)}\n"
    )
    with open(mesh_dir / "mesh.nodes", "w") as f:
        n = 1
        for y in ys:
            for x in xs:
                f.write(f"{n} -1 {x:.12e} {y:.12e} 0.0\n")
                n += 1
    with open(mesh_dir / "mesh.elements", "w") as f:
        for eid, (body, n1, n2, n3, n4) in enumerate(elements, start=1):
            f.write(f"{eid} {body} 404 {n1} {n2} {n3} {n4}\n")
    with open(mesh_dir / "mesh.boundary", "w") as f:
        for bid, (bc, parent, n1, n2) in enumerate(boundary, start=1):
            f.write(f"{bid} {bc} {parent} 0 202 {n1} {n2}\n")

    return {
        "y_mid_gap": t_rotor_yoke + t_pm + t_gap / 2.0,
        "n_nodes": nxn * nyn,
        "n_elements": len(elements),
        "n_boundary": len(boundary),
    }
