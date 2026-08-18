"""Nitsche rotation spike — decision gate for R2.

Evaluates mesh-deformation rotation vs the R1 per-angle remeshing approach.
Script-only (not package code).

Plan reference: two-mass rotation plan §R2.

Decision criteria:
  - Adopt if accuracy within 5% AND clearly wins on noise or wall-clock cost
  - Adoption = fem version bump + mesh-prefix bump + reference_br recapture

Usage: .venv/bin/python scripts/nitsche_rotation_spike.py
"""

from __future__ import annotations

import time
from math import cos, gcd, pi, sin

import numpy as np


def _build_mesh(r_outer, r_stator, r_magnet, r_rotor,
                n_slots, slot_depth, slot_opening_width,
                n_p, maxh):
    """Build a combined (Glue'd) mesh with rotor/stator/slot regions."""
    from netgen.occ import Glue, OCCGeometry, WorkPlane
    from ngsolve import Mesh

    r_midgap = (r_stator + r_magnet) / 2

    # ----- stator yoke (outside the slots) -----
    r_slot_bottom = r_stator + slot_depth  # inrunner: slots go outward
    f_outer = WorkPlane().Circle(r_outer).Face()
    f_slot_bottom = WorkPlane().Circle(r_slot_bottom).Face()
    yoke = f_outer - f_slot_bottom
    yoke.faces.name = "yoke"

    # ----- stator teeth + slots -----
    f_stator = WorkPlane().Circle(r_stator).Face()
    stator_ring = f_slot_bottom - f_stator

    slot_half_angle = pi / n_slots * (slot_opening_width / (r_stator * 2 * pi / n_slots))
    R_big = 2 * r_outer
    slot_faces = []
    for k in range(n_slots):
        theta_k = 2 * pi * k / n_slots
        a_lo = theta_k - slot_half_angle
        a_hi = theta_k + slot_half_angle
        wp = WorkPlane()
        wp.MoveTo(0., 0.)
        wp.LineTo(R_big * cos(a_lo), R_big * sin(a_lo))
        wp.LineTo(R_big * cos(a_hi), R_big * sin(a_hi))
        wp.Close()
        wedge = wp.Face()
        f_o = WorkPlane().Circle(r_slot_bottom).Face()
        f_i = WorkPlane().Circle(r_stator).Face()
        slot_face = (wedge * f_o) - f_i
        slot_face.faces.name = "slot"
        slot_faces.append(slot_face)
        stator_ring = stator_ring - slot_face

    stator_ring.faces.name = "stator"

    # ----- airgap: split at midgap -----
    f_midgap = WorkPlane().Circle(r_midgap).Face()
    airgap_s = f_stator - f_midgap
    airgap_s.faces.name = "airgap_s"
    f_midgap2 = WorkPlane().Circle(r_midgap).Face()
    f_magnet = WorkPlane().Circle(r_magnet).Face()
    airgap_r = f_midgap2 - f_magnet
    airgap_r.faces.name = "airgap_r"

    # ----- PM ring (full ring, alpha_p=1) -----
    f_rotor = WorkPlane().Circle(r_rotor).Face()
    pm_ring = f_magnet - f_rotor
    pm_ring.faces.name = "pm"

    # ----- shaft -----
    shaft = WorkPlane().Circle(r_rotor).Face()
    shaft.faces.name = "shaft"

    all_faces = [yoke, stator_ring, *slot_faces, airgap_s, airgap_r, pm_ring, shaft]
    combined = Glue(all_faces)

    circ = 2 * pi * r_outer
    for e in combined.edges:
        if abs(e.mass - circ) < 1e-6 * circ:
            e.name = "outer"

    mesh = Mesh(OCCGeometry(combined, dim=2).GenerateMesh(maxh=maxh))
    return mesh, r_midgap


def _deformation_gf(mesh, r_midgap, phi):
    """Create a deformation GridFunction that rotates nodes at r <= r_midgap."""
    from ngsolve import GridFunction, VectorH1

    V_vec = VectorH1(mesh, order=1)
    gf = GridFunction(V_vec)
    cos_p, sin_p = cos(phi), sin(phi)
    gf_x = gf.components[0]
    gf_y = gf.components[1]
    for i in range(mesh.nv):
        mp = mesh.vertices[i].point
        x0, y0 = mp[0], mp[1]
        r = (x0**2 + y0**2) ** 0.5
        if r < r_midgap + 1e-8:
            gf_x.vec[i] = x0 * cos_p - y0 * sin_p - x0
            gf_y.vec[i] = x0 * sin_p + y0 * cos_p - y0
    return gf


def _solve_deformation(mesh, n_p, B_rem, mu_r_pm, phi, r_midgap):
    """Solve magnetostatic with rotor rotated via mesh deformation.

    The magnetization sign pattern uses reference-frame coordinates
    (pre-deformation) offset by phi, so it rotates WITH the rotor.
    """
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        IfPos,
        LinearForm,
        dx,
        grad,
        sqrt,
        x,
        y,
    )

    MU0 = 4e-7 * pi
    gf_def = None

    if abs(phi) > 1e-12:
        gf_def = _deformation_gf(mesh, r_midgap, phi)
        mesh.SetDeformation(gf_def)

    region_names = list(set(mesh.GetMaterials()))
    nu_cf = {}
    for name in region_names:
        if name == "pm":
            nu_cf[name] = 1 / (MU0 * mu_r_pm)
        elif name in ("yoke", "shaft", "stator"):
            nu_cf[name] = 1 / (MU0 * 5000.0)
        else:
            nu_cf[name] = 1 / MU0
    nu = mesh.MaterialCF(nu_cf)

    r_cf = sqrt(x ** 2 + y ** 2)
    cos_t, sin_t = x / r_cf, y / r_cf
    cr, si = cos_t, sin_t
    for _ in range(n_p - 1):
        cr, si = cr * cos_t - si * sin_t, cr * sin_t + si * cos_t

    # Rotate the magnetization sign pattern: cos(n_p(θ - φ)) via
    # cos(n_p·θ)·cos(n_p·φ) + sin(n_p·θ)·sin(n_p·φ).
    # With SetDeformation active, x/y are deformed (physical) coords.
    # The rotor nodes have moved by +φ, so θ_physical = θ_ref + φ.
    # We want the sign based on θ_ref = θ_physical - φ, i.e.
    # cos(n_p·(θ_physical - φ)).
    cos_np_phi = float(np.cos(n_p * phi))
    sin_np_phi = float(np.sin(n_p * phi))
    cr_rot = cr * cos_np_phi + si * sin_np_phi

    nu_pm = 1 / (MU0 * mu_r_pm)
    M_r = nu_pm * B_rem * IfPos(cr_rot, 1.0, -1.0)
    mx_cf = {name: (M_r * cos_t if name == "pm" else 0.0) for name in region_names}
    my_cf = {name: (M_r * sin_t if name == "pm" else 0.0) for name in region_names}
    Mx = mesh.MaterialCF(mx_cf)
    My = mesh.MaterialCF(my_cf)

    fes = H1(mesh, order=3, dirichlet="outer")
    u, v = fes.TnT()

    a = BilinearForm(nu * grad(u) * grad(v) * dx)
    a.Assemble()
    lf = LinearForm((Mx * grad(v)[1] - My * grad(v)[0]) * dx("pm"))
    lf.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

    return mesh, gfu, gf_def


def _arkkio_torque_from_gfu(mesh, gfu, r_inner, r_outer, n_theta=360):
    """Arkkio torque per unit stack length (radial average of Maxwell stress)."""
    thetas = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    d_theta = 2 * pi / n_theta
    radii = np.linspace(r_inner, r_outer, 5)
    MU0 = 4e-7 * pi

    tau_per_r = np.zeros(5)
    for j, r in enumerate(radii):
        integrand = np.zeros(n_theta)
        for i, th in enumerate(thetas):
            pt = mesh(cos(th) * r, sin(th) * r)
            grad_val = gfu.Deriv()(pt)
            dAdx, dAdy = grad_val[0], grad_val[1]
            B_r = dAdy * cos(th) - dAdx * sin(th)
            B_t = -(dAdy * sin(th) + dAdx * cos(th))
            integrand[i] = B_r * B_t / MU0
        tau_per_r[j] = r * np.sum(integrand) * d_theta

    return float(np.trapezoid(tau_per_r, radii) / (r_outer - r_inner))


def _run_deformation_sweep(geo_kw, n_p, B_rem, mu_r_pm, maxh, n_pts):
    """Run cogging sweep via mesh deformation (rotate rotor, keep stator)."""
    mesh, r_midgap = _build_mesh(**geo_kw, n_p=n_p, maxh=maxh)

    lcm_val = geo_kw["n_slots"] * 2 * n_p // gcd(geo_kw["n_slots"], 2 * n_p)
    period = 2 * pi / lcm_val
    angles = np.linspace(0, period, n_pts, endpoint=False)

    g = abs(geo_kw["r_stator"] - geo_kw["r_magnet"])
    margin = 0.05 * g
    r_ark_inner = geo_kw["r_magnet"] + margin
    r_ark_outer = geo_kw["r_stator"] - margin

    tau_list = []
    t_list = []
    for phi in angles:
        t0 = time.perf_counter()
        m, gfu, gf_def = _solve_deformation(mesh, n_p, B_rem, mu_r_pm, phi, r_midgap)
        # Evaluate torque WITH deformation still active
        tau = _arkkio_torque_from_gfu(m, gfu, r_ark_inner, r_ark_outer)
        if gf_def is not None:
            mesh.UnsetDeformation()
        t_list.append(time.perf_counter() - t0)
        tau_list.append(tau)

    return np.array(angles), np.array(tau_list), np.array(t_list), lcm_val


def _run_remesh_sweep(geo_kw, n_p, B_rem, mu_r_pm, mu_r_fe, maxh, n_pts):
    """Run cogging sweep via per-angle remeshing (R1 production path)."""
    from phasesweep.machines.geometry import inrunner
    from phasesweep.solvers.fem_field import arkkio_torque, solve_field_fem

    geo = inrunner(
        r_outer=geo_kw["r_outer"], r_stator=geo_kw["r_stator"],
        r_magnet=geo_kw["r_magnet"], r_rotor=geo_kw["r_rotor"],
        n_slots=geo_kw["n_slots"], slot_depth=geo_kw["slot_depth"],
        slot_opening_width=geo_kw["slot_opening_width"],
    )
    alpha_p = 1.0

    lcm_val = geo.n_slots * 2 * n_p // gcd(geo.n_slots, 2 * n_p)
    period = 2 * pi / lcm_val
    angles = np.linspace(0, period, n_pts, endpoint=False)

    g = abs(geo.r_stator - geo.r_magnet)
    margin = 0.05 * g
    r_ark_inner = geo.r_magnet + margin
    r_ark_outer = geo.r_stator - margin

    tau_list = []
    t_list = []
    for phi in angles:
        t0 = time.perf_counter()
        _, _, mesh_obj, gfu = solve_field_fem(
            geo=geo, n_p=n_p, B_rem=B_rem,
            mu_r_pm=mu_r_pm, mu_r_fe=mu_r_fe,
            maxh_fraction=maxh / geo.r_outer,
            n_theta=360, j_s=0.0, alpha_p=alpha_p,
            rotation=float(phi), return_full=True,
        )
        tau = arkkio_torque(mesh_obj, gfu, r_ark_inner, r_ark_outer, n_theta=360)
        t_list.append(time.perf_counter() - t0)
        tau_list.append(tau)

    return np.array(angles), np.array(tau_list), np.array(t_list), lcm_val


def main():
    # 6-slot / 4-pole synthetic geometry (matches test_cogging.py)
    geo_kw = dict(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
        n_slots=6, slot_depth=0.005, slot_opening_width=0.003,
    )
    n_p = 2
    B_rem = 1.2
    mu_r_pm = 1.05
    mu_r_fe = 5000.0
    maxh = 0.003
    n_pts = 12

    print("=" * 60)
    print("Nitsche Rotation Spike — Decision Gate R2")
    print("=" * 60)
    print(f"Geometry: 6-slot / 4-pole, r_outer={geo_kw['r_outer']*1e3:.1f} mm")
    print(f"Mesh: maxh={maxh*1e3:.1f} mm, {n_pts} angles/period")
    print()

    # --- Deformation path ---
    print("--- Path A: Mesh deformation (single mesh, rotate rotor nodes) ---")
    t0_a = time.perf_counter()
    try:
        ang_a, tau_a, t_a, lcm_a = _run_deformation_sweep(
            geo_kw, n_p, B_rem, mu_r_pm, maxh, n_pts)
        t_total_a = time.perf_counter() - t0_a
        print(f"  Total wall-clock: {t_total_a:.1f}s")
        print(f"  Per-angle: {t_a.mean():.2f}s (std {t_a.std():.2f}s)")
        print(f"  tau_pp: {(tau_a.max() - tau_a.min()):.4e} N·m/m")
        amps_a = np.abs(np.fft.rfft(tau_a)) / n_pts
        amps_a[1:n_pts // 2] *= 2
        dom_a = int(np.argmax(amps_a[1:])) + 1
        print(f"  Dominant bin: {dom_a} (order {dom_a * lcm_a}/rev)")
        print(f"  Mean (should be ~0): {np.mean(tau_a):.4e}")
        path_a_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        tau_a = None
        path_a_ok = False
        t_total_a = time.perf_counter() - t0_a

    print()

    # --- Remesh path (R1 production) ---
    print("--- Path B: Per-angle remeshing (R1 production path) ---")
    t0_b = time.perf_counter()
    try:
        ang_b, tau_b, t_b, lcm_b = _run_remesh_sweep(
            geo_kw, n_p, B_rem, mu_r_pm, mu_r_fe, maxh, n_pts)
        t_total_b = time.perf_counter() - t0_b
        print(f"  Total wall-clock: {t_total_b:.1f}s")
        print(f"  Per-angle: mean {t_b.mean():.2f}s (first {t_b[0]:.2f}s, rest {t_b[1:].mean():.2f}s)")
        print(f"  tau_pp: {(tau_b.max() - tau_b.min()):.4e} N·m/m")
        amps_b = np.abs(np.fft.rfft(tau_b)) / n_pts
        amps_b[1:n_pts // 2] *= 2
        dom_b = int(np.argmax(amps_b[1:])) + 1
        print(f"  Dominant bin: {dom_b} (order {dom_b * lcm_b}/rev)")
        print(f"  Mean (should be ~0): {np.mean(tau_b):.4e}")
        path_b_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        tau_b = None
        path_b_ok = False
        t_total_b = time.perf_counter() - t0_b

    # --- Comparison ---
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if path_a_ok and path_b_ok:
        pp_a = tau_a.max() - tau_a.min()
        pp_b = tau_b.max() - tau_b.min()
        if pp_b > 1e-15:
            pp_err = (pp_a - pp_b) / pp_b * 100
        else:
            pp_err = float("inf")
        print(f"  p-p agreement: {pp_err:+.1f}%")
        print(f"  Wall-clock ratio: {t_total_a / t_total_b:.2f}x "
              f"(deformation / remesh)")

        max_diff = np.max(np.abs(tau_a - tau_b))
        print(f"  Max point-wise difference: {max_diff:.4e} N·m/m")
        if pp_b > 1e-15:
            print(f"  Relative to p-p: {max_diff / pp_b * 100:.1f}%")

    print()
    print("=" * 60)
    print("DECISION")
    print("=" * 60)

    if not path_a_ok:
        print("  Deformation path FAILED — do not adopt.")
        print("  Per-angle remeshing (R1) remains the production path.")
    elif not path_b_ok:
        print("  Remesh path failed — cannot compare. Investigate.")
    else:
        speedup = t_total_b / t_total_a
        pp_err_abs = abs(pp_err)
        noise_ok = pp_err_abs < 5.0
        speed_ok = speedup > 1.5

        if noise_ok and speed_ok:
            verdict = "ADOPT"
            reason = f"accuracy OK ({pp_err:+.1f}%), {speedup:.1f}x faster"
        elif noise_ok:
            verdict = "MARGINAL — defer"
            reason = f"accuracy OK ({pp_err:+.1f}%) but only {speedup:.1f}x speedup"
        else:
            verdict = "DO NOT ADOPT"
            reason = f"accuracy {pp_err:+.1f}% exceeds 5% threshold"

        print(f"  Verdict: {verdict}")
        print(f"  Reason: {reason}")
        print()
        print("  Note: the single-mesh deformation approach has two")
        print("  compounding failure modes:")
        print("  (a) Shared nodes at the midgap distort stator-side")
        print("      elements when rotated, corrupting the solution;")
        print("  (b) NGSolve point evaluation (mesh(x,y)) searches the")
        print("      reference mesh, not the deformed one, so torque")
        print("      sampling lands at wrong physical coordinates.")
        print("  A true Nitsche sliding-interface (separate meshes,")
        print("  ContactBoundary coupling) avoids both, but requires a")
        print("  compound system assembly that NGSolve 6.2 does not")
        print("  natively support for 2D scalar problems.")

        if verdict != "ADOPT":
            print()
            print("  Next steps if adopting later:")
            print("  - fem version bump + mesh-prefix bump")
            print("  - reference_br recapture")
            print("  - Requires NGSolve compound-mesh or multi-mesh support")


if __name__ == "__main__":
    main()
