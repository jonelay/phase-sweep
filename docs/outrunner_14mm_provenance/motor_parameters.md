# 14 mm Outrunner — Motor Parameter Provenance

Bench-measurement provenance for the `outrunner_14mm_steel` validation
motor. Consolidated from lab notes (Oct 2025 -- Mar 2026); these are the
raw measurements and cross-checks behind the values committed in the
motor TOML.

Derived geometry is computed in [`derive_geometry.py`](derive_geometry.py)
(run it to reproduce the radii, air gap, and `alpha_p` from the raw
caliper/micrometer/gauge-pin numbers below).

## Topology

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | SPMSM, outrunner | — |
| Poles / pole pairs | 12 / 6 | — |
| Slots | 9 | — |
| Winding | Double-layer concentrated | — |
| Turns per coil (N) | 8 (bottom layer; top may have 1–2 fewer) | Measurement |
| Coils per phase | 3 | — |
| k_w | 0.866 (ideal 9s/12p: kp=sin(2π/3), kd=1; unequal layer turns put true effective k_w slightly below) | Ideal 9s/12p |

## Electrical

| Parameter | Value | Notes |
|-----------|-------|-------|
| R_s (phase) | 139 mΩ | LCR 1 kHz L-L avg 277.6 mΩ / 2 (2026-03-16) |
| L_d = L_q (phase) | 12.5 µH | L_self − M = 9.0 + 3.5 µH; confirmed by L-N + L-L decomposition |
| L_self (phase) | 9.0 µH | LCR 1 kHz L-N avg (8.6/9.0/9.5 µH per phase) |
| M (mutual) | −3.5 µH | From L_LL = L_a + L_b − 2M |
| ψ_f | 0.268 mWb | Steel rotor, 3-channel back-EMF speed sweep, linear fit R²=0.99997 (2026-03-18) |
| Ke | 1.607 mV/(rad/s) | Same 3-channel fit; ψ_f×n_p round-trips to 1.608 from rounded ψ_f |
| Kt | 2.412 mNm/A | FOC, from 1.5 × n_p × ψ_f (amplitude-invariant Park) |
| THD | < 1.6% | Clean sinusoidal back-EMF |
| Drive voltage | 24 V | Stator-on-stick bench setup |
| Drive max current | 3.2 A | Bench supply current limit; 2.0 A shutdown threshold during testing |
| No-load current (rotor only, 120 rps) | 0.22–0.35 A | Varies by direction |
| Ambient temperature | 21.5 °C | During back-EMF test (2026-03-16) |

## Geometry — Stator

| Parameter | Value | Notes |
|-----------|-------|-------|
| Stator OD | 13.9 mm (TOML) / 14.06 mm (.5535" gauge) | Two measurements; gauge may include tooth tips |
| Stator ID | 6.0 mm | Nominal drawing dimension |
| Stack length | 7.4 mm lamination stack; 7.0 mm active (magnet-limited) | — |
| Slot depth | 3.0 mm | Caliper estimate |
| Slot width ratio | 0.165 at OD | Caliper estimate |
| Stator pole runout | 0.001" (0.025 mm) | Supported on V-blocks |

## Geometry — Rotor

Steel rotor shell with press-fit bearings.

| Parameter | Steel rotor | Units |
|-----------|-------------|-------|
| Rotor OD | 19.33 | mm |
| Wall thickness | 0.90 | mm |
| Magnet thickness | 1.287 | mm |
| Gauge pin (max fit) | .589 (14.96) | in (mm) |

The gauge pin measures the magnet-face bore diameter. Cross-check:
OD − 2×(wall + magnet) = 19.33 − 2×(0.90 + 1.287) = **14.96 mm** ✓

## Geometry — Air gap (derived)

Computed by `derive_geometry.py` from the radial stack
(rotor OD → wall → magnet → stator OD):

| Rotor variant | Air gap (mm) | Magnet bore dia (mm) | α_p |
|---------------|--------------|----------------------|-----|
| Steel | 0.528 | 14.956 | 0.635 |

Gauge-pin cross-check (steel): derived bore 14.956 mm vs gauge .589" =
14.961 mm — **0.005 mm agreement**.

## Geometry — Runout and clearance

| Measurement | Value | Notes |
|-------------|-------|-------|
| Bearing runout (2× in ring gauge) | 0.003" (0.076 mm) | Gauge pin in V-block |
| Bare rotor runout | 0.0015" (0.038 mm) | V-block |
| Steel rotor + bearing deflection | 0.002" (0.051 mm) | Gentle axial push |

## Direction-dependent current asymmetry

Multiple tests show consistent direction-dependent no-load current (e.g.,
0.26 A forward vs 0.35 A reverse for the steel rotor). Possible contributors:

- Rotor eccentricity (static or dynamic)
- Asymmetric magnetization (non-uniform magnet placement or grade)
- Mechanical assembly: bearing preload asymmetry, rotor axial position

The effect sets a floor on how well a symmetric model can match test data,
and could help validate eccentricity / non-uniform-air-gap modeling if added.

## Materials

| Parameter | Value | Notes |
|-----------|-------|-------|
| Magnet grade | NdFeB N52 (nominal); measured ~N42–N45 | Modestly below the N52 spec (gaussmeter peak-hold + back-EMF); not severely |
| B_rem | 1.45 T model input (N52 ideal); effective ~N42–N45 | Deficit folded into the correction factor, not B_rem itself |
| µ_r,PM | 1.05 | — |
| Magnet shape | Flat rectangular on curved rotor ID | Non-uniform air gap across width |
| Steel grade | Unknown | Generic soft iron assumed |

## Flux-linkage model comparison

Zhu & Howe (phasesweep) over-predicts ψ_f by +27% relative to the measured
0.268 mWb (after k_w = 0.866, Carter correction, square-wave magnetization).
The magnet grade is measured ~N42–N45 — modestly below the N52 spec.
