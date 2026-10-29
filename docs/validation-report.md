# Validation Report

Phase-sweep validates its electromagnetic models against published research,
manufacturer datasheets, and bench measurements across a fleet of motors
spanning different topologies, pole counts, and magnet grades.

Validation anchors live under `data/`; their executable comparisons live in
`tests/crossval/`. The tables below summarize the generated results without
maintaining separate fleet or test counts.

### Cross-Validation Results

Comparisons from the automated validation report are summarized below;
non-passing or qualified results are explained after the table.

| Motor | Quantity | Computed | Reference | Error | Tol | Result | Source |
|-------|----------|---------|-----------|-------|-----|--------|--------|
| CREATOR Case PMSM | MTPA angle at 0.15 A | 12.3° | 10° | 23.0% | 25% | PASS | arXiv:2501.15921 |
| CREATOR Case PMSM | MTPA angle at 0.30 A | 20.6° | 20° | 3.2% | 15% | PASS | arXiv:2501.15921 |
| CREATOR Case PMSM | rated torque (Nm) | 0.107 | ≥ 0.10 | +6.8% | bound | PASS | arXiv:2501.15921 |
| Belkhadir ER-PMSM | B_ag fundamental (T) | 1.026 | 0.963 | 6.6% | 10% | PASS | IECON 2023 |
| Belkhadir ER-PMSM | back-EMF peak (V) | 67.5 | 67.5 | 0.0% | 8% | by construction (fitted `psi_f`) | IECON 2023 |
| Belkhadir ER-PMSM | flux linkage (Wb) | 0.097 | 0.100 | 3.0% | 30% | PASS | IECON 2023 |
| Belkhadir ER-PMSM | B_ag (FEM, T) | 1.020 | 0.963 | 6.0% | 10% | PASS | IECON 2023 |
| Belkhadir ER-PMSM | rated torque (Nm) | 25.1 | ≥ 24 | +4.5% | bound | PASS | IECON 2023 |
| Deylami Fan | B_ag peak (FEM, T) | 0.408 | 0.49 | 16.7% | 25% | PASS | TechRxiv |
| Deylami Fan | B_ag peak (analytical, T) | 0.365 | 0.49 | 25.5% | 25% | FAIL | TechRxiv |
| 14 mm Outrunner (Steel) | back-EMF (V) | 1.97 | 1.68 | 16.9% | 1% | FAIL | bench (ODrive) |

### Notes on Partial Results

**Belkhadir back-EMF:** `psi_f = 0.0977 Wb` in
`motors/belkhadir_outrunner.toml` is fitted to this 67.5 V reading, and the
analytical model reports `backemf_fundamental` from `psi_f` when it is set
(`backemf_source = "psi_f"`), so the 0.0% row is a consistency check, not a
validation. With `psi_f` unset, the field model's own back-EMF is 67.0 V
(−0.7%). The flux-linkage row is computed from `B_rem` and does not use
`psi_f`.

**Deylami Fan analytical B_ag_peak:** The analytical model slightly exceeds
tolerance for this simplified slot geometry, while the FEM model passes. The
published reference itself is from a simplified 2D ANSYS Maxwell model, not a
physical measurement — both values are model-vs-model comparisons.

**14 mm Outrunner (Steel) back-EMF:** The measured speed sweep supports
measurement consistency. Applying the computed end-effect correction reduces
the discrepancy; calibration attributes most of the remainder to nominal
remanence exceeding its effective value. This motor is the framework's
flagship calibration case.

## Analytical vs FEM Agreement

The analytical (Zhu & Howe subdomain) and nonlinear FEM (NGSolve damped
Newton by default, with Picard startup/fallback) solvers agree closely on
the airgap flux density fundamental across the applicable reference
configurations:

| Motor | Analytical B₁ (T) | FEM B₁ (T) | Delta |
|-------|-------------------|------------|-------|
| 14 mm Outrunner (Steel) | 1.004 | 0.998 | 0.5% |
| Belkhadir 22p/24s ER-PMSM | 1.026 | 1.020 | 0.6% |
| Deylami 8p/12s Cooling Fan | 0.453 | 0.445 | 1.6% |
| FEMM LRK 14p/12s | 0.950 | 0.945 | 0.5% |
| Zhu & Howe 8-pole inrunner | 0.565 | 0.564 | 0.2% |
| Zhu & Howe 8-pole outrunner | 0.552 | 0.552 | 0.0% |

The analytical model is substantially faster than FEM while maintaining
the stated agreement bounds on the fundamental.

CREATOR Case PMSM is absent from this table: its rotor uses
`magnet_mount = "inset"` (interpole iron), which the Zhu & Howe
subdomain model does not support — the analytical solver rejects the
motor by design, and CREATOR is validated on the nonlinear FEM path
only.

## Per-Motor Details

### CREATOR Case PMSM

4-pole inrunner from the TU Graz CREATOR open dataset (arXiv:2501.15921).
Validated on MTPA angle and rated torque. The rotor uses inset magnets
(interpole iron), so only the nonlinear FEM solver applies — the earlier
analytical back-EMF comparison was retired when the magnet mount was
corrected; the measured back-EMF capture remains in the anchor set as an
import gate (that JSON is generated locally by `scripts/fetch_creator_dataset.py`
and is not shipped). (Iron loss excluded: `B_core` was
calibrated against the same measured loss curve — see CHANGELOG 0.3.0.)

![Airgap flux density waveform — CREATOR Case PMSM](images/validation/br_waveform_creator_case_pmsm.png)

![Validation deviations — CREATOR Case PMSM](images/validation/deviation_creator_case_pmsm.png)

![MTPA angle vs current — CREATOR Case PMSM](images/validation/mtpa_gamma_creator_case_pmsm.png)

![FEM field solution — CREATOR Case PMSM](images/validation/field_creator_case_pmsm.png)

### Belkhadir 22p/24s ER-PMSM

22-pole / 24-slot external-rotor PMSM from Belkhadir et al. (IECON 2023).
All rows pass, but none is an independent measured comparison. The air-gap
reference (0.963 T) is the paper's flat-top formula value (0.776 T)
converted to a fundamental with this project's own (4/π)·sin(π·α_p/2)
factor; the back-EMF row matches by construction (fitted `psi_f`); the
flux-linkage tolerance is 30 % because the fundamental model misses this
fractional-slot topology's sub-harmonic coupling; and rated torque is a
one-sided bound.

![Airgap flux density waveform — Belkhadir ER-PMSM](images/validation/br_waveform_belkhadir_22p_24s_er_pmsm.png)

![Validation deviations — Belkhadir ER-PMSM](images/validation/deviation_belkhadir_22p_24s_er_pmsm.png)

![FEM field solution — Belkhadir ER-PMSM](images/validation/field_belkhadir_22p_24s_er_pmsm.png)

### Deylami 8p/12s Cooling Fan

8-pole / 12-slot outrunner from Deylami et al. (TechRxiv). FEM passes the
airgap flux comparison; the analytical model slightly exceeds the tolerance.

![Airgap flux density waveform — Deylami Fan](images/validation/br_waveform_deylami_8p_12s_cooling_fan.png)

![FEM B_r with published peak marker — Deylami Fan](images/validation/br_peak_deylami_8p_12s_cooling_fan.png)

![Validation deviations — Deylami Fan](images/validation/deviation_deylami_8p_12s_cooling_fan.png)

### 14 mm Outrunner (Steel)

12-pole / 9-slot outrunner (bench-measured with RTB2004 oscilloscope).
Most of the remaining, end-effect-corrected back-EMF discrepancy reflects the gap
between nominal and calibrated effective remanence, not a measurement error.

![Airgap flux density waveform — 14 mm Outrunner (Steel)](images/validation/br_waveform_outrunner_14mm_steel.png)

![Validation deviations — 14 mm Outrunner (Steel)](images/validation/deviation_outrunner_14mm_steel.png)

![FEM field solution — 14 mm Outrunner (Steel)](images/validation/field_outrunner_14mm_steel.png)

## Additional Validation Coverage

The following motors have validation anchors and cross-validation tests
(`pytest tests/crossval/`) but are not included in the report
figures above:

| Motor | Source | Quantities Tested |
|-------|--------|-------------------|
| Toyota Prius 2004 IPM | ORNL/TM-2004/185 | stall torque ≥ 340 Nm, MTPA angle |
| Awan 2.2 kW IPM | Aalto dissertation | saliency ratio, rated torque ≥ 14 Nm |
| ETEL TMB (5 sizes) | ETEL datasheet | `psi_f` fitted from Ku (back-EMF equal by construction, not a validation); tested: base/max-speed envelope bands, copper loss vs datasheet Pc and R_th budget, reverse-salient MTPA, stall-torque bounds |
| Rexroth MS2N04 | Bosch Rexroth manual | `psi_f` fitted from KE (back-EMF equal by construction, not a validation); tested: KE/KM convention consistency, rated-point voltage feasibility, envelope corner |
| Tecnotion QTR-A-105 | Tecnotion datasheet | `psi_f` fitted from Ke (back-EMF equal by construction, not a validation); tested: hot-R reconstruction from Pc, max speed at two DC-bus voltages |
| JMT 1806 2400KV | bench measurement | back-EMF speed sweep (8 speeds) |
| Parviainen 5 kW AFPM | LUT D.Sc. thesis | back-EMF 211 V (`analytical_axial`); `fem_axial` bracket only — see below |
| CREATOR Case IM | TU Graz CREATOR dataset (DOI 10.3217/kh3v7-dhn98) | locked-rotor current (45 points, ≤2.5%), torque-slip (hot-parameter agreement; +34% at 20 °C pinned as a finding), nameplate, no-load friction/windage. Machine-level iron loss is not validated: the M800-50A table fit is pinned, but the stator-core estimate under-predicts the measured no-load loss by ~3.9× (pinned shortfall) |
| NWPU 12 kW Halbach SPM | WEVJ 16(8):413 (2025) | `psi_f` fitted from the measured 75.2 V line-peak back-EMF at 1000 rpm (reproduces it by construction); tested: Halbach vs radial FEM B1 and THD, and an arithmetic consistency check of the paper's own numbers (the fitted `psi_f` reproduces its FEM back-EMF at 5000 rpm within 5%; no phasesweep model runs in that check) (the paper's own FEM 74.4 V / analytical 76.4 V at 1000 rpm cited for context) |
| Jilin 5.5 kW SynRM | PMC11679518 (2024) | reluctance-only MTPA through rated_torque/torque_speed/thermal_duty/drive_sim |
| BMW i3 2016 IPM | Gierczynski & Grzesiak 2021 (FEM flux map), Burress 2017 (ORNL) | numeric MTPA on the flux map within 1% of the source torque sheet; hot peak torque bracketed by the unknown map temperature; `motor_thermal` fit to the 7000 rpm continuous test is a pin of the fit, not a validation |

### Axial-Flux FEM (`fem_axial`)

`fem_axial` is verified by code-to-code and limit comparison rather
than by a measured back-EMF gate:

| Check | Result |
|---|---|
| Mirror BC vs true periodic identification | 2–3·10⁻⁵ on B1 |
| Slotless plane vs analytical layer model | 0.06% |
| Cross-code vs Elmer (independent solver) | 0.39% at the motor's mu_r_fe = 1000; 0.001% with ideal iron |
| 1D magnetic-circuit limit | 0.06% |

It is **not** gated on the Parviainen E_PM: that anchor does not
publish its slot opening, and the explicit-slot model spans
192.6–234.3 V across the plausible slot geometry, so the anchor cannot
discriminate. The crossval test asserts that the measured value falls
inside that bracket. `analytical_axial` remains the E_PM-gated axial
model.

### Radial FEM Cross-Code (Elmer)

The radial `fem` model — the project's primary field solver — is
cross-checked against ElmerFEM (independent code, mesh, basis, and
nonlinear linearization) on surface-PM inrunner cases covering
smooth bore, discrete magnet arcs, nonlinear BH, slots, and armature
current sheet (`scripts/elmer_radial/`, pinned JSON re-checked by
`tests/crossval/test_elmer_radial_pinned.py`):

| Check | Result |
|---|---|
| Slotless and nonlinear cases, B₁ at r_ag | 0.07–0.11% |
| Slotted cases (open-circuit and j_s) | 0.17–0.21% |
| Linear slotless cases vs Zhu/Howe/Chan series | ≤ 0.2% |

Rotation/cogging is excluded from the cross-check: the mean field
agrees, but the sub-0.1% ripple over a slot pitch is discretization
noise in both codes at these mesh levels.

## How Validation Works

Each validation anchor is a JSON file in `data/<motor>/` containing a
published or measured value, its source, tolerance, and comparison method:

- **Delta**: direct percentage comparison (most quantities)
- **Bound**: inequality check — e.g., rated torque ≥ published rating
- **Curve**: extract a scalar from a computed waveform (e.g., max of B_r)
  and compare to a published value

The `phasesweep.validation.crossval` module computes deviations, applies
tolerances, and produces structured pass/fail diagnostics. Cross-validation
tests are assigned to the CI lanes defined by the workflow configuration.

## Reproducing This Report

```bash
pip install -e ".[all]"
uv run python scripts/validation_report.py --no-sensitivity --workers 4
```

To include sensitivity analysis, omit `--no-sensitivity`:

```bash
uv run python scripts/validation_report.py --workers 4
```

Output goes to `output/validation_report/` (gitignored).
