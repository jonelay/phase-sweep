# Validation Report

Phase-sweep compares its electromagnetic models with published research,
manufacturer datasheets, and bench measurements. Reference data lives under
`data/`, with executable comparisons in `tests/`.

## Cross-Validation Results

The table summarizes the automated report. “Measured” means physical test or
bench data, “published” means a rating or other reported scalar, “computed”
means a reference produced by FEM or a formula, and “fitted” means the model
input was derived from the same reference. The notes below identify bounds and
comparisons that are not independent validation.

| Motor | Quantity | Computed | Reference | Error | Tol | Result | Evidence / source |
|-------|----------|---------|-----------|-------|-----|--------|--------|
| CREATOR Case PMSM | MTPA angle at 0.15 A | 12.3° | 10° | 23.0% | 25% | PASS | computed (published FEM-derived), arXiv:2501.15921 |
| CREATOR Case PMSM | MTPA angle at 0.30 A | 20.6° | 20° | 3.2% | 15% | PASS | computed (published FEM-derived), arXiv:2501.15921 |
| CREATOR Case PMSM | rated torque (Nm) | 0.107 | ≥ 0.10 | +6.8% | bound | PASS | published rating, arXiv:2501.15921 |
| Belkhadir ER-PMSM | B_ag fundamental (T) | 1.026 | 0.963 | 6.6% | 10% | PASS | computed (published formula), IECON 2023 |
| Belkhadir ER-PMSM | back-EMF peak (V) | 67.5 | 67.5 | 0.0% | 8% | by construction (fitted `psi_f`) | measured, IECON 2023 |
| Belkhadir ER-PMSM | B_ag (FEM, T) | 1.020 | 0.963 | 6.0% | 10% | PASS | computed (published formula), IECON 2023 |
| Belkhadir ER-PMSM | rated torque (Nm) | 25.1 | ≥ 24 | +4.5% | bound | PASS | published rating, IECON 2023 |
| Deylami Fan | B_ag peak (FEM, T) | 0.408 | 0.49 | 16.7% | 25% | PASS | computed (published FEM), TechRxiv |
| Deylami Fan | B_ag peak (analytical, T) | 0.365 | 0.49 | 25.5% | 25% | FAIL | computed (published FEM), TechRxiv |
| 14 mm Outrunner (Steel) | back-EMF (V) | 1.97 | 1.68 | 16.9% | 1% | FAIL | measured (bench, ODrive) |

## Notes on Partial Results

**Belkhadir back-EMF:** `psi_f = 0.0977 Wb` in
`motors/belkhadir_outrunner.toml` is fitted to this 67.5 V reading, and the
analytical model reports `backemf_fundamental` from `psi_f` when it is set
(`backemf_source = "psi_f"`), so the 0.0% row is a consistency check, not a
validation. With `psi_f` unset, the field model's own back-EMF is 67.0 V
(−0.7%). No test asserts the previously reported flux-linkage PASS, so that
row is omitted.

**Deylami Fan analytical B_ag_peak:** The analytical model exceeds tolerance
for this simplified slot geometry, while the FEM model passes. The
published reference itself is from a simplified 2D ANSYS Maxwell model, not a
physical measurement - both values are model-vs-model comparisons.

**14 mm Outrunner (Steel) back-EMF:** The 1.97 V prediction already
includes the computed end-effect factor (`k_end = 0.918`) and uses the
nominal N52 remanence of 1.45 T. The measured voltage is linear in speed
(R² = 0.99997), so the residual +17 % is model bias, not measurement
scatter; calibration absorbs it into an effective remanence of 1.24 T
(1.14 T without the end-effect factor). This is the project's main
calibration example; the motor TOML records the full accounting.

## Analytical vs FEM Agreement

The table compares the computed airgap flux density fundamental from the
analytical (Zhu & Howe subdomain) and nonlinear FEM (NGSolve damped Newton
by default, with Picard startup/fallback) solvers:

| Motor | Analytical B₁ (T) | FEM B₁ (T) | Delta |
|-------|-------------------|------------|-------|
| 14 mm Outrunner (Steel) | 1.004 | 0.998 | 0.5% |
| Belkhadir 22p/24s ER-PMSM | 1.026 | 1.020 | 0.6% |
| Deylami 8p/12s Cooling Fan | 0.453 | 0.445 | 1.6% |
| FEMM LRK 14p/12s | 0.950 | 0.945 | 0.5% |
| Zhu & Howe 8-pole inrunner | 0.565 | 0.564 | 0.2% |
| Zhu & Howe 8-pole outrunner | 0.552 | 0.552 | 0.0% |

CREATOR Case PMSM is absent from this table: its rotor uses
`magnet_mount = "inset"` (interpole iron), which the Zhu & Howe
subdomain model does not support. The analytical solver rejects this motor;
CREATOR's field comparisons use the nonlinear FEM path only.

## Per-Motor Details

### CREATOR Case PMSM

4-pole inrunner from the TU Graz CREATOR open benchmark
(DOI 10.3217/sns1d-77m43). The table above compares the analytical MTPA angles
with FEM-derived values reported in the paper and checks rated torque against
a one-sided published-rating bound. The repository disagrees on the paper
section containing the MTPA values, so no section number is asserted here.
The rotor uses inset magnets, so only the nonlinear FEM solver applies. The
measured back-EMF capture is an import gate, not a validation (the motor's
`psi_f` is the benchmark's own), and iron loss is excluded because
`B_core` was calibrated against the same measured loss curve. The
benchmark JSON is fetched locally by `scripts/fetch_creator_dataset.py`
and is not shipped.

![Airgap flux density waveform - CREATOR Case PMSM](images/validation/br_waveform_creator_case_pmsm.png)

![Validation deviations - CREATOR Case PMSM](images/validation/deviation_creator_case_pmsm.png)

![MTPA angle vs current - CREATOR Case PMSM](images/validation/mtpa_gamma_creator_case_pmsm.png)

![FEM field solution - CREATOR Case PMSM](images/validation/field_creator_case_pmsm.png)

### Belkhadir 22p/24s ER-PMSM

22-pole / 24-slot external-rotor PMSM from Belkhadir et al. (IECON 2023).
All displayed rows pass, but none is an independent measured comparison.

- The air-gap reference (0.963 T) is the paper's flat-top formula value
  (0.776 T) converted to a fundamental with this project's own
  (4/π)·sin(π·α_p/2) factor.
- The back-EMF row matches by construction (fitted `psi_f`).
- Rated torque is a one-sided bound.

![Airgap flux density waveform - Belkhadir ER-PMSM](images/validation/br_waveform_belkhadir_22p_24s_er_pmsm.png)

![Validation deviations - Belkhadir ER-PMSM](images/validation/deviation_belkhadir_22p_24s_er_pmsm.png)

![FEM field solution - Belkhadir ER-PMSM](images/validation/field_belkhadir_22p_24s_er_pmsm.png)

### Deylami 8p/12s Cooling Fan

8-pole / 12-slot outrunner from Deylami et al. (TechRxiv). FEM passes the
airgap flux comparison; the analytical model exceeds the tolerance.

![Airgap flux density waveform - Deylami Fan](images/validation/br_waveform_deylami_8p_12s_cooling_fan.png)

![FEM B_r with published peak marker - Deylami Fan](images/validation/br_peak_deylami_8p_12s_cooling_fan.png)

![Validation deviations - Deylami Fan](images/validation/deviation_deylami_8p_12s_cooling_fan.png)

### 14 mm Outrunner (Steel)

12-pole / 9-slot outrunner (bench-measured with RTB2004 oscilloscope).
Most of the remaining, end-effect-corrected back-EMF discrepancy reflects the gap
between nominal and calibrated effective remanence, not a measurement error.

![Airgap flux density waveform - 14 mm Outrunner (Steel)](images/validation/br_waveform_outrunner_14mm_steel.png)

![Validation deviations - 14 mm Outrunner (Steel)](images/validation/deviation_outrunner_14mm_steel.png)

![FEM field solution - 14 mm Outrunner (Steel)](images/validation/field_outrunner_14mm_steel.png)

## Additional Automated Checks

The following motors have automated comparisons or integrity checks but are
not included in the report figures above. Some are independent validation
anchors; others exercise formulas, compare two computations, or pin a fitted
dataset. The wording below states which kind each test actually performs.

| Motor | Evidence | What the tests assert |
|-------|----------|-----------------------|
| Toyota Prius 2004 IPM | measured value reported by ORNL | one-sided stall-torque bound ≥ 340 Nm; MTPA angle is only a model plausibility/regression band |
| Awan 2.2 kW IPM | published parameters and rating | parameter saliency ratio; MTPA torque ≥ the 14 Nm continuous rating |
| ETEL TMB (5 sizes) | published datasheets; fitted `psi_f` | envelope and stall-torque bounds, copper loss vs datasheet Pc, and reverse-salient MTPA checks; back-EMF is not independently tested because `psi_f` comes from Ku |
| Rexroth MS2N04 | published manual; fitted `psi_f` | KE/KM convention and rated-point voltage feasibility; `psi_f` comes from KE |
| Tecnotion QTR-A-105 | published datasheet; fitted `psi_f` | hot-R reconstruction from Pc and max speed at two bus voltages; `psi_f` comes from Ke |
| JMT 1806 2400KV | measured bench dataset | dataset integrity, speed linearity, and electrical-convention checks only; no motor model is compared with the sweep |
| Parviainen 5 kW AFPM | measured thesis value | `analytical_axial` back-EMF within 5 % of 211 V; `fem_axial` only brackets 211 V over assumed slot geometry |
| CREATOR Case IM | measured dataset and published nameplate; circuit parameters fitted from the same tests | locked-rotor current consistency (45 points, ≤2.5 %), measured locked-rotor torque bands, nameplate bounds, and friction/windage data checks. Iron loss is not validated: the stator-core estimate under-predicts measured no-load loss ~3.9× (pinned). Details: `data/creator_case_im/README.md` |
| NWPU 12 kW Halbach SPM | measured back-EMF fitted into `psi_f`; computed FEM comparison | fitted back-EMF consistency plus the inequalities Halbach FEM B1 > radial FEM B1 by 2 % and Halbach FEM THD < radial FEM THD; no external B1 or THD reference is asserted |
| Jilin 5.5 kW SynRM | published circuit parameters; closed-form checks | reluctance-only MTPA and runner outputs are checked against the same closed-form equations; no runner output is compared with an external result |
| BMW i3 2016 IPM | computed (published FEA map and torque sheet) plus published ratings | numeric MTPA agrees within 5 % with the torque sheet derived from the same FEA; separate torque-speed tests assert published peak-torque and peak-power bounds; the `motor_thermal` fit is a pin, not a validation |

### Axial-Flux FEM (`fem_axial`)

`fem_axial` is verified by code-to-code and limit comparison rather
than by a measured back-EMF gate:

| Check | Result |
|---|---|
| Mirror BC vs true periodic identification | B1 and field L2 error < 0.1% |
| Slotless plane vs analytical layer model | B1 error < 2% |
| Cross-code vs pinned Elmer results (independent solver) | B1 error ≤ 1% at the motor's mu_r_fe = 1000 and with ideal iron |
| 1D magnetic-circuit limit | error < 1% |

It is **not** gated on the Parviainen E_PM: that anchor does not
publish its slot opening, and the explicit-slot model spans
192.6–234.3 V across the plausible slot geometry, so the anchor cannot
discriminate. The crossval test asserts that the measured value falls
inside that bracket. `analytical_axial` remains the E_PM-gated axial
model.

### Radial FEM Cross-Code (Elmer)

The radial `fem` model is cross-checked against ElmerFEM (independent code,
mesh, basis, and nonlinear linearization) on surface-PM inrunner cases covering
smooth bore, discrete magnet arcs, nonlinear BH, slots, and armature
current sheet (`scripts/elmer_radial/`, pinned JSON re-checked by
`tests/crossval/test_elmer_radial_pinned.py`):

| Check | Result |
|---|---|
| Slotless and nonlinear cases, B₁ at r_ag | Elmer vs pinned NGSolve FEM error ≤ 0.5% |
| Slotted cases (open-circuit and j_s) | Elmer vs pinned NGSolve FEM error ≤ 1% |
| Linear slotless cases vs Zhu/Howe/Chan series | Elmer error ≤ 0.5% |

Rotation/cogging is excluded from the cross-check: the mean field
agrees, but the sub-0.1% ripple over a slot pitch is discretization
noise in both codes at these mesh levels.

## How Validation Works

`data/anchors.toml` is the registry of independent validation anchors and the
tests that exercise them. Supporting JSON files under `data/<motor>/` contain
measured or published references, their sources, tolerances, and comparison
methods. Computed references, fitted-value checks, and regression pins are not
independent anchors merely because they have automated tests.

- **Delta**: direct percentage comparison (most quantities)
- **Bound**: inequality check - e.g., rated torque ≥ published rating
- **Curve**: extract a scalar from a computed waveform (e.g., max of B_r)
  and compare to a published value

The `phasesweep.validation.crossval` module computes deviations, applies
tolerances, and produces structured pass/fail diagnostics.

## Reproducing This Report

```bash
uv sync --extra all
uv run python scripts/validation_report.py --no-sensitivity --workers 4
```

To include sensitivity analysis, omit `--no-sensitivity`:

```bash
uv run python scripts/validation_report.py --workers 4
```

Output goes to `output/validation_report/` (gitignored).
