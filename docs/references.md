# Physics References

Canonical sources for the models and data used in phasesweep. Organized by
topic. Where a result is already implemented, the relevant function is noted.

---

## Air-Gap Field Analysis

### Zhu, Howe & Chan (2002) — **primary reference** (`zhu_howe_Br`)
**Zhu, Z. Q., Howe, D., & Chan, C. C. (2002).** "Improved Analytical Model
for Predicting the Magnetic Field Distribution in Brushless Permanent-Magnet
Machines." *IEEE Trans. Magn.*, 38(1), pp. 229–238.

Builds on the 1993 four-part series (same first two authors) with improved
treatment of both internal (inrunner) and external (outrunner) rotor
topologies. Our `zhu_howe_Br()` implements the 2002 model: open-circuit,
PM-only field with square-wave radial magnetization and infinite iron
permeability at boundary surfaces. Solves the 2×2 interface-matching system
numerically.

### Zhu & Howe (1993) — original four-part series
Four-part series, all in *IEEE Transactions on Magnetics*, 29(1), Jan. 1993.

- **Part I — Open-Circuit Field** (magnet source only)
  https://doi.org/10.1109/20.195557 — pp. 124–135
- **Part II — Armature-Reaction Field**
  https://doi.org/10.1109/20.195558 — pp. 136–142
- **Part III — Effect of Stator Slotting**
  https://doi.org/10.1109/20.195559 — pp. 143–151
- **Part IV — Magnetic Field on Load**
  https://doi.org/10.1109/20.195560 — pp. 152–158

Part III (slotting) is the primary reference for the Carter factor
correction. Eqs 15–17 define k_c using the effective airgap
g' = g + h_m/μ_r and the modified stator radius R_se. It is also the
natural reference for the cogging-torque sweep.

## Axial-Flux Field Analysis

### Parviainen (2005) — **primary reference** (`analytical_axial`)
**Parviainen, A. (2005).** "Design of Axial-Flux Permanent-Magnet
Low-Speed Machines and Performance Comparison Between Radial-Flux and
Axial-Flux Machines." D.Sc. thesis, Lappeenranta University of
Technology, Acta Universitatis Lappeenrantaensis 208.
ISBN 952-214-029-5. Open access:
https://web.mit.edu/kirtley/binlustuff/literature/electric%20machine/designOfAxialFluxPMM.pdf

Chapter 2 defines the quasi-3D approach (unroll the annulus at
computation-plane radii into Cartesian linear machines, Eqs. 2.1–2.5),
the layer-model air-gap field (Eq. 2.18), Carter applied to the
physical gap (Eq. 2.11, g' = k_C·g — the magnet is carried explicitly
by the layer model's coth term), the Qu PM-leakage adoption
(Eqs. 2.13–2.17, k_σPM,add = 0.95 reconciling analytical with 3D FEA),
and the back-EMF (Eq. 2.28). Chapter 3 documents the 5 kW / 300 rpm
1R2S prototype used as the validation anchor
(`data/parviainen_afpm/`). `analytical_axial` implements the
single-plane (r_mean) case; `fem_axial` implements the multi-plane
case with the planes solved by FEM rather than the layer expression.

### Chung & Kim (2002) / Hanselman (2006) — Cartesian layer field
Source form of thesis Eq. (2.18): per-harmonic air-gap flux density of
a linear PM machine with iron boundaries on both sides — equivalent to
the hyperbolic transfer
B_n = b_src / (cosh(m·g') + μ_r·sinh(m·g')·coth(m·l_PM)), whose m → 0
limit is the 1D magnetic circuit b_src·l_PM/(l_PM + μ_r·g').
Implemented in `axial_layer_Bz_series`.

### Qu & Lipo (2004) — PM leakage factors (`pm_leakage_factor`)
**Qu, R., & Lipo, T. A. (2004).** "Analysis and Modeling of Air-Gap and
Zigzag Leakage Fluxes in a Surface-Mounted Permanent-Magnet Machine."
*IEEE Trans. Ind. Appl.*, 40(1), pp. 121–127.
Closed-form magnet-to-rotor, magnet-to-magnet, and zigzag leakage
factors; adopted for axial machines by Parviainen (2005) Eqs. 2.15–2.17.

---

## Motor Design Textbooks

### Hanselman (2006) — general PMSM design reference
**Hanselman, D. C. (2006).** *Brushless Permanent Magnet Motor Design.*
2nd ed. Magna Physics Publishing. ISBN 978-1-881855-15-5.

Primary reference for winding factor `k_w`, slot/tooth geometry, iron
concentration factors for loss, and the flux linkage derivation used in
`derive_B_rem`. Chapter 3 covers air-gap field and flux linkage; Chapter 9
covers losses.

**What we use:**
- Winding factor definition and values (k_w = 0.966 for 72-slot/12-pole)
- Tooth flux concentration factor: `B_tooth ≈ B_gap · τ_slot / w_tooth`
- Yoke flux factor: `B_yoke ≈ B_gap · R_SI / (2·n_p·h_yoke)`

### Gieras (2010) — axial flux / harmonic analysis
**Gieras, J. F., Wang, R. J., & Kamper, M. J. (2010).** *Axial Flux Permanent
Magnet Brushless Machines.* 2nd ed. Springer. ISBN 978-90-481-2399-0.

Secondary reference. Relevant for harmonic decomposition of slot effects and
the general slot harmonic order formula `Q ± n_p`.

### Bianchi & Bolognani (2002) — cogging reduction techniques
**Bianchi, N., & Bolognani, S. (2002).** "Design techniques for reducing the
cogging torque in surface-mounted PM motors." *IEEE Transactions on Industry
Applications*, 38(5), 1259–1265.
https://doi.org/10.1109/TIA.2002.802989

Reference for cogging torque reduction techniques (slot/pole combination,
magnet shaping, skew). Focuses on SPM cogging rather than IPM saliency.

---

## Iron Loss

### Bertotti (1988) — Steinmetz loss model
**Bertotti, G. (1988).** "General properties of power losses in soft
ferromagnetic materials." *IEEE Transactions on Magnetics*, 24(1), 621–630.
https://doi.org/10.1109/20.43994

Theoretical basis for the two-term (hysteresis + eddy) modified Steinmetz
equation:

```
P = k_h · f · B^α + k_e · f² · B²   [W/kg]
```

The M19 test fixture fits coefficients from the cited loss data. The exact
fixture values live with the iron-loss tests so this bibliography does not
become a second configuration source. Confirm coefficients against the
specific grade and lamination thickness before production use; the eddy term
is especially thickness-sensitive.

### EMERF Lamination Steels — M-19 29 Ga core loss tables
**Electric Motor Education and Research Foundation (2004).** "Lamination
Steels, Third Edition: A Compendium of Lamination Steel Alloys Commonly Used
in Electric Motors." South Dartmouth, MA. Excerpt "Di-Max M-19, Fully
Processed, .014 inch (.36 mm, 29 gauge)" published as MIT OpenCourseWare
6.685 Electric Machines (Fall 2013), Steel Data for Problem Set 3.
https://ocw.mit.edu/courses/6-685-electric-machines-fall-2013/resources/mit6_685f13_ps03data/

Typical total AC core loss (W/lb and W/kg) of as-sheared 29 gauge Di-Max
M-19 at 50, 60, 100, 150, 200, 300, 400, 600, 1000, 1500 and 2000 Hz,
0.1 to 1.7 T, from information provided by AK Steel in 2000; also DC and
AC magnetization curves and RMS exciting power. Density 7.65 g/cc. The
W/kg table is fitted to the BMW i3 `[iron]` coefficients. MIT OCW terms:
CC BY-NC-SA, so release archives cite the table rather than redistribute it;
`tests/models/test_iron_loss_m19.py` reads a transcription at
`data/bmw_i3/m19_29ga_core_loss.csv` and skips when it is absent.

### M19 B-H data
**AK Steel Corporation.** "DI-MAX Non-Oriented Electrical Steel M-15 through M-47."
Product data sheet, 2020. Available from Cleveland-Cliffs (formerly AK Steel).
PDF: https://e-magnetica.pl/database-em/01_Soft/Electrical_steels/AKSteel_2020/NO-DI-MAX-M-15_M-19_M-22_M-27_M-36_M-43_M-47_2020.pdf

Tabulated B-H data for silicon steel at 60 Hz, 0–2.2 T range. Used for
the monotone PCHIP H(B) spline (`fem_materials.BHCurve`) in nonlinear FEM. Approximately 20 measurement points
available in the public datasheet. Multi-grade PDF — M-19 section starts
alongside M-15, M-22, M-27, M-36, M-43, M-47.

---

## Permanent Magnets

### NdFeB demagnetization
**Arnold Magnetic Technologies.** "Neodymium Iron Boron — N42." Grade datasheet.
PDF: https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf
Full Neo catalog: https://www.arnoldmagnetics.com/wp-content/uploads/2019/06/Arnold-Neo-Catalog.pdf

For N42 grade at rated temperature range:
- `B_rem(20°C) ≈ 1.30 T`
- `H_cj(20°C) ≈ −900 kA/m` (intrinsic coercivity)
- `α_Br ≈ −0.12 %/°C` (B_rem temperature coefficient)
- `β_Hcj ≈ −0.55 %/°C` (coercivity temperature coefficient)

Temperature coefficients are grade-specific; the knee temperature
dependence is steeper than that of B_rem. **What we use:** `alpha_Br`,
`B_knee` and `alpha_B_knee` in `[materials]` are taken per motor from the
grade datasheet, not from these N42 values.

---

## FEM Formulation

### Silvester & Ferrari (1996) — nonlinear iteration
**Silvester, P. P., & Ferrari, R. L. (1996).** *Finite Elements for
Electrical Engineers.* 3rd ed. Cambridge University Press. Chapter 7
(nonlinear problems).

**What we use:** the damped-Newton default and relaxed-Picard
startup/fallback for the B-H iteration in the 2D and axial FEM solvers
(`RunConfig.nonlinear_solver`). The built-in B-H curve is a generic
soft-steel composite; grade-specific data is the M19 entry above.

### NGSolve documentation
**Schöberl, J. (2014–present).** NGSolve finite element library.
https://ngsolve.org — https://docu.ngsolve.org

A-formulation (vector potential), 2D magnetostatic:
```
−∇·(ν ∇A_z) = J_z + (∇×(ν M))_z
```
This is the weak form used by the radial FEM solver; its module documentation
records the source-term derivation.

### ElmerFEM (cross-code verification)
**CSC — IT Center for Science (1995–present).** Elmer finite element software.
https://www.elmerfem.org — https://github.com/ElmerCSC/elmerfem

Used only as an independent check on the NGSolve solvers — different
mesher, basis, and nonlinear linearization (`MagnetoDynamics2D` /
`MagnetoDynamics`). Pinned comparisons and harness docs:
`scripts/elmer_radial/`, `scripts/elmer_axial/`; results are in
`docs/validation-report.md`.

### TEAM Problem 30a — Davey — time-harmonic eddy-current benchmark (`team30a`)
**Davey, K. R.** "Induction Motor Analyses." International TEAM Workshop
Problem 30 (COMPUMAG). Problem sheet: `problem30a.pdf` at
https://www.compumag.org/wp/team/

Two-dimensional exposed-winding induction motor with a solid conducting
rotor, solved analytically as a layered cylinder. Exact verification
target for the complex-A_z time-harmonic solver in `solvers/fem_harmonic.py`
(stator-frame velocity term σΩ∂A/∂θ). Two conventions not stated on the
sheet are pinned by reproducing the tables: the current density is an rms
phasor, and the exterior beyond the stator is open.

**What we use:** `analytic_team30a` (modified-Bessel series, each space
harmonic at its own slip frequency) and `solve_team30a` /
`build_team30a_mesh` (FE path) in `solvers/team30a.py`; Tables I, II' and
III transcribed in `data/team30a/`. Code verification, not a validation
anchor.

### Maxwell stress tensor torque
**Coulomb, J. L., & Meunier, G. (1984).** "Finite element implementation of
virtual work principle for magnetic or electric force and torque computation."
*IEEE Transactions on Magnetics*, 20(5), 1894–1896.
https://doi.org/10.1109/TMAG.1984.1063232

Canonical reference for Maxwell stress tensor torque computation in FEM via
contour integral on a circular air-gap path — the formulation behind
`maxwell_stress_torque`; the circuit models
(rated_torque.py) remain the primary torque path:
```
τ = (L_stk · R_AG² / μ₀) · ∫₀²π B_r(θ) · B_θ(θ) dθ
```
Also relevant: https://en.wikipedia.org/wiki/Maxwell_stress_tensor

For general EM field theory background:
**Sadiku, M. N. O. (2014).** *Elements of Electromagnetics.* 6th ed. Oxford.
ISBN 978-0-19-964584-7.

---

## Torque and MTPA

### Awan (2019) — control methods for PM synchronous reluctance motor drives
**Awan, H. A. A. (2019).** "Control Methods for Permanent-Magnet Synchronous
Reluctance Motor Drives." Doctoral dissertation, Aalto University School of
Electrical Engineering (defended 15 November 2019).

Equation and section numbers below refer to the 2019 dissertation, not the
2018 journal paper (Awan et al., *IEEE TIA* 54(6), 6110–6120,
https://doi.org/10.1109/TIA.2018.2862410).

Torque equation (eq. 2.14, peak-valued space vectors):
```
T = (3p/2) · (ψ_d · i_q − ψ_q · i_d)
```
With linear magnetics (eq. 2.7–2.8: ψ = L·i + ψ_f):
```
T = (3p/2) · [ψ_f · i_q + (L_d − L_q) · i_d · i_q]
```
This is the form implemented in motulator (`1.5 * n_p * Im(i_s * conj(psi_s))`).
MTPA computation for saturated machines via look-up tables (§5.3).
For unsaturated (constant L_d, L_q), MTPA reduces to a closed-form quadratic
(classical result from Morimoto 1994).

**What we use:** torque equation for rated torque and stall torque models;
MTPA current angle for salient machines. Same equation and conventions as
motulator.

**Motor data (Table 6.2):** 2.2-kW 6-pole IPM (ABB M2BJ 100L 6 B3) used as
rated torque validation case: n_p=3, L_d=36 mH, L_q=51 mH, ψ_f=0.545 Wb,
R_s=3.6 Ω, I_rated=4.3 A_rms (6.08 A peak), rated torque 14 Nm at 1500 rpm.
TOML: `data/awan_ipm/awan_2p2kw_ipm.toml`. Tests: `tests/models/test_rated_torque.py`,
`tests/crossval/test_awan_crossval.py`.

### Morimoto et al. (1994) — MTPA for IPM drives
**Morimoto, S., Sanada, M., & Takeda, Y. (1994).** "Wide-speed operation of
interior permanent magnet synchronous motors with high-performance current
regulator." *IEEE Transactions on Industry Applications*, 30(4), 920–926.
https://doi.org/10.1109/28.297908

Original reference for the closed-form MTPA quadratic used in IPM drives.
The unsaturated MTPA condition (dT/dγ = 0 at constant |I_s|) yields a
quadratic in sin(γ):
```
2·(L_q − L_d)·I_s·sin²(γ) + ψ_f·sin(γ) − (L_q − L_d)·I_s = 0
```
with solution:
```
sin(γ) = [−ψ_f + √(ψ_f² + 8·(L_q − L_d)²·I_s²)] / [4·(L_q − L_d)·I_s]
i_d = −I_s·sin(γ),  i_q = I_s·cos(γ)
```
For SPM (L_d = L_q): γ = 0, i_d = 0, i_q = I_s. The constant-parameter MTPA
locus is the baseline that Awan (2019) §5.1 contrasts its look-up-table
method against (Fig. 5.1 dashed loci).

**What we use:** closed-form MTPA for rated torque and stall torque of
salient machines with constant (unsaturated) inductances. Known limitation: at high stall currents,
saturation reduces L_d/L_q and ψ_f — linear model overpredicts torque.

### Soong & Miller (1994) — field-weakening envelope (`torque_speed`)
**Soong, W. L., & Miller, T. J. E. (1994).** "Field-weakening performance
of brushless synchronous AC motor drives." *IEE Proceedings — Electric
Power Applications*, 141(6), 331–340.
https://doi.org/10.1049/ip-epa:19941470

Unified treatment of the constant-parameter dq envelope: current-limit
circle, voltage-limit ellipse, MTPA below base speed, and the MTPV locus
above it. Classifies machines by ψ_f versus L_d·I_max: when the ellipse
center lies inside the current circle the MTPV branch extends to
unbounded speed; otherwise torque reaches zero at a finite speed. The
circle/ellipse picture itself originates in Jahns, Kliman & Neumann
(1986), *IEEE Trans. Ind. Appl.*, IA-22(4), 738–747,
https://doi.org/10.1109/TIA.1986.4504786.

**What we use:** `torque_speed.py` follows this framework with R_s
retained — `base_speed` solves the voltage quadratic at the MTPA
currents, `max_speed` applies the ψ_f ≤ L_d·I_max test, and
`_envelope_core` finds the voltage-limited optimum by i_d search rather
than the closed-form mode split.

---

## Drive Simulation

### motulator
**Hinkkanen, M., et al.** motulator: Motor Drive Simulator. Version 0.7.3.
https://github.com/Aalto-Electric-Drives/motulator (MIT License)

Used for time-domain drive simulation. Classes used (`simulation/sim.py`):
- Model: `SynchronousMachine`, `SynchronousMachinePars`, `Drive`,
  `MechanicalSystem`, `VoltageSourceConverter`, `Simulation`, `Step`
- Control: `CurrentVectorController(Cfg)`, `SpeedController`,
  `VectorControlSystem`

API signatures verified against v0.7.3 source.

---

## Cogging Torque

### Cogging period formula
Standard result from winding theory:
```
T_cogging = 2π / LCM(Q, 2·n_p)
```
See Hanselman (2006) Ch. 7, or:
**Zhu, Z. Q., & Howe, D. (2000).** "Influence of design parameters on
cogging torque in permanent magnet machines." *IEEE Transactions on Energy
Conversion*, 15(4), 407–412.
https://doi.org/10.1109/60.900501
Free PDF: https://eprints.whiterose.ac.uk/889/1/zhuzq20.pdf

---

## Induction Motor

### Boldea & Nasar (2010) — **primary reference** (`im_circuit`, `im_design_audit`)
**Boldea, I., & Nasar, S. A. (2010).** *The Induction Machines Design
Handbook.* 2nd ed. CRC Press. ISBN 978-1-4200-6668-5.
Chapter summary: `docs/references/boldea-nasar-imh-2e.md`.

Source for the steady-state equivalent circuit and its component physics:
magnetization curve and saturation factor (Ch 5), leakage inductances —
slot, differential, zig-zag, end-winding, skin effect (Ch 6), R_fe shunt
placement, Thevenin breakdown, and torque-slip (Ch 7), deep-bar impedance
(Ch 9), and loss separation (Ch 11).

**What we use:**
- `models/im_circuit.py`: circuit topology and R_fe placement (Ch 7 §7.1,
  Ch 11 §11.2); R_fe/X_m ≈ 20 diagnostic
- `audit/im_design_audit.py`: advisory bands for rated slip (§7.6),
  breakdown ratio (eq 7.74), starting current (eq 7.35), no-load current,
  B_g and K_s (§5.4)
- `models/im_bar_impedance.py`: end-ring permeance (Ch 6) and Layer-2
  invariants (eqs 6.71–72, derivation Ch 9)
- `models/im_fem_harmonic.py`: stator end-winding leakage L_ew in the
  stator circuit and the FEM-identified stator leakage

### IEEE Std 112-2017 — induction motor test procedure
**IEEE (2017).** *IEEE Standard Test Procedure for Polyphase Induction
Motors and Generators.* IEEE Std 112-2017.
https://doi.org/10.1109/IEEESTD.2018.8291810

Defines the DC / no-load / locked-rotor identification of the equivalent
circuit, the empirical stator/rotor split of locked-rotor leakage, the
no-load loss separation (P_rot = P − 3·I²·R_1 extrapolated to zero
voltage), and the method-B stray-load-loss residual with assumed
percentages by rating (Table 9-2).

**What we use:** `identify_from_tests`, `identify_locked_rotor`,
`separate_no_load_losses`, `compute_stray_load_residual` in
`models/im_circuit.py`; the virtual no-load and locked-rotor steps in
`models/im_fem_harmonic.py` feed the same identification.

---

## Validation Anchors

### Gierczynski & Grzesiak (2021) — BMW i3 IPM FEM model (`motors/bmw_i3.toml`)
**Gierczynski, M., & Grzesiak, L. M. (2021).** "Comparative Analysis of the
Steady-State Model Including Non-Linear Flux Linkage Surfaces and the
Simplified Linearized Model when Applied to a Highly-Saturated Permanent
Magnet Synchronous Machine—Evaluation Based on the Example of the BMW i3
Traction Motor." *Energies*, 14(9), 2343. CC BY 4.0.
https://doi.org/10.3390/en14092343

Ansys Maxwell 2D reconstruction of the 125 kW, 12-pole / 72-slot V-shape
IPM from ORNL teardown data, validated against ORNL locked-rotor torque.
All circuit values are FEA-extracted, not measured; the supplementary DXF
and xlsx are archived in `data/bmw_i3/` (provenance in its README).

**What we use:**
- Table 2 circuit parameters (ψ_PM 43.6 mWb, L_d 71.2 µH, L_q 141.3 µH,
  R_s 5.3 mΩ); Table A1 geometry, A2 winding, A3 materials
- Tables A4–A5 flux-linkage maps ψ_d(i_d,i_q), ψ_q(i_d,i_q) →
  `data/bmw_i3/flux_maps_7x7.json`
- Tests: `tests/models/test_torque_speed.py` (`TestBmwI3Envelope`),
  `tests/crossval/test_bmw_i3_flux_maps.py`

### Burress (2016, 2017) — BMW i3 ORNL teardown and benchmarking
**Burress, T. (2016).** "Benchmarking EV and HEV Technologies," DOE VTO
Annual Merit Review, Project EDT006, June 2016.
https://www.energy.gov/sites/prod/files/2016/06/f32/edt006_burress_2016_o_web.pdf

**Burress, T. (2017).** "Electrical Performance, Reliability Analysis, and
Characterization," DOE VTO Annual Merit Review, Project EDT087, June 2017.
https://www.energy.gov/sites/default/files/2017/06/f34/edt087_burress_2017_o.pdf

ORNL teardown and dynamometer testing of the 2016 BMW i3 traction motor
(125 kW IPM). Provides mass breakdown, slot/pole count, efficiency maps at
300/500/700 Vdc, continuous thermal test at 7000 rpm, and power density
comparisons. No d-q circuit parameters or back-EMF waveforms published.

**What we use:** motor geometry envelope, mass, rated performance, and
efficiency map extent for `motors/bmw_i3.toml`. Circuit parameters come
from Gierczyński (2021) above, not from ORNL directly.

### Merwerth (2014) — BMW i3/i8 traction motor presentation
**Merwerth, A. (2014).** "The Hybrid-Synchronous Machine of the New BMW
i3 & i8," HEV 2014 presentation, BMW Group.
http://hybridfordonscentrum.se/wp-content/uploads/2014/05/20140404_BMW.pdf

BMW's own presentation of the i3 traction motor. Source for system-level
parameters: phase resistance R_s = 5.3 mΩ, max phase current 565.7 A,
max phase voltage 159.2 V, max speed 11400 rpm.

**What we use:** R_s and drive limits in `motors/bmw_i3.toml`.

### Tikadar, Kumar, Joshi & Kumar (2020) — BMW i3 electro-thermal model (`[thermal_network]` R_WFe)
**Tikadar, A., Kumar, N., Joshi, Y., & Kumar, S. (2020).** "Coupled
Electro-Thermal Analysis of Permanent Magnet Synchronous Motor for
Electric Vehicles." *19th IEEE ITherm*, pp. 249–256. OSTI 1843426.
https://www.osti.gov/servlets/purl/1843426

Two-way coupled Motor-CAD + Fluent model of the jacket-cooled i3 motor.

**What we use:** winding-to-iron resistance `R_WFe` (1/`G_WFe`) in the
i3 `[thermal_network]` block, back-solved from two points of the paper's
hot-spot curve; derivation in `data/bmw_i3/README.md`.

### Becker & Kaye (1962) — air-gap heat transfer (`[thermal_network]` G_FeM)
**Becker, K. M., & Kaye, J. (1962).** "Measurements of Diabatic Flow in an
Annulus With an Inner Rotating Cylinder." *ASME J. Heat Transfer*, 84(2),
pp. 97–104.

Taylor–Couette Nusselt correlation for the rotor–stator gap.

**What we use:** `G_FeM` in the i3 block (≈ 8.2 W/K at 7000 rpm;
derivation in `data/bmw_i3/README.md`).

### CREATOR open benchmark — PMSM and IM datasets (`motors/creator_case_pmsm.toml`, `data/creator_case_im/`)
**Muetze, A., Heidarikani, K., Dhakal, P. K., Seebacher, R., & Schöps, S.
(2025).** CREATOR Case open benchmark. *COMPEL*,
https://doi.org/10.1108/compel-11-2024-0462 (preprint
arXiv:2501.15921). Datasets: PMSM DOI 10.3217/sns1d-77m43 (CC BY-NC-ND
4.0), IM DOI 10.3217/kh3v7-dhn98 (CC BY-NC 4.0).

**What we use:** measured back-EMF, MTPA angle, rated torque, cogging and
no-load loss for the sintered-ferrite inset PMSM; locked-rotor sweeps,
nameplate and no-load tests for the induction machine. Attribution and
what is and is not shipped: `data/creator_case_pmsm/README.md`,
`data/creator_case_im/README.md`.

### Belkhadir et al. (2023) — 22p/24s external-rotor PMSM (`motors/belkhadir_outrunner.toml`)
"Electromagnetic Design and Optimization of a New 22-Pole/24-Slot
External Rotor PMSM." *IECON 2023*.
https://doi.org/10.1109/IECON51785.2023.10312419

**What we use:** published air-gap flux density, back-EMF, flux linkage
and rated torque as the outrunner analytical/FEM anchor.

### Deylami et al. (2024) — 8p/12s outer-rotor cooling fan (`motors/deylami_fan.toml`)
TechRxiv preprint, https://doi.org/10.36227/techrxiv.171439839.97768997/v1

**What we use:** the paper's 2D ANSYS Maxwell air-gap peak as a
model-vs-model anchor for the slotted FEM path.

### ORNL Prius 2004 reports (`motors/prius_2004.toml`)
ORNL/TM-2004/137 (teardown, https://doi.org/10.2172/885676),
ORNL/TM-2004/185 (back-EMF, https://doi.org/10.2172/921782),
ORNL/TM-2006/423 (dynamometer, https://doi.org/10.2172/890029).

**What we use:** circuit parameters and the 340 N·m stall-torque bound;
sources and assumptions in `data/prius_2004/README.md`.
