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
numerically. FEM agrees within 1% on fundamental amplitude for the Zhu & Howe
reference configurations and within 2% for the other tested configurations
(test-enforced bounds).

### Zhu & Howe (1993) — foundational series
Four-part series, all in *IEEE Transactions on Magnetics*, 29(1), Jan. 1993.
All paywalled on IEEE Xplore; no free PDFs found.

- **Part I — Open-Circuit Field** (magnet source only)
  https://doi.org/10.1109/20.195557 — pp. 124–135
- **Part II — Armature-Reaction Field**
  https://doi.org/10.1109/20.195558 — pp. 136–142
- **Part III — Effect of Stator Slotting**
  https://doi.org/10.1109/20.195559 — pp. 143–151

Part III (slotting) is the primary reference for the Carter factor
correction. Eqs 15–17 define k_c using the
effective airgap g' = g + h_m/μ_r and the modified stator radius R_se.
Also the natural reference for the cogging-torque sweep.

---

## Motor Design Textbooks

### Hanselman (2006) — general PMSM design reference
**Hanselman, D. C. (2006).** *Brushless Permanent Magnet Motor Design.*
2nd ed. Magna Physics Publishing. ISBN 978-1-881855-15-5.

Primary reference for winding factor `k_w`, slot/tooth geometry, iron
concentration factors for loss, and the flux linkage derivation used in
`_derive_B_rem`. Chapter 3 covers air-gap field and flux linkage; Chapter 9
covers losses.

**What we use:**
- Winding factor definition and values (k_w = 0.966 for 12-slot/4-pole)
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
Paywalled on IEEE Xplore; no free PDF found.

General reference for cogging torque reduction (slot/pole combination, magnet
shaping, skew). Cited as a starting point for IPM barrier geometry, though
this paper focuses on SPM cogging rather than IPM saliency — a more targeted
IPM barrier reference should be substituted when implementing that feature.

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

**M19 29-gauge coefficients** (fitted from 60 Hz loss data, SI units):
- `k_h ≈ 0.0275` [W·s/kg·T^α] — hysteresis coefficient
- `k_e ≈ 0.00026` [W·s²/kg·T²] — eddy current coefficient
- `α ≈ 1.7` — Steinmetz exponent

Source for coefficient values: AK Steel M19 silicon steel datasheet (public
domain). Confirm against the specific grade and lamination thickness before
use in production. Coefficients are highly sensitive to lamination thickness
(eddy current term scales as t²).

### M19 B-H data
**AK Steel Corporation.** "DI-MAX Non-Oriented Electrical Steel M-15 through M-47."
Product data sheet, 2020. Available from Cleveland-Cliffs (formerly AK Steel).
PDF: https://e-magnetica.pl/database-em/01_Soft/Electrical_steels/AKSteel_2020/NO-DI-MAX-M-15_M-19_M-22_M-27_M-36_M-43_M-47_2020.pdf

Tabulated B-H data for silicon steel at 60 Hz, 0–2.2 T range. Used for
the polynomial ν(B²) fit in nonlinear FEM. Approximately 20 measurement points
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

Temperature coefficients are grade-specific. N42 is a medium-grade reference;
N52 has higher B_rem but worse temperature performance; grades ending in
"H", "SH", "UH" have better high-temperature coercivity.

**Caution:** The knee field `H_knee` is geometry-dependent and typically read
from the second-quadrant B-H curve at the inflection point. The temperature
dependence of the knee is steeper than that of B_rem.

---

## FEM Formulation

### Picard iteration for nonlinear iron permeability
Fixed-point (Picard) iteration for solving nonlinear magnetostatic problems
where iron permeability depends on flux density. At each iteration:
1. Solve linear system with current ν(B) distribution
2. Compute |B| per element from the solution
3. Update ν from B-H curve lookup (iron elements only)
4. Under-relax: ν_new = α·ν_BH + (1−α)·ν_old, with α = 0.2 (defaults.toml)

Convergence criterion: max relative change in ν over iron elements < 1%.
Typically converges in 5–10 iterations for moderate saturation.

**Standard reference:**
Silvester, P. P., & Ferrari, R. L. (1996). *Finite Elements for Electrical
Engineers.* 3rd ed. Cambridge University Press. Chapter 7 (nonlinear problems).

The B-H curve used is a generic soft magnetic steel composite (not
grade-specific). See `_BH_B` / `_BH_H` in `fem_field.py`. For grade-specific
data, see the M19 datasheet entry above (Iron Loss section).

### NGSolve documentation
**Schöberl, J. (2014–present).** NGSolve finite element library.
https://ngsolve.org — https://docu.ngsolve.org

A-formulation (vector potential), 2D magnetostatic:
```
−∇·(ν ∇A_z) = J_z + (∇×(ν M))_z
```
Weak form used in `solve_field_fem()`. See `fem_field.py` module docstring
for the specific source term derivation.

### Maxwell stress tensor torque
**Coulomb, J. L., & Meunier, G. (1984).** "Finite element implementation of
virtual work principle for magnetic or electric force and torque computation."
*IEEE Transactions on Magnetics*, 20(5), 1894–1896.
https://doi.org/10.1109/TMAG.1984.1063232
Paywalled on IEEE Xplore; no free PDF confirmed.

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

Every equation and section number below is the **dissertation's**. The
underlying journal paper — Awan, Song, Saarakkala & Hinkkanen (2018),
"Optimal torque control of saturated synchronous motors: Plug-and-play
method," *IEEE TIA* 54(6), 6110–6120, https://doi.org/10.1109/TIA.2018.2862410
— is Publication VI of the thesis and numbers its equations differently.

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
TOML: `data/awan_ipm/awan_2p2kw_ipm.toml`. Tests: `tests/test_rated_torque.py`.

### Morimoto et al. (1994) — MTPA for IPM drives
**Morimoto, S., Sanada, M., & Takeda, Y. (1994).** "Wide-speed operation of
interior permanent magnet synchronous motors with high-performance current
regulator." *IEEE Transactions on Industry Applications*, 30(4), 920–926.
https://doi.org/10.1109/28.297908
Paywalled on IEEE Xplore; no free PDF found.

Foundational reference for MTPA current angle optimization in IPM drives.
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
locus is the baseline Awan (2019) §5.1 contrasts its look-up-table method
against (Fig. 5.1 dashed loci).

**What we use:** closed-form MTPA for rated torque and stall torque of
salient machines with constant (unsaturated) inductances. Known limitation: at high stall currents,
saturation reduces L_d/L_q and ψ_f — linear model overpredicts torque.

---

## Drive Simulation

### motulator
**Hinkkanen, M., et al.** motulator: Motor Drive Simulator. Version 0.7.3.
https://github.com/Aalto-Electric-Drives/motulator (MIT License)

Used for time-domain drive simulation. Classes used (`sim.py`):
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
