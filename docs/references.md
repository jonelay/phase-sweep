# Physics References

Canonical sources for the models and data used in phasesweep. Organized by
topic. Where a result is already implemented, the relevant function is noted.

---

## Air-Gap Field Analysis

### Zhu & Howe (1993) — **implemented** (`zhu_howe_Br`)
Four-part series, all in *IEEE Transactions on Magnetics*, 29(1), Jan. 1993.
All paywalled on IEEE Xplore; no free PDFs found.

- **Part I — Open-Circuit Field** (magnet source only)
  https://doi.org/10.1109/20.195557 — pp. 124–135
- **Part II — Armature-Reaction Field**
  https://doi.org/10.1109/20.195558 — pp. 136–142
- **Part III — Effect of Stator Slotting**
  https://doi.org/10.1109/20.195559 — pp. 143–151

**What we implement:** the Part I geometry (open-circuit, PM-only field) with
sinusoidal radial magnetization and infinite iron permeability at both bore
surfaces. `zhu_howe_Br()` solves the 2×2 interface-matching system numerically.
FEM matches within 0.18–0.28% on fundamental amplitude (validated against
Zhu & Howe analytical solution for smooth-bore geometry).

Part III (slotting) is the natural next analytical reference for cogging
validation once §1.1 is implemented.

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
shaping, skew). Cited in the spec as a starting point for IPM barrier geometry
(§1.4), though this paper focuses on SPM cogging rather than IPM saliency —
a more targeted IPM barrier reference should be substituted when implementing §1.4.

---

## Iron Loss

### Bertotti (1988) — Steinmetz loss model
**Bertotti, G. (1988).** "General properties of power losses in soft
ferromagnetic materials." *IEEE Transactions on Magnetics*, 24(1), 621–630.
https://doi.org/10.1109/20.43994

Theoretical basis for the two-term (hysteresis + eddy) modified Steinmetz
equation used in spec §3.1:

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
the polynomial ν(B²) fit in spec §1.2. Approximately 20 measurement points
available in the public datasheet. Multi-grade PDF — M-19 section starts
alongside M-15, M-22, M-27, M-36, M-43, M-47.

---

## Permanent Magnets

### NdFeB demagnetization (spec §3.2)
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
4. Under-relax: ν_new = α·ν_BH + (1−α)·ν_old, with α = 0.3

Convergence criterion: max relative change in ν over iron elements < 1%.
Typically converges in 5–10 iterations for moderate saturation.

**Standard reference:**
Silvester, P. P., & Ferrari, R. L. (1996). *Finite Elements for Electrical
Engineers.* 3rd ed. Cambridge University Press. Chapter 7 (nonlinear problems).

The B-H curve used is a generic soft magnetic steel composite (not
grade-specific). See `_BH_B` / `_BH_H` in `fem_field.py`. For grade-specific
data, see the M19 datasheet entry below.

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
contour integral on a circular air-gap path — exactly the method in spec §1.1:
```
τ = (L_stk · R_AG² / μ₀) · ∫₀²π B_r(θ) · B_θ(θ) dθ
```
Also relevant: https://en.wikipedia.org/wiki/Maxwell_stress_tensor

For general EM field theory background:
**Sadiku, M. N. O. (2014).** *Elements of Electromagnetics.* 6th ed. Oxford.
ISBN 978-0-19-964584-7.

---

## Drive Simulation

### motulator
**Hinkkanen, M., et al.** motulator: Motor Drive Simulator. Version 0.7.3.
https://github.com/Aalto-Electric-Drives/motulator (MIT License)

Used for time-domain drive simulation. Key classes:
- `SynchronousMachinePars`, `SaturatedSynchronousMachinePars`
- `MachineCharacteristics`, `ControlLoci`, `MagneticModel`

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
