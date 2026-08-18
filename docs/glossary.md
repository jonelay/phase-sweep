# Glossary — Validation Report & Sensitivity Analysis

Terms, symbols, and abbreviations appearing in the phase-sweep validation
report (`scripts/validation_report.py`) and supporting modules.

See also: [naming-crossref.md](naming-crossref.md) for naming collisions across domains.

---

## Conventions

**Peak vs RMS:** All current and flux linkage values in the code follow
motulator's peak-value convention. TOML files may specify `I_rated_rms`
which is converted to peak via `I_rated = I_rated_rms × √2` at load time
(`machines/configs.py`). Torque formulas use the 3/2 factor that pairs with
peak values: `τ = 1.5 × n_p × (ψ_f × i_q + (L_d − L_q) × i_d × i_q)`.

**Units:** All geometry values are in SI (meters), not millimeters or
normalized radii. The Geometry dataclass enforces physical ordering.

---

## Motor Types

| Term | Meaning | Verified against |
|------|---------|-----------------|
| PMSM | Permanent Magnet Synchronous Machine | — |
| SPMSM | Surface-mounted PMSM (non-salient: L_d ≈ L_q) | Zhu & Howe: surface-mounted assumed throughout |
| IPM / IPMSM | Interior PM machine (salient: L_q > L_d) | motulator: SynchronousMachinePars supports L_d ≠ L_q |
| ER-PMSM | External-Rotor PMSM (outrunner topology) | Zhu Eq 18: external rotor motor, R_s < R_m < R_r |
| Inrunner | Rotor inside stator (conventional layout) | Zhu Fig 1(a): "Internal rotor" |
| Outrunner | Stator inside rotor (e.g. Belkhadir 22p/24s) | Zhu Fig 1(b): "External rotor" |
| FSCW | Fractional-Slot Concentrated Winding | — (Zhu §IV covers nonoverlapping windings) |

> **Zhu disambiguation:** Zhu calls these "internal rotor" and "external rotor."
> We use the more common "inrunner"/"outrunner" terminology.

## Geometry

| Symbol | Units | Meaning | Zhu & Howe equivalent |
|--------|-------|---------|-----------------------|
| r_outer | m | Motor frame outer radius | — (outside model boundary) |
| r_stator | m | Stator bore radius (air-gap boundary) | R_s (stator bore; Eq 11a) |
| r_magnet | m | Magnet outer radius (PM/air-gap interface) | R_m (Eq 11c–d) |
| r_rotor | m | Rotor iron radius (PM inner boundary) | R_r (Eq 11b) |
| r_inner | m | Shaft / hollow center radius | — (not modeled by Zhu) |
| r_ag | m | Air-gap center radius = (r_stator + r_magnet)/2 | — (derived) |
| OD | — | Outer Diameter (sensitivity parameter) | — |
| L_stk | m | Stack (core) axial length | L (Table I: "Axial length") |
| n_slots | — | Number of stator slots | Q_s (Table I) |
| slot_depth | m | Slot depth perpendicular to bore | — (Zhu uses b_o for slot opening) |
| slot_width_ratio | — | Slot body width as fraction of slot pitch | — |
| slot_opening_width | m | Slot opening width at the bore (stored field; Carter input) | Zhu: b_o |
| slot_opening_ratio | — | Derived: slot_opening_width / slot pitch at bore | — |
| topology | — | `"inrunner"` or `"outrunner"` | "internal" / "external" rotor |
| α_p | — | Magnet pole-arc / pole-pitch ratio (default 1.0) | Zhu Eq 7a: α_p. On Motor, AnalyticalParams, FemParams. See note below. |

> **α_p scaling (Zhu Eq 7a).** For the fundamental harmonic, the ratio of
> partial-pitch to full-pitch magnetization is exactly `sin(πα_p/2)`.
> **Analytical:** `zhu_howe_Br` applies this as `k_mp = sin(πα_p/2)` directly.
> **FEM:** The mesh models discrete magnet arcs (`_cut_magnet_arcs`
> in `solvers/fem_field.py`) — `fem_runner` passes unscaled
> `B_rem` together with `alpha_p`, and the geometry itself captures the
> inter-magnet gaps. Arcs are skipped (full ring) when the gap width
> would be < 0.5 mm, to avoid OCC boolean failures.

> **Key collision (resolved):** Zhu's R_s (stator bore radius) ≠ motulator's R_s
> (stator winding resistance). Our `r_stator` sidesteps this. See `naming-crossref.md §1`.

> **Outrunner radius ordering** flips: r_outer > r_rotor > r_magnet > r_stator > r_inner.
> The names stay the same — r_stator is always the bore, r_magnet always the PM/air
> boundary — but their geometric relationship reverses. Code enforces this in `Geometry.__post_init__`.

## Magnetic & Material Properties

| Symbol | Units | Meaning | Zhu / motulator mapping |
|--------|-------|---------|------------------------|
| B_rem | T | Permanent magnet remanence | Zhu: B_r (Eq 2, 7a). We use B_rem to avoid collision with B_r(θ) waveform. |
| psi_f (ψ_f) | Vs (Wb) | PM flux linkage per phase (peak value) | motulator: `psi_f` on SynchronousMachinePars ✓ |
| mu_r_pm (μ_r,pm) | — | Relative recoil permeability of magnet | Zhu: μ_r (Eq 1b, assumed constant) ✓ |
| mu_r_fe (μ_r,fe) | — | Relative permeability of stator/rotor iron | Zhu: assumes μ_r,fe → ∞ (Eq 11a–b); our FEM relaxes this |
| MU0 (μ₀) | H/m | Permeability of free space (4π × 10⁻⁷) | Zhu: μ₀ (Eq 1a) ✓ |
| n_p | — | Number of pole pairs | Zhu: p (Eq 6a). motulator: `n_p` ✓ |

> **Zhu assumes infinite iron permeability** (boundary conditions Eq 11a–b impose
> H_θ = 0 at iron surfaces). Our FEM solver relaxes this via finite mu_r_fe and
> optional B-H curve with Picard iteration.

## Air-Gap Flux Density

| Symbol | Units | Meaning |
|--------|-------|---------|
| B_r(θ) | T | Radial flux density waveform at air-gap mid-line |
| B₁ | T | Fundamental harmonic amplitude (at spatial order n_p) |
| B₁_analytical | T | B₁ from the Zhu & Howe analytical model |
| B₁_fem | T | B₁ from the NGSolve FEM solver |
| fundamental | T | Same as B₁ — metric key in RunResult |
| B_ag_peak | T | Peak |B_r| over one pole pitch |
| harmonics_1sided | T[] | One-sided FFT amplitudes of B_r(θ) |
| thd_pct | % | Total Harmonic Distortion |
| sh_pct | % | Slot harmonic amplitude as % of fundamental |
| theta (θ) | rad | Mechanical angle, 0 to 2π around full circumference |

> **θ convention:** Zhu defines θ "with reference to the center of a magnet pole"
> (Fig 2). Our code samples θ from 0 to 2π around the full circumference. The
> cos(npθ) form in Eq 15a handles this — the n_p-th order harmonic captures the
> fundamental pole-pair component regardless of reference frame.

## Torque & Current Control

| Symbol | Units | Meaning |
|--------|-------|---------|
| τ_rated (tau_rated) | N·m | Rated torque: MTPA torque at I_rated (metric key: `tau_mtpa`) |
| τ_stall (tau_stall) | N·m | Stall torque: MTPA torque at I_stall (drive-limited peak current) |
| saturation_ratio | — | I_stall / I_rated — how far beyond rated the stall current is |
| saturation_warning | — | Boolean flag: true when saturation_ratio > 3.0 and linear model is unreliable |
| k_T | N·m/A_peak | Torque constant = 1.5 × n_p × ψ_f (peak current convention) |
| k_T_rms | N·m/A_rms | RMS torque constant = k_T × √2 |
| k_T_effective | N·m/A_peak | Effective torque per ampere at MTPA = τ / I_s (includes reluctance torque; equals k_T for non-salient) |
| MTPA | — | Maximum Torque Per Ampere (Morimoto 1994 quadratic) |
| γ (gamma) | rad | MTPA angle from q-axis (0 for non-salient; < 0 for reverse saliency, magnetizing i_d) |
| gamma_opt_deg | deg | Optimal MTPA angle at I_rated |
| I_s | A_peak | Stator current magnitude (peak, per motulator convention) |
| I_rated | A_peak | Rated continuous stator current (peak; TOML may specify I_rated_rms) |
| I_stall | A_peak | Stall current = drive limit (I_LIMIT, else MAX_I_S); electromagnetic limit U_DC / (√3 × R_s) reported separately as I_stall_em |
| i_d | A_peak | d-axis current = −I_s sin(γ) (field-weakening) |
| i_q | A_peak | q-axis current = I_s cos(γ) (torque-producing) |
| L_d | H | d-axis inductance |
| L_q | H | q-axis inductance |
| saliency | — | L_q/L_d ratio; >1 = salient (IPM), ~1 = non-salient |

> **Peak convention:** see Conventions section above. motulator's
> `i_s_max` on CurrentVectorControllerCfg is also peak amps.

## Drive Simulation

| Symbol | Units | Meaning | motulator equivalent |
|--------|-------|---------|---------------------|
| W_REF | rad/s | Motor reference speed (**mechanical**) | — (our convention; TOML comments confirm "rpm mechanical") |
| U_DC | V | DC bus voltage | `u_dc` on VoltageSourceConverter (lowercase in motulator) |
| MAX_I_S | A_peak | Maximum stator current | `i_s_max` on CurrentVectorControllerCfg ✓ |
| tau_peak | N·m | Peak torque during acceleration transient | — (extracted from tau_M time series) |
| i_ss | A_peak | Steady-state stator current magnitude | — (extracted from |i_s_ab| time series) |
| speed_droop | — | Fractional speed dip after load step (w_dip / w_ref) | — |
| t_settle | s | Time from speed step to within 5% of target | — |
| w_M | rad/s | Mechanical rotor speed | motulator: `w_M` on MechanicalSystem ✓ |
| w_m | rad/s | Electrical rotor speed = w_M × n_p | motulator: local `w_m = par.n_p * inp.w_M` in rhs() |
| tau_M | N·m | Electromagnetic torque | motulator: `tau_M` ✓ |
| i_s_ab | A (complex) | Stator current in αβ frame (peak) | motulator: `i_s_ab` ✓ |
| J | kg·m² | Moment of inertia | motulator: `J` on MechanicalSystem ✓ |

> **backemf_fundamental** formula in `registry.py`: `w_e = W_REF * n_p` correctly
> converts mechanical → electrical, then `E = w_e × ψ_f`. Validated: CREATOR 47.91 V
> computed vs 47.37 V measured (1.1%).

## Winding Parameters

| Symbol | Units | Meaning | Zhu equivalent |
|--------|-------|---------|----------------|
| N | — | Turns per coil (single slot) | — (Zhu's W is total series turns per phase) |
| N_eff | — | Effective series turns per phase = N × coils_series | Zhu: W (Table I: "Series turns/phase") |
| k_w | — | Winding factor (= k_d × k_p in general) | Zhu: K_dpν (Eq 22, combined distribution × pitch factor) |
| coils_series | — | Coils in series per phase (required explicitly when N/k_w are set; no auto-derivation) | — |
| R_s | Ω | Stator winding resistance | motulator: `R_s` on SynchronousMachinePars ✓ |
| backemf_fundamental | V | Fundamental back-EMF (peak) = ω_e × ψ_f | — |

> **N vs W:** Zhu's W = total series turns per phase. Our N = turns per coil.
> The bridge is N_eff = N × coils_series ≈ W. For CREATOR: N=328, coils_series=2
> (set in the TOML [winding] section), N_eff=656.

## FEM Solver

| Term | Meaning |
|------|---------|
| NGSolve | Open-source FEM library (Netgen/NGSolve, LGPL) for 2-D magnetostatics |
| maxh_fraction | Max mesh element size as fraction of r_outer (default 0.05) |
| n_theta | Angular sampling points in air-gap (default 360) |
| nonlinear | Enables B-H curve with Picard iteration |
| Picard iteration | Fixed-point iteration updating reluctivity from local |B| |
| picard_relax | Relaxation factor (0–1) for Picard updates |
| A-formulation | Magnetic vector potential: −∇·(ν ∇A_z) = source in PM region |
| fes | Finite Element Space (H1 for potential, L2 for reluctivity) |
| gfu | GridFunction (FEM solution: A_z potential) |
| ν (nu) | Reluctivity = 1/μ; appears in FEM weak form as CoefficientFunction |
| Material CF | CoefficientFunction mapping mesh regions → material properties |

> **Zhu uses scalar potential φ, not vector potential A.** Zhu's Eq 9a–b solve
> Laplacian/quasi-Poissonian equations for φ; our FEM uses the A-formulation
> (vector potential A_z). Both are valid for 2-D magnetostatics and give equivalent
> results; they differ in formulation, not physics.

## Analytical Model

| Term | Meaning |
|------|---------|
| Zhu & Howe 2002 | "Improved Analytical Model for Predicting the Magnetic Field Distribution in Brushless Permanent-Magnet Machines," IEEE Trans. Magn., 38(1), pp. 229–238 |
| ratio-invariant | Proportional radius scaling preserves all radius ratios → exactly 0% B₁ change |
| p, q, s | Abstract radii in our `aform_Br` test implementation (topology-agnostic). NOT from Zhu — Zhu uses R_s, R_m, R_r directly. Our mapping: s=iron BC, q=PM/airgap, p=far iron BC |
| K_B(n) | Zhu Eq 17/18: amplitude coefficient per harmonic. Contains geometry ratios and μ_r. |
| f_Br(r) | Zhu Eq 17/18: radial variation function. Depends on evaluation radius and R_s/R_m. |
| M_n | Zhu Eq 10b: effective magnetization harmonic = M_rn + np × M_θn |
| A_3n | Zhu: magnetization shape factor (= np for radial magnetization, more complex for parallel) |
| α_p | Zhu Eq 7a: magnet pole-arc / pole-pitch ratio. Configurable on Motor (default 1.0). Scaling: bp *= sin(πα_p/2). |

## Sensitivity Analysis

| Term | Meaning |
|------|---------|
| perturbation (δ) | Fractional change applied to a parameter (e.g. +5% = δ=0.05) |
| baseline | Model output at δ=0 for each parameter (recomputed per parameter with ψ_f cleared) |
| track | Output metric being tracked (τ_rated, τ_stall, B₁_analytical, B₁_fem, τ_sim) |
| pct_change | Response change: 100 × (perturbed − baseline) / |baseline| |
| response change | Y-axis of sensitivity plots (% change in tracked metric) |
| R² | Coefficient of determination for linear fit through perturbed points |
| linearity | R² > 0.999 → "linear"; otherwise reported as "R²=value" |
| rotor shell | Magnet + yoke; shifts outward with the frame during OD perturbation (thickness fixed) |
| intensive | Property independent of motor size (e.g. B₁ in Tesla) |
| extensive | Property that scales with motor size (e.g. torque in N·m) |

### Perturbation parameters

| Key | What changes | What stays fixed | Expected response |
|-----|-------------|------------------|-------------------|
| OD | Frame outer diameter (stator + rotor shift together; bore grows) | Air gap, magnet thickness, shaft | Amplified — bore grows faster than OD (frame-size model) |
| gap | Air-gap thickness | Magnet, yoke | ~Linear (inverse); L scales ∝ 1/gap |
| L_stk | Stack axial length | Cross-section | Linear (ψ_f ∝ L_stk) |
| B_rem | Magnet remanence (grade) | Geometry | Exactly linear (B₁ = C(geometry) × B_rem) |
| k_w | Winding factor (bounded to [0.75, 1.0]; skipped outside) | Geometry, B_rem | Linear in ψ_f-derived metrics; no effect on B₁ |

> **Why B_rem is exactly linear:** Zhu's K_B(n) (Eq 17) is proportional to M_n
> (Eq 10b), and M_n = (B_rem/μ₀) × geometric factor (Eq 7a). So B_r(θ) = C(geometry) × B_rem × Σ cos(npθ).
> Changing B_rem scales the entire waveform uniformly.

> **Why proportional radius scaling gives 0% B₁ change:** K_B depends only on
> radius RATIOS (R_r/R_m, R_m/R_s, etc.). Scaling all radii by the same factor
> preserves all ratios. The OD perturbation is NOT proportional — it shifts
> stator and rotor outward together (frame-size model), holding air gap,
> magnet thickness, and shaft fixed, so the bore grows faster than the OD.

## Cross-Validation

| Term | Meaning |
|------|---------|
| cross-validation | Comparing model outputs against measured/published data or each other |
| ComparisonRow | Single quantity comparison (quantity, values, delta, tolerance, pass/fail) |
| compare_results | Pairwise comparison function → list of ComparisonRow |
| compare_all | All-pairs comparison across multiple results |
| diagnose | Returns verdict: "validated", "models agree", "models disagree", etc. |
| DiagnosisSummary | Structured result with delta/bound/curve row lists |

### Comparison types (precedence order)

| Type | Mechanism |
|------|-----------|
| bound | Inequality constraint: computed ≥ measured (or other direction). Margin in %. |
| curve | Interpolate/extract (interp, max, min, rms) from computed curve at measured point |
| key_mapping | Direct scalar-to-scalar mapping with per-quantity tolerance |
| delta | Fallback: direct difference on shared metric keys |

### Tolerance tiers

| Tier | Typical range | When used |
|------|---------------|-----------|
| Per-dataset | varies | Overrides in individual measured-data JSON (highest priority) |
| Model-to-model | 1–10% | Comparing computed vs computed (analytical, fem, drive_sim) |
| Model-to-published | 3–15% | Comparing computed vs literature values |
| Model-to-measured | 5–20% | Comparing computed vs lab measurements |
| Default | 10% | Fallback when quantity not in any table |

## Reference Motors

| Name | Topology | Poles/Slots | n_p | Key validation |
|------|----------|-------------|-----|----------------|
| CREATOR Case PMSM | Inrunner | 4p/6s | 2 | Back-EMF: 47.91 V computed vs 47.37 V measured (1.1%) |
| Belkhadir 22p/24s ER-PMSM | Outrunner | 22p/24s | 11 | Air-gap B₁ ≈ 1.026 T (analytical) / 1.020 T (FEM) |
| Awan 2.2-kW IPM | Inrunner (salient) | 6p | 3 | Torque curve, MTPA angle |
| Zhu & Howe 8-pole | Both | 8p/smooth | 4 | Paper reference geometry (analytical verification) |

## Data & Result Types

| Type | Purpose |
|------|---------|
| RunConfig | Motor + solver parameters for a single model run |
| RunResult | Output from any model run (config, status, metrics, elapsed, source) |
| MeasuredResult | Imported lab/published data with MeasurementConditions and comparison metadata |
| ResultStore | JSONL persistence layer for computed results |
| config_id | Three-tier hash: Geometry.config_id (geometry fields) → Motor.config_id (geometry.config_id + electrical/material fields) → RunConfig.config_id |
| run_id | Hash of config_id + model + solver-specific hash_fields |

## Report Sections

| # | Section | What it shows |
|---|---------|---------------|
| 1 | Overview | Reference motor catalog + model eligibility matrix |
| 2 | Per-Motor Analysis | Per motor: model outputs, analytical vs FEM, linear vs nonlinear FEM, cross-validation, sensitivity analysis |
| 3 | Cross-Motor Summary Tables | Air-gap B₁, linear vs nonlinear FEM, rated torque, stall torque, drive simulation |
| 4 | Infrastructure | Registry/run statistics |
| — | Appendix: Timing Analysis | Per-model elapsed-time breakdown |

## Plot Types

| Figure | What it shows |
|--------|---------------|
| br_waveform_*.png | B_r(θ) waveform + harmonic bar chart (analytical vs FEM) |
| br_peak_*.png | FEM B_r(θ) with published peak overlay |
| mtpa_gamma_*.png | MTPA angle γ vs stator current I_s with measured points |
| deviation_*.png | Computed/published ratio per quantity with tolerance bands |
| geometry_*.png | Motor cross-section sketch from Geometry |
| field_*.png | Rasterized FEM field solution (|B| over the cross-section) |
| sensitivity_*.png | Response change (%) vs parameter change (%) per track |

