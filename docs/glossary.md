# Glossary

Terms, symbols, and abbreviations used across phase-sweep's models,
TOML fields, result keys, and validation report.

---

## Conventions

**Peak vs RMS:** PM-tier current and flux-linkage values follow motulator's
peak-value convention; outputs with an `_rms` suffix (e.g. `k_T_rms`,
`I_s_rms`) are the RMS companions. Induction-tier voltages and currents
are per-phase RMS, and its power and torque are whole-machine totals (see [Induction-Machine Circuit Tier](#induction-machine-circuit-tier-im_-models)). TOML files may specify `I_rated_rms`
which is converted to peak via `I_rated = I_rated_rms × √2` at load time
(`machines/configs.py`). Torque formulas use the 3/2 factor that pairs with
peak values: `τ = 1.5 × n_p × (ψ_f × i_q + (L_d − L_q) × i_d × i_q)`.

**Units:** All geometry values are in SI (meters), not millimeters or
normalized radii. The Geometry dataclass enforces physical ordering.
Temperatures are °C for absolute values and K for differences.

**"Peak" means amplitude** in symbols and units (`A_peak`, `I_s_peak`).
The traction-rating sense (peak power, a short-time capability) is
covered under [Thermal](#thermal).

---

## Motor Types

| Term | Meaning | Source / mapping |
|------|---------|-----------------|
| PMSM | Permanent Magnet Synchronous Machine | — |
| SPMSM | Surface-mounted PMSM (non-salient: L_d ≈ L_q) | Zhu & Howe: surface-mounted assumed throughout |
| IPM / IPMSM | Interior PM machine (salient: L_q > L_d) | motulator: SynchronousMachinePars supports L_d ≠ L_q |
| ER-PMSM | External-Rotor PMSM (outrunner topology) | Zhu Eq 18: external rotor motor, R_s < R_m < R_r |
| Inrunner | Rotor inside stator (conventional layout) | Zhu Fig 1(a): "Internal rotor" |
| Outrunner | Stator inside rotor (e.g. Belkhadir 22p/24s) | Zhu Fig 1(b): "External rotor" |
| FSCW | Fractional-Slot Concentrated Winding | — (Zhu §IV covers nonoverlapping windings) |
| SCIM / IM | Squirrel-Cage Induction Machine (`[induction]` TOML section; `motor.induction is not None`) | CREATOR IM anchor (`data/creator_case_im/`) |

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
| magnet_overhang | m | Magnet axial extension per side beyond L_stk; input to the offline 3D k_end solve, not used by the 2D solvers | — |
| n_slots | — | Number of stator slots | Q_s (Table I) |
| slot_depth | m | Slot depth perpendicular to bore | — (Zhu uses b_o for slot opening) |
| slot_width_ratio | — | Slot body width as fraction of slot pitch | — |
| slot_opening_width | m | Slot opening width at the bore (stored field; Carter input) | Zhu: b_o |
| slot_opening_ratio | — | Derived: slot_opening_width / slot pitch at bore | — |
| topology | — | `"inrunner"`, `"outrunner"`, or `"axial"` | "internal" / "external" rotor |
| α_p | — | Magnet pole-arc / pole-pitch ratio (default 1.0) | Zhu Eq 7a: α_p. On Motor, AnalyticalParams, FemParams. See note below. |

> **α_p scaling (Zhu Eq 7a).** For the fundamental harmonic, the ratio of
> partial-pitch to full-pitch magnetization is exactly `sin(πα_p/2)`.
> **Analytical:** the harmonic source applies `sin(kπα_p/2)` to each odd harmonic k.
> **FEM:** the mesh models discrete magnet arcs from unscaled `B_rem` and
> `alpha_p`, so the geometry captures the inter-magnet gaps. An interpole gap
> narrower than 0.5 mm raises `ValueError` (`fem_field.solve_field_fem`);
> pass `alpha_p = 1.0` for a full ring.

> **Outrunner radius ordering** flips: r_outer > r_rotor > r_magnet > r_stator > r_inner.
> The names stay the same — r_stator is always the bore, r_magnet always the PM/air
> boundary — but their geometric relationship reverses.

## Magnetic & Material Properties

| Symbol | Units | Meaning | Zhu / motulator mapping |
|--------|-------|---------|------------------------|
| B_rem | T | Permanent magnet remanence | Zhu: B_r (Eq 2, 7a); `B_rem` avoids collision with the B_r(θ) waveform |
| psi_f (ψ_f) | Vs (Wb) | PM flux linkage per phase (peak value) | motulator: `psi_f` on SynchronousMachinePars |
| mu_r_pm (μ_r,pm) | — | Relative recoil permeability of magnet | Zhu: μ_r (Eq 1b, assumed constant) |
| mu_r_fe (μ_r,fe) | — | Relative permeability of stator/rotor iron | Zhu: assumes μ_r,fe → ∞ (Eq 11a–b); our FEM relaxes this |
| MU0 (μ₀) | H/m | Permeability of free space (4π × 10⁻⁷) | Zhu: μ₀ (Eq 1a) |
| n_p | — | Number of pole pairs | Zhu: p (Eq 6a); motulator: `n_p` |

> **Zhu assumes infinite iron permeability** (boundary conditions Eq 11a–b impose
> H_θ = 0 at iron surfaces). Our FEM solver relaxes this via finite mu_r_fe and
> optional B-H curve with damped Newton iteration by default and Picard
> startup/fallback.

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
| saturation_ratio | — | I_stall / I_rated: how far beyond rated the stall current is |
| saturation_warning | — | Boolean flag: true when saturation_ratio > 3.0 and linear model is unreliable |
| k_T | N·m/A_peak | Torque constant = 1.5 × n_p × ψ_f (peak current convention) |
| k_T_rms | N·m/A_rms | RMS torque constant = k_T × √2 |
| k_T_effective | N·m/A_peak | Effective torque per ampere at MTPA = τ / I_s (includes reluctance torque; equals k_T for non-salient) |
| MTPA | — | Maximum Torque Per Ampere (Morimoto 1994 quadratic) |
| γ (gamma) | rad | MTPA angle from q-axis (0 for non-salient; < 0 for reverse saliency, magnetizing i_d) |
| gamma_opt_deg | deg | Optimal MTPA angle at I_rated |
| I_s | A_peak | Stator current magnitude (peak, per motulator convention) |
| I_rated | A_peak | Rated continuous stator current from the datasheet; "continuous" is the source's definition (see [Thermal](#thermal)). Peak amplitude; TOML may specify I_rated_rms |
| I_stall | A_peak | Stall current = drive limit (I_LIMIT, else MAX_I_S); electromagnetic limit U_DC / (√3 × R_s) reported separately as `I_stall_electromagnetic` (with `tau_stall_electromagnetic`) |
| i_d | A_peak | d-axis current = −I_s sin(γ) (field-weakening) |
| i_q | A_peak | q-axis current = I_s cos(γ) (torque-producing) |
| L_d | H | d-axis inductance |
| L_q | H | q-axis inductance |
| saliency | — | L_q/L_d ratio; >1 = salient (IPM), ~1 = non-salient |
| i_d_curve_peak / i_q_curve_peak | A_peak | `torque_speed`: envelope currents at each `speed_curve` point (`_cont` at I_rated) |
| w_mech / tau_ref | rad/s, N·m | `operating_point` demand on the RunConfig (mechanical speed, torque) |
| I_s_peak / I_s_rms | A_peak / A_rms | `operating_point`: stator current magnitude, peak and rms |
| u_d / u_q / u_mag / u_max | V_peak | `operating_point`: dq voltage components, magnitude, and the limit U_DC/√3 |
| m_index | — | `operating_point`: modulation index 2·u_mag/U_DC (≤ 1 SPWM linear, ≤ 2/√3 SVPWM linear) |
| pf | — | `operating_point`: power factor cos∠(u, i); `None` at zero current |
| regime | — | `operating_point`: `"mtpa"` (voltage-feasible MTPA point) or `"field_weakening"` |
| op_source | — | `operating_point`: `"flux_map"` or `"analytical"` (constant-L) surface used |
| R_s_op / psi_f_op | Ω, Wb | `operating_point`: resistance and PM flux at the temperatures the solve used: its `winding_temp` argument (the motor's winding_temp_limit by default) and magnet_temp, or the node temperatures when `motor_thermal` drives the solve |
| p_cu / p_shaft / p_elec | W | `operating_point`: copper loss 1.5·R_s_op·I_s², shaft power tau·w_mech, their sum (no iron loss) |
| psi_d / psi_q | Wb_peak | `operating_point`: dq flux linkages the voltages were built from (constant-L or map) |

> **Peak convention:** see Conventions section above. motulator's
> `i_s_max` on CurrentVectorControllerCfg is also peak amps.

## Thermal

Two motor-side thermal models: `thermal_duty` (one winding-to-ambient
resistance, continuous loss budget and first-order transient) and
`motor_thermal` (a three-node lumped network coupled to
`operating_point`).

| Symbol | Units | Meaning |
|--------|-------|---------|
| winding_temp_limit | °C | `[thermal]` maximum winding temperature (datasheet θmax or a thermal-class value). Circuit models evaluate R_s at it, the worst-case convention; `motor_thermal` evaluates R_s at its solved winding temperature and reports headroom against the limit. A limit is only comparable with a model temperature on the same measurement basis (below) |
| winding_temp_limit_basis / _source | — | `[thermal]` tags on the limit: basis `thermometer`, `resistance`, `ETD` or `hot_spot`; source `published`, `measured`, `computed`, `fitted` or `assumed`. Unhashed |
| magnet_temp_max | °C | `[thermal]` magnet grade maximum operating temperature: a limit only, never a derating temperature. Same `_basis` / `_source` tags |
| insulation_class | — | `[thermal]` thermal class label, IEC 60085: 130 (B), 155 (F), 180 (H), 200 (N), 220 (R). Descriptive; not parsed into a limit |
| magnet_temp | °C | `[thermal]` magnet operating temperature for the ψ_f / flux-map derating. An operating assumption, not a limit |
| r_th / ambient_temp | K/W, °C | `[thermal]` winding-to-ambient thermal resistance and ambient for the `thermal_duty` budget |
| thermal_time_constant | s | `[thermal]` winding time constant for the `thermal_duty` transient march |
| p_s1_budget / within_s1 / over_budget_ratio | W, —, — | `thermal_duty`: continuous (S1) loss budget (`budget_source` names the path that set it), the verdict, and consumed loss over budget. This `within_s1` compares loss with a budget at the rated point; `motor_thermal`'s compares node temperatures with limits at the demanded point. Both are S1 equilibrium verdicts |
| winding_temp_peak / within_s1_transient | °C, — | `thermal_duty`: peak winding temperature of the first-order transient march and its verdict |
| T_coolant | °C | `motor_thermal` coolant temperature on the RunConfig (an operating condition, not a Motor field). Industry specifications state coolant *inlet* temperature; whether a fitted network's reference was inlet or mean depends on its source |
| T_winding / T_iron / T_magnet | °C | `motor_thermal`: steady node temperatures of the three-node network; T_magnet = T_iron at steady state |
| G_WFe / G_FeS / G_FeM | W/K | `[thermal_network]` conductances winding→iron, iron→coolant, iron→rotor (G = 1/R_th) |
| C_W / C_Fe / C_M | J/K | `[thermal_network]` node heat capacities |
| psi_s / psi_ratio | Wb_peak, — | `motor_thermal`: stator flux linkage magnitude and its ratio to the cold reference (map ψ_d(0,0) or psi_f) that scales B_core in the iron loss |
| iterations / residual_K | —, K | `motor_thermal`: fixed-point iterations to converge and the last unrelaxed temperature update |
| headroom_winding_K / headroom_magnet_K | K | `motor_thermal`: node limit minus node temperature; negative is over the limit (a result, not an error); `None` without a limit |
| binding_limit / within_s1 | —, — | `motor_thermal`: the node with the least headroom (a tie reports `"winding"`), and whether no headroom is negative (the S1 equilibrium verdict, named as in `thermal_duty`); `None` with no limits |
| winding_basis / magnet_basis | — | `[thermal_network]` measurement basis of a fitted node; a limit on a different basis is rejected |

### Industry terms

| Term | Source / usage | In phase-sweep |
|------|----------------|----------------|
| LPTN (lumped-parameter thermal network) | Motor thermal design (Motor-CAD; Boglietti et al.) | `[thermal_network]` is a three-node LPTN; `thermal_duty` is the one-node case |
| Thermal resistance R_th / conductance G | Datasheets, thermal-network tools (`Rth_*`, K/W) | `[thermal_network]` stores conductances (W/K); `[thermal] r_th` is a resistance |
| Measurement method: thermometer, resistance, ETD | IEC 60034-1 (resistance = winding average; ETD = embedded temperature detector), with a permitted temperature rise per method; IEC 60085 thermal class rates the insulation system | The basis of a winding temperature. The BMW i3 winding node is fitted to a trace from a coil-mounted sensor, so its basis is taken as ETD |
| Hot spot | Design allowance and thermal-model output (e.g. Motor-CAD winding hot spot); not a measurement method | Above any measured winding temperature; a hot-spot value and a measured one are not directly comparable |
| Coolant inlet temperature | IEC 60034-1; automotive component specifications | `T_coolant` |
| S1 duty | IEC 60034-1: continuous operation at constant load to thermal equilibrium | `thermal_duty` budget; `motor_thermal` steady solve |
| Continuous power | SAE J2907: maximum power held for 30 min | Not S1. A rating labeled "continuous" should name which definition it uses |
| Peak power | SAE J2907 maximum (net) power; manufacturers state a duration (BMW i3: 125 kW for at most 30 s) | A short-time capability; unrelated to peak amplitude |
| Derating | Automotive: the controller reduces power when a temperature limit is approached ("power reduction") | Not modeled. `solver_params.derate` is a different sense: the temperature correction of R_s and ψ_f |
| Thermal runaway | Loss rises with temperature faster than heat removal; no steady state exists | `motor_thermal` raises `ThermalRunaway`, `InfeasibleDemand` or `NoConvergence`; a limit exceedance is not an error but negative headroom |
| Losses / heat source | Motor design / FEM and thermal-network tools | `p_cu`, `p_fe` are the heat sources of the winding and iron nodes |

## Induction-Machine Circuit Tier (im_* models)

Voltages and currents are per-phase RMS (suffixed `_rms`), not the PM
tier's peak convention. Power and torque are whole-machine (three-phase)
totals: `p_ag = 3·|I_2|²·R_2/s`, `tau = p_ag/ω_s`.

| Symbol | Units | Meaning |
|--------|-------|---------|
| s (slip) | — | (n_sync − n) / n_sync; RunConfig.slip, motoring range (0, 1] |
| f_rotor_Hz | Hz | Rotor (slip) frequency s·f — the deep-bar table key |
| R_2 | Ω | Rotor resistance referred to the stator (DC reference; R_2_vs_f table above it) |
| tau | N·m | Machine electromagnetic torque = p_ag / ω_s (ω_s = 2πf/n_p mechanical synchronous) |
| tau_breakdown / slip_breakdown | N·m / — | Swept curve maximum (deep-bar-aware); `_thevenin` variants are the constant-parameter closed form |
| tau_starting / I_locked_rotor_rms | N·m / A_rms | s = 1 point of the sweep |
| p_ag / p_cu_s / p_cu_r / p_mech | W | Three-phase totals: air-gap power, stator/rotor copper loss (p_cu_r = s·p_ag), gross mechanical power |
| eta_circuit | — | p_mech / p_in: circuit losses only (no iron, friction, windage; documented over-estimate) |

## Drive Simulation

| Symbol | Units | Meaning | motulator equivalent |
|--------|-------|---------|---------------------|
| W_REF | rad/s | Motor reference speed (**mechanical**) | — (our convention; TOML comments confirm "rpm mechanical") |
| U_DC | V | DC bus voltage | `u_dc` on VoltageSourceConverter (lowercase in motulator) |
| MAX_I_S | A_peak | Maximum stator current | `i_s_max` on CurrentVectorControllerCfg |
| tau_peak | N·m | Peak torque during acceleration transient | — (extracted from tau_M time series) |
| i_ss | A_peak | Steady-state stator current magnitude | — (extracted from |i_s_ab| time series) |
| speed_droop | — | Fractional speed dip after load step (w_dip / w_ref) | — |
| t_settle | s | Time from speed step to within 5% of target | — |
| w_M | rad/s | Mechanical rotor speed | motulator: `w_M` on MechanicalSystem |
| w_m | rad/s | Electrical rotor speed = w_M × n_p | motulator: local `w_m = par.n_p * inp.w_M` in rhs() |
| tau_M | N·m | Electromagnetic torque | motulator: `tau_M` |
| i_s_ab | A (complex) | Stator current in αβ frame (peak) | motulator: `i_s_ab` |
| J | kg·m² | Moment of inertia | motulator: `J` on MechanicalSystem |

> **backemf_fundamental:** mechanical speed is converted to electrical
> speed before applying `E = ω_e × ψ_f`. Numerical comparisons are
> in the validation report.

## Winding Parameters

| Symbol | Units | Meaning | Zhu equivalent |
|--------|-------|---------|----------------|
| N | — | Turns per coil (single slot) | — (Zhu's W is total series turns per phase) |
| N_eff | — | Effective series turns per phase = N × coils_series | Zhu: W (Table I: "Series turns/phase") |
| k_w | — | Winding factor (= k_d × k_p in general) | Zhu: K_dpν (Eq 22, combined distribution × pitch factor) |
| coils_series | — | Coils in series per phase (required explicitly when N/k_w are set; no auto-derivation) | — |
| R_s | Ω | Stator winding resistance | motulator: `R_s` on SynchronousMachinePars |
| backemf_fundamental | V_peak | Fundamental phase back-EMF (peak) = ω_e × ψ_f | — |
| k_end | — | 3D axial end-effect factor on B_rem-derived flux linkage; computed offline by `scripts/compute_k_end.py` (never applied to a measured ψ_f). Bounds are documented in the motor TOML guide | — |

> **N vs W:** Zhu's W = total series turns per phase. Our N = turns per coil.
> The bridge is N_eff = N × coils_series ≈ W; motor-specific values belong
> in the corresponding TOML provenance comments.

## FEM Solver

| Term | Meaning |
|------|---------|
| NGSolve | Open-source FEM library (Netgen/NGSolve, LGPL) for 2-D magnetostatics |
| ElmerFEM | Independent open-source FEM code (CSC) used only to cross-verify the `fem`/`fem_axial` solvers via pinned comparisons (`scripts/elmer_*/`); never imported by the package |
| maxh_fraction | Max mesh element size as fraction of r_outer; default 0.05 (`RunConfig.maxh_fraction`) |
| n_theta | Angular sampling points in air-gap; default 360 (`RunConfig.n_theta`) |
| nonlinear | Enables the B-H curve; damped Newton is the default nonlinear method, with Picard startup/fallback |
| Newton iteration | Line-search-damped nonlinear solve using the B-H tangent; the v0.5 default |
| Picard iteration | Fixed-point iteration updating reluctivity from local \|B\|; used for Newton startup/fallback or when explicitly selected |
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
| α_p | Zhu Eq 7a: magnet pole-arc / pole-pitch ratio. Configurable on Motor (default 1.0). Scaling: odd harmonic k of the square-wave magnetization carries (4/(kπ))·sin(kπα_p/2) (`field_solution.py`, analytical v4); the fundamental factor is sin(πα_p/2). |

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
| R_2, L_m, L_sigma_s, L_sigma_r | Induction-circuit scalar + its matching table (scaled together) | Geometry, other circuit params | Accepted by `perturb_motor` for `[induction]` motors — fittable in `calibrate()` and swept on the validation report's `τ_rated_im` track; None for synchronous machines |

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
| cross-validation | Umbrella for the comparison machinery (`phasesweep.validation.crossval`, `phasesweep-crossval`, `tests/crossval/`). Covers the two kinds below, which carry different evidence |
| validation (against data) | Computed result compared with a measured or published record. Only this kind can yield the verdict "validated", and only with a two-sided (delta or curve) row; bound-only passes read "within bounds only" |
| model agreement | Two computed results compared with each other (e.g. analytical vs FEM, linear vs nonlinear FEM). Agreement shows consistency, not correctness; verdicts are "models agree" / "models disagree" |
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

### Tolerance ownership

Per-dataset tolerances take precedence. Shared policy and fallback values are
defined by `phasesweep.validation.crossval`; they are not duplicated in the
glossary.

## Reference Motors

The reference motors, their sources, and the quantities compared are listed
in the [validation report](validation-report.md).

## Data & Result Types

| Type | Purpose |
|------|---------|
| RunConfig | Motor + solver parameters for a single model run |
| RunResult | Output from any model run (config, status, metrics, elapsed, source) |
| MeasuredResult | Imported lab/published data with MeasurementConditions and comparison metadata |
| ResultStore | JSONL persistence layer for computed results |
| config_id | Three-tier hash: Geometry.config_id (geometry fields) → Motor.config_id (geometry.config_id + electrical/material fields) → RunConfig.config_id |
| run_id | Hash of config_id + model + solver-specific hash_fields |

## Agent Access (MCP)

Tool reference and usage: [mcp.md](mcp.md).

| Term | Meaning |
|------|---------|
| MCP | Model Context Protocol; `phasesweep-server` serves it at `/mcp` for AI agents |
| headline | The 2–5 outputs that summarize one result (`ModelInfo.headline`), each with its stored unit |
| caveat | `{code, severity, text, basis}` derived at read time from a result's provenance outputs; `basis` names the triggering output |
| info / warn | Caveat severities; warn means the headline is likely optimistic and should be reported with the caveat |
| limits | Model-inherent omissions (`ModelInfo.limits`), reported as `limits` info caveats on every OK result |
| applicability | Whether a model's input validation passes on a given motor (`list_models(motor=)`), with the reason when it does not |
| dry run | `submit_job(dry_run=True)`: the job's sub-task and cached counts and a run-time estimate, with nothing queued |
| sweep table | A sweep job read as rows of swept dimensions × headline metrics (`sweep_table`) |
| Agent tab | Dashboard tab showing the agent's design-space plot, tool calls, and finished jobs |

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
