# Actuator Aluminum Rotor — measured data

Second validation point for the actuator (aluminum-shell rotor variant, same
stator as `actuator_steel_rotor`). Tighter air gap (0.262 mm vs 0.528 mm).

## What's here

- `torque_constant_sweep.json` — distilled measurement: flux linkage
  `psi_f ≈ 0.31 mWb` as recorded (through-origin Kt; Ke ≈ 1.87 mV/(rad/s),
  Kt ≈ 2.81 mNm/A), via the **power-balance torque-constant** method.
  **Bias-corrected best estimate ≈ 0.30 mWb** (range 0.285–0.32): the
  through-origin fit carries ~20 mW of speed-independent inverter loss in the
  DC-bus power, inflating Kt ~5%.
- `telemetry/` — supporting raw ODrive telemetry (two no-load drag sweeps,
  2026-04-14, 10–80 rps, CW+CCW). Columns: `timestamp_s, speed_target_rps,
  direction, rps, Iq_A, Id_A, Vbus_V, P_elec_W, T_fet_C`.

## Why no back-EMF sweep (unlike the steel rotor)

The steel rotor was validated with a clean 3-channel **externally-driven,
open-phase** back-EMF speed sweep. That route was not repeated for the aluminum
rotor — but **not because the signal is too small**: aluminum's back-EMF
(1.41 V at 120 rps) actually *exceeds* steel's (1.21 V). The failure was the
*fixture* — aluminum was measured self-powered / coast-down (current present):

- **Disconnect coast-down** captures are dominated by 60 Hz mains pickup
  (~234 mV on the open phase lead) plus Vbus filter-cap discharge, with no
  peak at the expected electrical fundamental.
- **Powered no-load** terminal voltage is contaminated by the winding
  impedance drop `I·Z` (~1.16 V at 120 rps — comparable to the back-EMF
  itself), so an extracted Ke wanders 1.07–2.90 instead of staying flat.

> **The open-phase route is likely recoverable.** Running the aluminum rotor on
> the steel rotor's external-drive jig (zero phase current) would remove the
> I·Z and pickup contamination and give an independent clean anchor —
> superseding the Kt method. This is the recommended next bench step, not
> shielded-lead rework.

So `psi_f` here is derived from the **torque constant**, using only reliable
digital telemetry: at each no-load steady speed,
`T_em = (P_elec − 1.5·R_s·Iq²)/ω_mech`, then `Kt = slope(T_em vs Iq)` and
`psi_f = Kt/(1.5·n_p)`.

## Caveats

- Kt-derived, not back-EMF-derived; the method cannot be cross-validated
  against the steel rotor (steel has no no-load ODrive telemetry).
- Through-origin fit R² ≈ 0.88 indicates a fixed-loss bias (~20 mW inverter
  loss inside the DC-bus `P_elec`) that inflates Kt. A physical fit
  (`T_em = Kt·Iq + P_inv/ω`, R² 0.98–0.99) removes it and gives `psi_f ≈ 0.299`
  — so 0.312 mWb is a through-origin **upper** bound, ~5% high.
- CW/CCW differ ±5% (0.287 / 0.318 mWb by direction); the canonical value
  magnitude-averages them.
- Single rotor build, single day, two runs (0.304 / 0.321 mWb).

The cleanest path to a clean anchor is the open-phase external-drive sweep
above (not shielded-lead rework); alternatively a calibrated torque measurement.
