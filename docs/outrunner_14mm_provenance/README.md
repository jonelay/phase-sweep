# 14 mm outrunner provenance

Bench-measurement provenance for the `outrunner_14mm_steel` validation
motor. These files make the 14 mm outrunner validation points
reproducible from raw caliper / micrometer / gauge-pin / LCR / back-EMF
measurements within this repo.

- [`motor_parameters.md`](motor_parameters.md) — consolidated measurement
  provenance, geometry cross-checks, and the flux-linkage model history.
- [`derive_geometry.py`](derive_geometry.py) — self-contained (stdlib-only)
  derivation that reproduces the radii, air gap, and `alpha_p` in the motor
  TOMLs from the raw measurements. Run it directly:
  `python docs/outrunner_14mm_provenance/derive_geometry.py`.
