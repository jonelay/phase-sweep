# Toyota Prius 2010 (Gen3 XW30) — 60 kW IPM traction motor

8-pole, 48-slot, V-shape NdFeB IPM. Complete ORNL teardown geometry but
no published circuit parameters (L_d, L_q, psi_f require FEA extraction).

## Source

Burress et al., "Evaluation of the 2010 Toyota Prius Hybrid Synergy
Drive System," ORNL/TM-2010/253, March 2011.
https://info.ornl.gov/sites/publications/files/Pub26762.pdf

## Files

| File | Content |
|---|---|
| `prius_2010.toml` | Motor definition — geometry-tier only (no d-q circuit parameters) |

## Notes

This is the generation after the 2004 Prius. Key differences: 60 kW
(vs 50 kW), shorter stack (50.8 mm vs 84 mm), higher speed (13,500 rpm
vs 6,000 rpm), higher boost voltage (650 Vdc vs 500 Vdc). Circuit-tier
solvers cannot run without L_d/L_q/psi_f from secondary FEA sources.
