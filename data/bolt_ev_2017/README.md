# 2017 Chevrolet Bolt EV

150 kW 8-pole V-shape IPM traction motor designed by GM, produced by LG.
Bar-wound (hairpin) stator, sintered NdFeB in double-layer V arrangement.
Nameplate and geometry only — no publicly available circuit parameters.

## Sources

- SAE 2016-01-1228: "Electric Motor Design of General Motors' Chevrolet
  Bolt Electric Vehicle"
- McMaster University Dataverse teardown dataset.
  DOI: [10.5683/SP3/LM35Z2](https://doi.org/10.5683/SP3/LM35Z2)
- InsideEVs deep dive (motor weight, gear ratio, max RPM)

## Files

| File | Content |
|------|---------|
| `bolt_ev_2017.toml` | Motor definition (nameplate + geometry tier) |

## Notes

- L_d, L_q, psi_f, R_s are NOT publicly web-accessible. Circuit-tier
  solvers cannot run on this motor without extracting parameters from
  the McMaster FEA dataset or the paywalled SAE paper.
