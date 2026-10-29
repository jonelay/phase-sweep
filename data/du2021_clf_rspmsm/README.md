# Du et al. 2021 CLF-RSPMSM

10 kW reverse-salient PMSM (L_d > L_q), 48-slot / 8-pole V-type IPM
with a controllable-leakage-flux rotor. FEM design study only — no
prototype built.

## Source

L. Du, X. Liu, J. Fu, J. Liang, C. Huang, "Design and Optimization
of Reverse Salient Permanent Magnet Synchronous Motor Based on
Controllable Leakage Flux," CES Transactions on Electrical Machines
and Systems, vol. 5, no. 2, pp. 163-173, June 2021.
DOI: [10.30941/cestems.2021.00020](https://doi.org/10.30941/cestems.2021.00020)

## Files

| File | Content |
|------|---------|
| `du2021_clf_rspmsm.toml` | Motor definition (circuit tier from Table VI) |

## Notes

- Reverse saliency ratio L_d/L_q = 1.08; L_q is derived from L_d and
  the ratio.
- No R_s or DC bus published; `[drive]` section omitted.
