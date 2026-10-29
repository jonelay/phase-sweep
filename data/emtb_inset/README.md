# Goh et al. e-MTB Surface-Inset PMSM

14-pole mid-drive prototype for electric mountain bikes. First
surface-inset magnet motor in the repo: tapered magnets recessed into
the rotor surface. Manufactured and dynamometer-tested at NTU Singapore.

## Source

Goh, Xie, Wang, Zhu, Yu, Lee (2025), "Development of a Surface-Inset
Permanent Magnet Motor for Enhanced Torque Density in Electric Mountain
Bikes," Energies 18(14):3709.
DOI: [10.3390/en18143709](https://doi.org/10.3390/en18143709)

## Files

| File | Content |
|------|---------|
| `emtb_inset_spm.toml` | Motor definition (geometry tier; no circuit parameters) |

## Notes

- R_s, L_d, L_q, psi_f are NOT published — this is a design paper, not
  a characterization paper.
- Measured performance at 500 r/min: 4.62 Nm peak torque (5.7% below FEA).
