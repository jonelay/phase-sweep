# Mahmoudi 10 kW TORUS AFPM

Double-sided slotted axial-flux SPM (1-stator 2-rotors, north-north
magnet configuration), 18s/6p. FEA-only (no prototype), but the most
complete cross-section geometry of any axial-flux motor in the
validation set. Primary target for future axial-flux analytical solver
cross-validation.

## Source

A. Mahmoudi, N. A. Rahim, H. W. Ping, "Axial-Flux Permanent-Magnet
Motor Design for Electric Vehicle Direct Drive Using Sizing Equation
and Finite Element Analysis," *Progress In Electromagnetics Research*,
Vol. 122, pp. 467–496, 2012.

## Files

| File | Content |
|------|---------|
| `mahmoudi_10kw_torus.toml` | Motor definition with full slot-profile geometry from Table 4 |

## Notes

Circuit parameters (psi_f, I_rated) are derived from FEA back-EMF, not
measured. No R_s, L_d, or L_q published.
