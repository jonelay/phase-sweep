# Bozkurt PMASynRM — 1 kW outer-rotor

48-slot/8-pole PM-assisted synchronous reluctance motor with a
3-barrier TLA rotor and arc-shaped NdFeB magnets. Prototype
manufactured and tested at four load points (750 rpm).

## Sources

1. Bozkurt, Oner, Baba, "Manufacturing and Experimental Validation of
   an Outer-Rotor Permanent Magnet-Assisted Synchronous Reluctance
   Motor for In-Wheel Electric Vehicle Drive," *Machines* 2026, 14,
   729. DOI: [10.3390/machines14070729](https://doi.org/10.3390/machines14070729)
2. Bozkurt, Baba, Oner, "Design of Outer-Rotor Permanent-Magnet-
   Assisted Synchronous Reluctance Motor for Electric Vehicles,"
   *Energies* 2021, 14, 3739.
   DOI: [10.3390/en14133739](https://doi.org/10.3390/en14133739)

## Files

| File | Content |
|------|---------|
| `or_pmasynrm_1kw.toml` | Motor definition — nameplate + measured performance only |

## Notes

No circuit parameters (R_s, L_d, L_q, psi_f) published in either
paper. Cannot run d-q solvers — exists as a nameplate + measured-
performance reference for the PMaSynRM topology.
