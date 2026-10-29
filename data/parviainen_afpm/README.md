# Parviainen 5 kW AFPM prototype

Axial-flux SPM research prototype (12-pole, 1-rotor-2-stators TORUS
topology, 300 rpm). First axial-flux anchor with full measured
cross-validation (back-EMF, efficiency, cogging, thermal).

## Source

A. Parviainen, "Design of Axial-Flux Permanent-Magnet Low-Speed
Machines and Performance Comparison Between Radial-Flux and Axial-Flux
Machines," D.Sc. thesis, Lappeenranta University of Technology, Acta
Universitatis Lappeenrantaensis 208, 2005. Chapter 3, Tables 3.1/3.3,
Appendix A.3.

Also: Parviainen, Niemelä, Pyrhönen, "Modeling Axial-Flux
Permanent-Magnet Machines," IEEE Trans. Ind. Appl., 40(5), 2004,
pp. 1333–1340.

## Files

| File | Content |
|---|---|
| `parviainen_5kw_afpm.toml` | Motor definition with circuit parameters derived from measured back-EMF |

## Notes

The two stators are connected electrically in parallel in wye. The TOML
stores terminal-equivalent values (R, L halved from per-stator). See the
TOML header for the full parallel-stator convention.
