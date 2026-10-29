# TEAM Problem 30a — induction motor benchmark tables

Source: Kent Davey, "Induction Motor Analyses", International TEAM
Workshop Problem 30 (COMPUMAG; problem sheet `problem30a.pdf` at
compumag.org). Tables I, II' and III transcribed verbatim from the
problem sheet. Analytical reference values, not measurements: this is
a code-verification benchmark, not a validation anchor.

| File | Content |
|---|---|
| `table1_three_phase.csv` | Table I: three-phase motor, torque / phase-A single-turn voltage / total rotor loss / rotor-steel loss vs speed, per metre of depth |
| `table2_field_profile_200rad_s.csv` | Table II': radial B and azimuthal H phasors on the +x axis between r3 and r4 at 200 rad/s |
| `table3_single_phase.csv` | Table III: single-phase motor, same columns as Table I |

Problem data (m, SI): rotor steel r < 0.02, rotor aluminium 0.02–0.03,
air gap 0.03–0.032, winding 0.032–0.052, stator steel 0.052–0.057.
Steel μr = 30 (rotor σ = 1.6e6 S/m, stator laminated σ = 0), aluminium
σ = 3.72e7 S/m, coils 310 A/cm² at 60 Hz, 45° belts; three-phase belts
A, -B, C, -A, B, -C on 60° centres with B and C lagging A by 120° and
240°; single-phase belts A, -A at 0° and 180°.

Two conventions the sheet leaves implicit were pinned by reproducing
the tables (`phasesweep/solvers/team30a.py`, `analytic_team30a`):
the current density is an rms phasor (torque and loss are time
averages, the voltage rms), and the exterior beyond the stator steel
is open air (no A = 0 boundary at r5). With both, the series solution
reproduces Table I to four significant figures.
