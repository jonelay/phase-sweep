# BMW i3 (2016) — 125 kW IPM traction motor

12-pole, 72-slot, V-shape NdFeB IPM with segmented stator (6 sections)
and skewed rotor. Jacket-cooled (spiral-groove water channel on stator OD).

## Source

No standalone ORNL/TM technical report was published for the BMW i3 motor
(unlike the Prius teardowns). Motor-side data is distributed across DOE
merit-review presentations and the EETT Roadmap appendix:

1. T. Burress, "Benchmarking EV and HEV Technologies," DOE VTO Annual
   Merit Review, Project EDT006, June 7, 2016.
   https://www.energy.gov/sites/prod/files/2016/06/f32/edt006_burress_2016_o_web.pdf

2. T. Burress, "Electrical Performance, Reliability Analysis, and
   Characterization," DOE VTO Annual Merit Review, Project EDT087,
   June 6, 2017.
   https://www.energy.gov/sites/default/files/2017/06/f34/edt087_burress_2017_o.pdf

3. U.S. DRIVE EETT, "Electrical and Electronics Technical Team Roadmap,"
   October 2017, Appendix C: "ORNL Testing and Evaluation Data on 2016
   BMW i3."
   https://www.energy.gov/sites/prod/files/2017/11/f39/EETT%20Roadmap%2010-27-17.pdf

4. A. Tikadar, N. Kumar, Y. Joshi, S. Kumar, "Coupled Electro-Thermal
   Analysis of Permanent Magnet Synchronous Motor for Electric Vehicles,"
   19th IEEE ITherm 2020, pp. 249-256.
   OSTI: https://www.osti.gov/servlets/purl/1843426
   (Georgia Tech / ARPA-E OPEN 2018, Award DE-AR0001023)

5. M. Gierczynski, L.M. Grzesiak, "Comparative Analysis of the
   Steady-State Model Including Non-Linear Flux Linkage Surfaces and
   the Simplified Linearized Model when Applied to a Highly-Saturated
   Permanent Magnet Synchronous Machine—Evaluation Based on the Example
   of the BMW i3 Traction Motor," Energies 2021, 14(9), 2343.
   DOI: https://doi.org/10.3390/en14092343
   (Warsaw University of Technology, open access CC BY 4.0)

## Motor parameters

From ORNL teardown (Burress 2016 slides 21–23, 2017 slides 8–9, 16, 20):

| Parameter | Value | Source |
|---|---|---|
| Type | IPM, V-shape NdFeB | Burress 2016 slide 23 |
| Poles / slots | 12 / 72 | Burress 2016 slide 22 |
| Peak power | 125 kW | Burress 2016 slide 21 |
| Peak torque | 250 Nm | Burress 2016 slide 21 |
| DC bus voltage | 355 V nominal | Burress 2016 slide 19 |
| Stator assembly mass | 42 kg (as received) | Burress 2016 slide 21 |
| Stator mass | 20.8 kg | Burress 2016 slide 22 |
| Stator core mass | 13.7 kg | EETT Roadmap App. C, Table C-3 |
| Copper mass | 7.1 kg | EETT Roadmap App. C, Table C-3 |
| Rotor mass | 14.2 kg | Burress 2016 slide 23 |
| Magnets per pole | 2 (1 large, 1 small) | Burress 2016 slide 23 |
| Stator segments | 6 (staggered axially) | Burress 2016 slide 22 |
| Cooling | Jacket, spiral-groove water channel | Burress 2016 slide 22 |

From Gierczynski & Grzesiak (Ansys Maxwell 2D FEM model):

| Parameter | Value | Source |
|---|---|---|
| Stator OD | 242.1 mm | Gierczynski Table A1 |
| Stator bore (ID) | 180.0 mm | Gierczynski Table A1 |
| Rotor OD | 178.6 mm | Gierczynski Table A1 |
| Rotor ID (lam bore) | 130.3 mm | Gierczynski Table A1 |
| Shaft diameter | 60 mm | Gierczynski Table A1 |
| Air gap | 0.7 mm | Gierczynski Table A1 |
| Stack length | 132.3 mm | Gierczynski Table A1 |
| Tooth width | 5.0 mm | Gierczynski Table A1 |
| Slot opening | 1.7 mm | Gierczynski Table A1 |
| Slot depth | 22.0 mm | Gierczynski Table A1 |
| Magnet grade | NdFeB 38/23 | Gierczynski Table A3 |
| Lamination steel | M-19 29 Ga | Gierczynski Table A3 |
| Winding | 9 turns/coil, full pitch | Gierczynski Table A2 |
| Parallel circuits | 6 per phase | Gierczynski Table A2 |

From Tikadar et al. (Motor-CAD FEM thermal model, citing ORNL/Ozpineci):

| Parameter | Value | Source | vs Gierczynski |
|---|---|---|---|
| Stator OD | 240.90 mm | Tikadar p.2 | -1.2 mm |
| Stator bore | 179.76 mm | Tikadar p.2 | -0.24 mm |
| Slot depth | 21.5 mm | Tikadar p.2 | -0.5 mm |
| Stack length | 130 mm | Tikadar p.2 | -2.3 mm |
| Air gap | 0.5 mm | Tikadar p.2 (CFD) | **-0.2 mm** |
| Steel | M250-35A | Tikadar p.2 | **different** |
| Magnet grade | N38UH | Tikadar p.2 | same family |
| Rotor skew | 6 segments | Tikadar p.2 | confirmed |

## Performance data (ORNL-measured)

ORNL test conditions (Burress 2017 slide 16): 355 Vdc, 375 Arms, 125 kW.

- Motor/inverter/combined efficiency maps at 355 Vdc: EETT Roadmap
  Appendix C, Figure C-3 (motor peak >94%; yellow diamond markers are
  measured operating points; no tabular CSV — must digitize from contours)
- Motor continuous test at 7,000 rpm: Burress 2017 slide 9, three
  power steps. Digitized winding plateaus, each the mean of the 43
  samples in the last 300 s of the samples within 8 kW of the step
  (`scripts/fit_i3_thermal_network.py`, `plateau_windows()`):
  83.4 C at 25 kW (t = 1611-1908 s), 95.3 C at 50 kW (3946-4243 s),
  106.5 C at 75 kW (5919-6217 s); 63.2 C over the opening segment at
  7,000 rpm and under 1 kW shaft power (`noload_segment()`, t = -476 to
  -271 s). Coolant temperature and flow are not stated; the opening
  reading bounds the coolant from above, if the motor is at equilibrium
  there, but is not the coolant (the
  motor has no-load iron loss at 7,000 rpm; the fitted block predicts
  73.9 C there, +10.7 K, not fitted). See
  `burress_2017_slide9_continuous_test_7000rpm.csv`. The
  `[thermal_network]` block in `motors/bmw_i3.toml` is fitted to the
  three loaded plateaus at an assumed 60 C coolant;
  `scripts/check_i3_thermal_transient.py` marches the block through the
  whole trace.
- Motor power density: 9.1 kW/L, 2.9 kW/kg (Burress 2017 slide 8)
- Inverter power density: 18.5 kW/L, 14.1 kW/kg (Burress 2017 slide 8)

## Circuit parameters (FEA-extracted)

No ORNL source tabulates d-q circuit parameters. Gierczynski & Grzesiak
(Energies 2021, DOI 10.3390/en14092343) reconstructed the motor in Ansys
Maxwell 2D from the ORNL data, Hendershot's teardown video, and BMW's
Merwerth presentation, and published:

| Parameter | Value | Unit | Note |
|---|---|---|---|
| ψ_PM | 43.6 | mWb | at (i_d=0, i_q=0) |
| L_d | 71.2 | µH | secant at peak torque point |
| L_q | 141.3 | µH | secant at peak torque point |
| R_s | 5.3 | mΩ | phase resistance |
| Saliency | 1.98 | — | L_q / L_d |
| i_max | 565.7 | A peak | |

Full nonlinear flux linkage maps ψ_d(i_d, i_q) and ψ_q(i_d, i_q) are
tabulated in the paper (Tables A4-A5), 7×7 grids from 0 to ±600 A.
All values are FEA-extracted, not measured. Validated against ORNL
locked-rotor torque measurements.

Geometry discrepancies vs Tikadar et al. (see motor TOML header for details):
air gap 0.7 vs 0.5 mm, steel M-19 29 Ga vs M250-35A, stator OD +1.2 mm,
stack length +2.3 mm.

## V-shape magnet pocket geometry (from DXF)

Extracted from the Gierczynski supplementary AutoCAD .dxf
(energies-14-02343-s001.zip, CC BY 4.0). Each pole has two
rectangular pockets forming a V:

| Pocket | r_inner (mm) | r_outer (mm) | Width (mm) | Thickness (mm) |
|---|---|---|---|---|
| Inner ("large") | 73.0 | 78.9 | ~24 | ~5.9 |
| Outer ("small") | 84.3 | 87.5 | ~13 | ~3.2 |

12 inner + 12 outer = 24 total pockets (12 poles × 2 per pole).
Yoke above outer pocket to air gap: 89.3 − 87.5 = 1.8 mm (thin,
saturates at load). Yoke between pockets: 84.3 − 78.9 = 5.4 mm.

## What is NOT published

- Back-EMF waveform: not published as data or graph by any source.

## Files

| File | Content |
|---|---|
| `README.md` | This file |
| `BMWi3_GeometryAutoCAD.dxf` | Gierczynski supplementary: rotor+stator cross-section |
| `FEMResults_1000rpm.xlsx` | Gierczynski supplementary: torque, ψ_d, ψ_q, phase voltages vs (i_d, i_q) at 1000 rpm |
| `flux_maps_7x7.json` | Angle-averaged 7×7 ψ_d/ψ_q maps extracted from the xlsx (peak, phase, mWb) |
| `m19_29ga_core_loss.csv` | M-19 29 Ga typical core loss (W/kg) at 50–2000 Hz, 0.1–1.7 T; transcribed from the EMERF/MIT OCW steel data page. Catalog grade data, not an i3 measurement. CC BY-NC-SA source, so not redistributed in release archives; transcribe it from the source cited in `docs/references.md` |
| `burress_2017_slide9_continuous_test_7000rpm.csv` | Winding temperature (C) and shaft power (kW) vs time (s) at 7,000 rpm, digitized by pixel from Burress 2017 slide 9 (300 dpi render, gridline-calibrated, legend masked). Time is relative to the first axis label; durations are exact, absolute offset is not. ORNL measurement, chart-digitized |

Motor TOML: `motors/bmw_i3.toml` (geometry + circuit parameters +
`[iron]` Bertotti fit of the M-19 table; `B_core` is a geometric
estimate, uncalibrated).

The flux maps carry no extraction temperature (the paper does not state
the FEM magnet temperature). phase-sweep derates from a 20 °C reference
by default, which brackets the hot-magnet peak torque: at 100 °C magnets
the envelope at MAX_I_S is 243 N·m if the map is a 20 °C map and the
250 N·m nameplate holds if the map was extracted at 100 °C. At its own
temperature the map peaks at 258 N·m, 3 % over the nameplate; a 20 °C
map with magnets at about 63 °C reproduces 250 N·m exactly. Set
`[flux_map] magnet_temp_ref_C` if the extraction temperature is ever
recovered.
