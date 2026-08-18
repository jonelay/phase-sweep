# Toyota Prius 2004 (Gen2 THS-II) — 50 kW IPM traction motor

8-pole, 48-slot, V-shape NdFeB interior magnets, distributed winding.
A widely studied IPM benchmark in the electric-machine literature —
the only generation with a complete public geometry + measurement chain.

## Sources

All primary data is from freely available US government reports:

| Report | OSTI | Contents |
|--------|------|----------|
| ORNL/TM-2004/137 | 885676 | Physical teardown: geometry, winding, materials |
| ORNL/TM-2004/185 rev. 2007 | 921782 | Measured back-EMF vs speed and temperature |
| ORNL/TM-2004/247 | 890029 | Dynamometer performance maps |
| ORNL/TM-2004/217 | — | ORNL's own parametric FEA |

Circuit parameters consistent with Kuptsov et al., *Machines* 7(4):75, 2019
(DOI [10.3390/machines7040075](https://doi.org/10.3390/machines7040075)) —
the published FEM-vs-ORNL validation template.

## Files

- `pyleecan_geometry.json` — Machine-readable IPM geometry from
  [Pyleecan](https://github.com/Eomys/pyleecan) (Apache-2.0). Parked for
  future IPM geometry ingest; not consumed by the current solver.
- `torque_rated.json` — ORNL locked-rotor torque at 250 A (340 Nm,
  TM-2004/185 rev. 2007 p. 7) for bound comparison against stall_torque.

## Declared assumptions

Following the Prius community-standard calibration practice:

- **Magnet grade / B_rem**: never disclosed by Toyota. The community
  calibrates ψ_f (or B_rem) to ORNL's measured back-EMF. The Pyleecan
  model uses B_rem = 1.24 T at 20 °C (alpha_Br = -0.001 /K).
- **Inductances**: strongly saturation-dependent for V-IPM. The TOML
  values (L_d = 2.26 mH, L_q = 6.15 mH) are from Kuptsov's FEM d-q
  parameterization at moderate load. At peak current (~250 A),
  L_d drops to ~1.8 mH and L_q to ~3 mH (Kuptsov Fig. 5).
  The linear model over-predicts reluctance torque at high current.
- **Steel**: Pyleecan uses M400-50A; some papers assume M19-class.
  Stacking factor 0.95 (Pyleecan Kf1).
- **Bridge dimensions**: traced from teardown photos, not engineering
  drawings — they dominate leakage, and circulating DXFs differ.

## License

`pyleecan_geometry.json` is from the Pyleecan project, licensed under
Apache-2.0 — see https://github.com/Eomys/pyleecan/blob/master/LICENSE.

Other data files contain parameter values transcribed from the cited
publications and are covered by the MIT license.
