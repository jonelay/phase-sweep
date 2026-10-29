# EMRAX 208 Medium Voltage

Axial-flux PMSM, 20-pole, WYE, YASA-like topology. Second axial-flux
anchor — same topology as the 228 but smaller stator (208 mm OD).
Circuit-tier anchor only (no geometry tier for axial-flux yet).

## Source

EMRAX 208 datasheet v1.7 (Aug 2026), Medium Voltage column, LC cooling.
https://emrax.com/e-motors/emrax-208/

## Files

| File | Content |
|------|---------|
| `emrax_208_mv.toml` | Motor definition (circuit tier) |

## Notes

- EMRAX unit conventions differ from header labels; see the TOML header
  for the detailed convention mapping. psi_f is peak per-phase; R_s is
  phase-to-phase at 25 C divided by 2.
