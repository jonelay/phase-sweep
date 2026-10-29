# EMRAX 228 Medium Voltage

Axial-flux PMSM, 20-pole, WYE, single-stator dual-rotor (YASA-like
segmented armature). First axial-flux motor in the validation set.
Circuit-tier anchor only (no geometry tier for axial-flux yet).

## Source

EMRAX 228 datasheet v1.7 (Aug 2026), Medium Voltage column, LC cooling.
https://emrax.com/e-motors/emrax-228/

## Files

| File | Content |
|------|---------|
| `emrax_228_mv.toml` | Motor definition (circuit tier) |

## Notes

- Mild saliency (L_q > L_d by 2.9%) from concentrated windings.
- Ke/Kt header labels say Vrms/Nm-Arms but values are consistent with
  peak conventions — documented, not corrected.
