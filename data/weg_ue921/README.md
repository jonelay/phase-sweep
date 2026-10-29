# WEG UE 921 — IEC frame-80 4-pole PMSM family

Two production units in the same IEC frame-80 housing, dual-sourced
with the NORD 80T1/4.

## Source

WEG motor nameplates (Ma_882_C designation). The 712 unit has
LCR-measured electricals (2026-06-11); the 662 B unit has nameplate
data only.

## Files

| File | Content |
|---|---|
| `weg_ue921_662b.toml` | UE 921/662 B — nameplate reference, R_s cross-imported from NORD |
| `weg_ue921_712.toml` | UE 921/712 — "golden anchor" unit with LCR-measured parameters |

## Notes

R_s = 2.60 Ohm on the 662 B is a cross-vendor import from the NORD
80T1/4 datasheet (R_ww/2). The NORD is wound for 400 V/70 Hz/2100 rpm;
the WEG for 187 V/50 Hz/1500 rpm — different windings in the same
frame. The 712 unit's own LCR measurements supersede this for circuit
work. Neither unit has published L_d/L_q/psi_f from the manufacturer.
