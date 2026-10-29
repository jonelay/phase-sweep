# Beckhoff AM8051-E

Synchronous servo motor, 8-pole. Multi-voltage torque-speed envelope
anchor: S1 nominal point tabulated at four mains voltages
(115/230/400/480 VAC), all below the 9000 rpm mechanical cap.

## Source

Beckhoff, "AM8000 and AM8500 | Synchronous Servomotors" Operation
Instructions, Version 4.9.1 (2023-09), section 4.7, column 51E.

## Files

| File | Content |
|------|---------|
| `beckhoff_am8051e.toml` | Motor definition with derived circuit parameters |

## Notes

- Corner voltages are mains AC (not DC bus); U_DC = sqrt(2) * U_N is
  a documented assumption unique to this anchor.
- KT is rated hot (100 K overtemperature) while KE is at 20 C; the
  -14.4% gap matches reversible NdFeB loss.
