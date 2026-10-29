# Kollmorgen GOLDLINE B-series — two-ambient rating columns

25 windings from the GOLDLINE brushless servomotor series, each with
continuous stall torque at two ambient temperatures (25 C and 40 C).
Used to validate thermal_duty's winding-temperature budget path — all
25 converge to ~156 C winding temperature through the absolute-path
method.

## Source

Kollmorgen, "GOLDLINE Brushless Servomotor Series" datasheet (Artisan
Technology Group reprint), 10x/20x/40x PERFORMANCE DATA tables.

## Files

| File | Content |
|------|---------|
| `two_ambient_columns.toml` | 25 winding entries with Tc, Ic, Kt, Rm, Km, Rth at both ambients |

## Notes

No L_d/L_q/psi_f — this dataset validates thermal rating consistency,
not the electrical model.
