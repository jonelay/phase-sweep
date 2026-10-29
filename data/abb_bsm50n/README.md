# ABB BSM50N-233

Brushless AC servo motor, surface PM, 4-pole, WYE. Multi-voltage
torque-speed envelope anchor: rated speed at two voltages
(300 V / 160 V) scales exactly with supply voltage.

## Source

ABB, "AC Servo Motors" brochure 9AKK106417 (Dec 2015), BSM N-series
data table column BSM50N-233.

## Files

| File | Content |
|------|---------|
| `abb_bsm50n233.toml` | Motor definition with derived circuit parameters |

## Notes

- Brochure prints all parameters as line-to-line (phase-to-phase).
- A documented voltage discrepancy exists between the series overview
  table (320 V) and the per-model table (300 V); 300/160 V is used
  based on the exact speed ratio match.
