# Kollmorgen Goldline B-104-B IPM

4-pole interior-PM (buried-magnet) servo with rotor saliency (L_q > L_d).
Model identity established by fingerprinting a thesis's measured parameters
against the datasheet columns.

## Sources

- **Electrical model:** University of Akron M.S. thesis, OhioLINK ETD
  accession akron1375726072, Section 4.1 + Table 4-1.
- **Nameplate + thermal:** Kollmorgen, "GOLDLINE Brushless Motor Series"
  datasheet, column B-104-B (Artisan Technology Group reprint).

## Files

| File | Content |
|------|---------|
| `kollmorgen_b104b_ipm.toml` | Motor definition with thesis-measured L_d/L_q/psi_f |
