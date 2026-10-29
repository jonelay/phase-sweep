# Belkhadir 22p/24s External Rotor PMSM

1.5 kW outer-rotor PMSM, 22-pole, 24-slot, FSCW, NdFeB N38SH magnets.
Validation data from FEM results published in the paper.

## Source

Belkhadir et al., "Electromagnetic Design and Optimization of a New
22-Pole/24-Slot External Rotor PMSM," IECON 2023.
DOI: [10.1109/IECON51785.2023.10312419](https://doi.org/10.1109/IECON51785.2023.10312419)

## Files

| File | Content |
|------|---------|
| `airgap_flux_published.json` | Published air-gap flux density data |
| `torque_rated.json` | Rated torque reference data |

## Notes

- Motor definition is in `motors/belkhadir_outrunner.toml`; this
  directory holds validation reference data only.
