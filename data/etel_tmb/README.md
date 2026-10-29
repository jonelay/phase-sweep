# ETEL TMB+xxxx-030 — direct-drive torque motor family

Five winding variants of the ETEL TMB conventional torque motor series
(sizes 0140 through 0450, 22–88 poles), all reverse-salient IPMs
(L_d > L_q). Each TOML includes a companion back-EMF JSON derived from
the datasheet Ku constant.

## Source

ETEL S.A., "CONVENTIONAL TORQUE MOTOR TMB+xxxx-030" datasheets,
07.01.2019, water-cooling columns. One datasheet per frame size.

## Files

| File | Motor | Poles |
|------|-------|------:|
| `etel_tmb0140_030_ra.toml` | TMB+0140-030, winding RA | 22 |
| `etel_tmb0210_030_ta.toml` | TMB+0210-030, winding TA | 44 |
| `etel_tmb0290_030_ra.toml` | TMB+0290-030, winding RA | 44 |
| `etel_tmb0360_030_ta.toml` | TMB+0360-030, winding TA | 66 |
| `etel_tmb0450_030_va.toml` | TMB+0450-030, winding VA | 88 |
| `backemf_from_ku_*.json` | Synthetic back-EMF derived from Ku | — |
