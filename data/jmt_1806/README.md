# JMT 1806 2400KV

Measured reference data for a **JMT 1806 2400KV** — a quadcopter-class
outrunner (18 mm stator, 12N14P, ~16 g). A budget catalog point in the
1806-class winding cluster (shares the 18×6
stator with e.g. the KDE 1806XF-2350).

## What's here

- `reference.toml` — distilled motor definition + measured electricals.
- `backemf_speed_sweep.json` — per-speed distilled results (the table below).
- `captures/` — 8 raw oscilloscope CSVs (3-channel phase voltages), the
  back-EMF speed sweep. ~6.9 MB each.
- Analysis: [`scripts/analyze_jmt_1806.py`](../../scripts/analyze_jmt_1806.py) —
  reproduces everything below from `captures/`.

## Measurement

The motor was spun **passively at 1:1 by an external backdriver** (open-circuit
DUT — no phase current). Back-EMF was captured at 8 commanded speeds labeled
20/30/40/65/70/80/90/100 RPS. Three single-ended scope channels on the three
phase wires shared a common chassis ground; the wye center was inaccessible.
Because the motor is open-circuit, `V_Y − V_chassis` is a slow DC drift with no
fundamental content, so each channel's fundamental amplitude **is** the
per-phase (phase-to-neutral) peak back-EMF.

**Pole pairs `n_p = 7`** fall directly out of `f_elec / RPS` on the four
non-slipping files (20/30/40/90). The backdriver slips on the heavier files
(65/70/80/100 read low), but the Ke fit recovers true mechanical RPS from
`f_elec / n_p`, so slip does not bias the result.

## Results (from `analyze_jmt_1806.py`)

Through-origin linear fit of peak phase-to-neutral amplitude vs mechanical ω,
**R² = 0.99994**:

| Quantity | Value | Notes |
|----------|-------|-------|
| Ke (phase-to-Y, peak) | 2.310 mV/(rad/s) | |
| ψ_f (n_p = 7) | 330 µWb | peak-amplitude basis |
| Kt (FOC, 1.5·n_p·ψ_f) | 3.465 mNm/A | |
| KV (vendor convention) | 2387 rpm/V | vs catalog 2400 → **within 0.5%** |
| R_LL | 180 mΩ | LCR 1 kHz; balanced within 2–4 mΩ |
| R_s (phase) | 90 mΩ | = R_LL/2 (wye) |

The catalog-KV match (2387 vs 2400) independently confirms the phase-to-Y
probing interpretation. Per-file Ke is constant to ±0.3% across 30–100 RPS
(−2.8% at 20 RPS from low SNR). THD 2.6–3.4% (clean sinusoidal), 6.2% at 20 RPS.

## Teardown — turn count (2026-06-17)

Hand-counted: **12 teeth, one concentrated coil per
tooth (12N14P)**, **N ≈ 20–24 turns/tooth (~22 nominal)**, single strand,
multi-layer wrap. A photo read alone undercounts (the outer visible layer is
only ~7–8 passes; inner layers don't show externally), so a physical
count was needed. This pins the per-turn flux basis (raw model ψ_f ∝ N).

## Back-EMF harmonics

The harmonic spectrum (`B_n/B_1`, channel-mean over the high-SNR files) is used
to bound the magnet arc ratio `α_p`. The 3rd-harmonic content places `α_p` on
two branches around the `magnet_only_ratio(α_p, 3)` zero at `α_p = 2/3`; the
5th/7th harmonics disambiguate. Both inferred values are lower bounds (the
12N14P concentrated winding partially filters higher harmonics). Run the script
for the full table.

## Speed-sweep detail

| Label (rps) | mech rps | f_elec (Hz) | slip % | V_pk LN (V) | Ke (mV/rad·s) | THD % |
|------------:|---------:|------------:|-------:|------------:|--------------:|------:|
| 20  | 20.07 | 140.5 | +0.4 | 0.2831 | 2.244 | 6.16 |
| 30  | 30.17 | 211.2 | +0.6 | 0.4373 | 2.307 | 4.24 |
| 40  | 39.98 | 279.8 | −0.1 | 0.5821 | 2.318 | 2.56 |
| 65  | 62.71 | 439.0 | −3.5 | 0.9108 | 2.311 | 2.96 |
| 70  | 63.54 | 444.8 | −9.2 | 0.9228 | 2.312 | 3.17 |
| 80  | 76.25 | 533.7 | −4.7 | 1.1071 | 2.311 | 2.88 |
| 90  | 89.28 | 625.0 | −0.8 | 1.2961 | 2.310 | 3.37 |
| 100 | 97.02 | 679.1 | −3.0 | 1.4082 | 2.310 | 3.08 |

## Caveats

- Stator stack length (6 mm) and wye connection are assumed from the catalog
  1806 convention, not measured.
- Turn count is a photo + hand count (±2), not an unwound-tooth count.
- Catalog winding resistance (Rm) is not published.
