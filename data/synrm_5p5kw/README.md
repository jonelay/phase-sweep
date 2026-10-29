# Jilin University 5.5 kW SynRM

Pure synchronous reluctance motor — no permanent magnets (psi_f = 0).
Torque arises entirely from saliency (L_d > L_q). Exercises the
reluctance-only MTPA path in the circuit-model chain.

## Source

I-f control paper, PMC11679518, 2024, Jilin University. Table 1
(R_s, J, n_p, rated voltage/current/power), Table 2 (L_d/L_q vs
current, saturation-dependent).

## Files

| File | Content |
|---|---|
| `synrm_5p5kw.toml` | Motor definition with saturated-point L_d/L_q at rated current |
