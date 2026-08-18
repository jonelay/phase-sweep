"""Decompose line-to-line impedance measurements into per-phase R_s and L_d.

14 mm outrunner: 12p9s, 3-phase wye, 4 wires (Red, Green, Blue + Brown neutral).
Measurements: LCR meter at 1 kHz, 6 wire pairs.
"""

import numpy as np

# Raw measurements (LCR 1 kHz)
# fmt: off
data = {
    ("Red",   "Brown"): {"R_mOhm": 223.2, "L_uH":  8.60},
    ("Red",   "Green"): {"R_mOhm": 282.6, "L_uH": 24.26},
    ("Green", "Brown"): {"R_mOhm": 218.4, "L_uH":  9.00},
    ("Blue",  "Brown"): {"R_mOhm": 336.0, "L_uH":  9.50},
    ("Blue",  "Red"):   {"R_mOhm": 276.2, "L_uH": 24.95},
    ("Blue",  "Green"): {"R_mOhm": 273.9, "L_uH": 25.91},
}
# fmt: on

# --- Identify topology from L clustering ---
pairs = list(data.keys())
L_vals = np.array([data[p]["L_uH"] for p in pairs])
threshold = (L_vals.min() + L_vals.max()) / 2

lo_pairs = [p for p in pairs if data[p]["L_uH"] < threshold]
hi_pairs = [p for p in pairs if data[p]["L_uH"] >= threshold]

# Neutral wire appears in all low-L (line-to-neutral) pairs
lo_wires = set()
for a, b in lo_pairs:
    lo_wires.update([a, b])
hi_wires = set()
for a, b in hi_pairs:
    hi_wires.update([a, b])
neutral = lo_wires - hi_wires
phase_wires = hi_wires

print("=== Topology Detection ===")
print(f"Low-L pairs (line-neutral):  {lo_pairs}")
print(f"High-L pairs (line-line):    {hi_pairs}")
print(f"Neutral wire: {neutral}")
print(f"Phase wires:  {phase_wires}")
print()

neutral_wire = neutral.pop()
phase_list = sorted(phase_wires)

# --- Per-phase resistance from line-to-line (3×3 system) ---
# R_XY = R_X + R_Y (series through internal neutral, no neutral lead involved)
print("=== Resistance Decomposition ===")
print()
print("Line-to-line: R_XY = R_phase_X + R_phase_Y")

ll_R = {}
for a, b in hi_pairs:
    ll_R[(a, b)] = data[(a, b)]["R_mOhm"]

# Solve: given R_AB, R_AC, R_BC → R_A, R_B, R_C
A, B, C = phase_list
R_AB = ll_R.get((A, B)) or ll_R.get((B, A))
R_AC = ll_R.get((A, C)) or ll_R.get((C, A))
R_BC = ll_R.get((B, C)) or ll_R.get((C, B))

# Find the right pairs (order may differ)
def get_ll_R(x, y):
    for (a, b), v in ll_R.items():
        if {a, b} == {x, y}:
            return v
    raise KeyError(f"No L-L data for {x}-{y}")

R_AB = get_ll_R(A, B)
R_AC = get_ll_R(A, C)
R_BC = get_ll_R(B, C)

R_sum = (R_AB + R_AC + R_BC) / 2
R_phase = {
    A: R_sum - R_BC,
    B: R_sum - R_AC,
    C: R_sum - R_AB,
}

for w in phase_list:
    print(f"  R_{w:5s} = {R_phase[w]:.2f} mOhm")
R_avg = np.mean(list(R_phase.values()))
R_spread = max(R_phase.values()) - min(R_phase.values())
print(f"  Average:  {R_avg:.2f} mOhm  (spread {R_spread:.1f} mOhm, {R_spread/R_avg*100:.1f}%)")
print(f"  R_s = {R_avg/1000:.4f} Ohm")
print()

# Verify L-L consistency
print("Verification (L-L back-check):")
for (a, b), v in ll_R.items():
    calc = R_phase[a] + R_phase[b]
    print(f"  {a}-{b}: measured {v:.1f}, calc {calc:.1f} mOhm")
print()

# --- Neutral lead resistance ---
print("=== Neutral Lead Resistance ===")
print("Line-to-neutral: R_XN = R_phase_X + R_neutral_lead")
R_neutral = {}
for a, b in lo_pairs:
    phase_wire = a if b == neutral_wire else b
    R_meas = data[(a, b)]["R_mOhm"]
    R_nl = R_meas - R_phase[phase_wire]
    R_neutral[phase_wire] = R_nl
    print(f"  {phase_wire}-{neutral_wire}: R_neutral_lead = {R_meas:.1f} - {R_phase[phase_wire]:.1f} = {R_nl:.1f} mOhm")

R_nl_vals = list(R_neutral.values())
print(f"  Range: {min(R_nl_vals):.1f} – {max(R_nl_vals):.1f} mOhm")
print()

# Flag anomalies
if max(R_nl_vals) - min(R_nl_vals) > 20:
    print("  *** ANOMALY: neutral lead resistance inconsistent across phases.")
    outlier = max(R_neutral, key=R_neutral.get)
    others = [v for k, v in R_neutral.items() if k != outlier]
    print(f"  *** {outlier}-{neutral_wire} gives R_neutral = {R_neutral[outlier]:.1f} mOhm")
    print(f"  *** Others give {others[0]:.1f}, {others[1]:.1f} mOhm")
    print(f"  *** Likely cause: extra lead length on {outlier} wire, or probe contact issue")
    print()

# --- Inductance decomposition ---
print("=== Inductance Decomposition ===")
print()

# Self inductance from line-to-neutral
L_self = {}
for a, b in lo_pairs:
    phase_wire = a if b == neutral_wire else b
    L_self[phase_wire] = data[(a, b)]["L_uH"]

print("Self inductance (from line-neutral):")
for w in phase_list:
    print(f"  L_{w:5s} = {L_self[w]:.2f} uH")
L_self_avg = np.mean(list(L_self.values()))
print(f"  Average:  {L_self_avg:.3f} uH")
print()

# Mutual inductance from L-L
# L_XY = L_X + L_Y - 2*M_XY  →  M_XY = (L_X + L_Y - L_XY) / 2
print("Mutual inductance (from line-line):")
print("  L_XY = L_self_X + L_self_Y - 2*M_XY")

def get_ll_L(x, y):
    for p in hi_pairs:
        if set(p) == {x, y}:
            return data[p]["L_uH"]
    raise KeyError(f"No L-L data for {x}-{y}")

M = {}
for i, x in enumerate(phase_list):
    for y in phase_list[i+1:]:
        L_xy = get_ll_L(x, y)
        m = (L_self[x] + L_self[y] - L_xy) / 2
        M[(x, y)] = m
        print(f"  M_{x}-{y} = ({L_self[x]:.2f} + {L_self[y]:.2f} - {L_xy:.2f}) / 2 = {m:.3f} uH")

M_avg = np.mean(list(M.values()))
print(f"  Average mutual: {M_avg:.3f} uH")
print()

# dq-frame inductance: L_d = L_self - M (for SPMSM, L_d = L_q)
L_d = L_self_avg - M_avg
print("=== dq-Frame Inductance (SPMSM: L_d = L_q) ===")
print(f"  L_d = L_self - M = {L_self_avg:.3f} - ({M_avg:.3f}) = {L_d:.3f} uH")
print(f"  L_d = {L_d:.2f} uH = {L_d*1e-6:.3e} H")
print()

# Verify L-L back-check
print("Verification (L-L back-check):")
for (x, y), m in M.items():
    L_xy_meas = get_ll_L(x, y)
    L_xy_calc = L_self[x] + L_self[y] - 2 * m
    print(f"  {x}-{y}: measured {L_xy_meas:.2f}, calc {L_xy_calc:.2f} uH")
print()

# --- Summary ---
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  R_s   = {R_avg/1000:.4f} Ohm  ({R_avg:.1f} mOhm)")
print(f"  L_d   = {L_d:.2f} uH  ({L_d*1e-6:.3e} H)")
print(f"  L_q   = {L_d:.2f} uH  (SPMSM, L_d = L_q)")
print(f"  L_self = {L_self_avg:.2f} uH  (average)")
print(f"  M     = {M_avg:.3f} uH  (average mutual)")
print()
print("Existing TOML values:  R_s = 0.139 Ohm,  L_d = 12.5 uH")
print(f"Delta:  R_s {(R_avg/1000 - 0.139)/0.139*100:+.1f}%,  L_d {(L_d - 12.5)/12.5*100:+.1f}%")
