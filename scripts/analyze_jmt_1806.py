"""JMT 1806 2400KV back-EMF sweep analysis.

Motor: JMT 1806 2400KV (12N14P, 18 mm stator, 16 g), quadcopter-class outrunner.
Captures 2026-06-08, motor spun passively 1:1 by an external backdriver
(open-circuit DUT, three single-ended scope channels on the phase wires).

Probing: phase-to-Y (single-ended scope on 3-wire wye, virtual-Y reference).
Confirmed via vendor-KV match: derived KV = 60/(2pi*Ke_LL_pk) = 2388 vs vendor 2400.
The +/-120 deg phase separations alone are consistent with either phase-to-Y or
line-to-line probing -- vendor catalog match resolved the ambiguity.
"""
import csv
import glob
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "jmt_1806", "captures")
PP = 7  # pole pairs, established from 20/30/40/90 RPS files


def load(path):
    t, c1, c2, c3 = [], [], [], []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if not row:
                continue
            t.append(float(row[0]))
            c1.append(float(row[1]))
            c2.append(float(row[2]))
            c3.append(float(row[3]))
    return np.asarray(t), np.asarray(c1), np.asarray(c2), np.asarray(c3)


def f_elec_fft(t, y):
    y = y - y.mean()
    n = len(y)
    dt = t[1] - t[0]
    nfft = n * 8
    F = np.abs(np.fft.rfft(y, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    idx = np.argmax(F[1:]) + 1
    a, b, c = np.log(F[idx-1] + 1e-30), np.log(F[idx] + 1e-30), np.log(F[idx+1] + 1e-30)
    delta = 0.5 * (a - c) / (a - 2*b + c)
    return f[idx] + delta * (f[1] - f[0])


def fund_amp(t, y, f0):
    """Single-bin DFT at f0 -> peak amplitude of the fundamental."""
    y = y - y.mean()
    dt = t[1] - t[0]
    n = len(y)
    w = np.hanning(n)
    yw = y * w
    tt = np.arange(n) * dt
    re = (yw * np.cos(2*np.pi*f0*tt)).sum()
    im = (yw * np.sin(2*np.pi*f0*tt)).sum()
    return 2.0 * np.hypot(re, im) / w.sum()


def thd_pct(t, y, f0, n_h=10):
    y = y - y.mean()
    n = len(y)
    dt = t[1] - t[0]
    nfft = n * 8
    F = np.abs(np.fft.rfft(y, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    df = f[1] - f[0]

    def peak_near(target):
        idx = round(target / df)
        w = max(1, round(5.0 / df / 8))
        lo, hi = max(1, idx - w), min(len(F), idx + w + 1)
        return F[lo:hi].max()

    fund = peak_near(f0)
    return 100.0 * np.sqrt(sum(peak_near(f0 * h)**2 for h in range(2, n_h + 1))) / fund


def fund_phase_deg(t, y, f0):
    """Returns fundamental phase in degrees, wrapped to (-180, 180]."""
    y = y - y.mean()
    dt = t[1] - t[0]
    n = len(y)
    w = np.hanning(n)
    yw = y * w
    tt = np.arange(n) * dt
    re = (yw * np.cos(2*np.pi*f0*tt)).sum()
    im = (yw * np.sin(2*np.pi*f0*tt)).sum()
    ang = np.degrees(np.arctan2(im, re))
    return ((ang + 180) % 360) - 180


def wrap(d):
    return ((d + 180) % 360) - 180


print("="*120)
print(f"{'file':<11} {'lbl_rps':>7} {'f_elec':>8} {'mech_rps':>8} {'slip%':>6} {'A1_LN':>7} {'A2_LN':>7} {'A3_LN':>7} {'A_LN':>7} {'THD':>5} {'sep12':>6} {'sep23':>6} {'sep13':>6}")
print(f"{'':11} {'':7} {'Hz':>8} {'(=f/7)':>8} {'vs lbl':>6} {'V_pk':>7} {'V_pk':>7} {'V_pk':>7} {'V_pk':>7} {'%':>5} {'deg':>6} {'deg':>6} {'deg':>6}")
print("-"*120)
results = []
for path in sorted(glob.glob(os.path.join(DATA, "*.CSV"))):
    name = os.path.basename(path)
    lbl_rps = float(name.replace("RPS.CSV", ""))
    t, c1, c2, c3 = load(path)
    f0 = f_elec_fft(t, c1)
    mech_rps = f0 / PP
    slip = (mech_rps - lbl_rps) / lbl_rps * 100
    a1, a2, a3 = fund_amp(t, c1, f0), fund_amp(t, c2, f0), fund_amp(t, c3, f0)
    # Phase-to-Y probing -- channel amplitudes ARE the phase-to-neutral peak.
    a_ln = (a1 + a2 + a3) / 3
    thd = thd_pct(t, c1, f0)
    p1, p2, p3 = fund_phase_deg(t, c1, f0), fund_phase_deg(t, c2, f0), fund_phase_deg(t, c3, f0)
    s12, s23, s13 = wrap(p2 - p1), wrap(p3 - p2), wrap(p3 - p1)
    print(f"{name:<11} {lbl_rps:>7.0f} {f0:>8.2f} {mech_rps:>8.3f} {slip:>+6.1f} {a1:>7.4f} {a2:>7.4f} {a3:>7.4f} {a_ln:>7.4f} {thd:>5.2f} {s12:>+6.1f} {s23:>+6.1f} {s13:>+6.1f}")
    results.append(dict(name=name, lbl=lbl_rps, f0=f0, mech_rps=mech_rps,
                        a1=a1, a2=a2, a3=a3, a_ln=a_ln,
                        thd=thd, s12=s12, s23=s23, s13=s13))

print()
print("="*60)
print("Ke fit (peak phase-to-neutral vs mechanical omega, through origin)")
print("="*60)

mech_rps = np.array([r["mech_rps"] for r in results])
omega_m = 2 * np.pi * mech_rps
a_ln = np.array([r["a_ln"] for r in results])

# Linear fit through origin
slope = (a_ln @ omega_m) / (omega_m @ omega_m)  # V/(rad/s)
fit = slope * omega_m
r2 = 1 - ((a_ln - fit)**2).sum() / ((a_ln - a_ln.mean())**2).sum()
Ke_mV = slope * 1000

# psi_f from Ke_LN_pk = pp * psi_f
psi_f_uWb = (slope / PP) * 1e6

# Vendor KV convention: KV = 60 / (2*pi * Ke_LL_pk),  Ke_LL_pk = sqrt(3) * Ke_LN_pk
slope_LL_pk = slope * np.sqrt(3)  # V_LL_pk / omega_m
Kv_rpm_per_V = 60 / (2 * np.pi) / slope_LL_pk

# Kt in FOC amplitude-invariant Park convention (Kt * I_q_pk = torque)
# Kt = 1.5 * pp * psi_f = 1.5 * Ke_LN_pk (since Ke = pp * psi_f)
Kt_mNm_per_A = 1.5 * slope * 1000  # mN*m / A

print(f"Ke (phase-to-Y, peak basis):         {Ke_mV:.4f} mV/(rad/s)")
print(f"R^2:                                 {r2:.6f}")
print(f"psi_f (assuming pp={PP}):            {psi_f_uWb:.1f} uWb")
print(f"Kt (FOC, 1.5*pp*psi_f):              {Kt_mNm_per_A:.3f} mNm/A")
print(f"KV (vendor convention):              {Kv_rpm_per_V:.1f} rpm/V")
print()

# Residuals - per-file Ke
print("Per-file Ke (should be constant; deviations = nonlinearity or slip jitter):")
print(f"{'file':<11} {'mech_rps':>8} {'A_LN':>7} {'Ke_mV':>7} {'resid_%':>8}")
for r in results:
    ke_i = r["a_ln"] / (2 * np.pi * r["mech_rps"]) * 1000
    resid = (ke_i - Ke_mV) / Ke_mV * 100
    print(f"{r['name']:<11} {r['mech_rps']:>8.3f} {r['a_ln']:>7.4f} {ke_i:>7.4f} {resid:>+7.2f}")


# ============================================================
# Harmonic spectrum — alpha_p inference from BEMF harmonics
# ============================================================

HARMONICS_ANALYZE = [1, 2, 3, 4, 5, 7, 9, 11, 13]


def harmonic_amps(t, y, f0, harmonics=HARMONICS_ANALYZE):
    """Single-bin DFT peak amplitude at each requested harmonic of f0."""
    y = y - y.mean()
    dt = t[1] - t[0]
    n = len(y)
    w = np.hanning(n)
    yw = y * w
    tt = np.arange(n) * dt
    out = []
    for h in harmonics:
        re = (yw * np.cos(2*np.pi*h*f0*tt)).sum()
        im = (yw * np.sin(2*np.pi*h*f0*tt)).sum()
        out.append(2.0 * np.hypot(re, im) / w.sum())
    return np.array(out)


def magnet_only_ratio(alpha_p, n):
    """Surface-charge magnet-only |B_n/B_1| for arc magnets, no winding filter.
    M_n = (4 B_rem / (n*pi)) * sin(n*pi*alpha_p/2); even n cancel by symmetry.
    """
    if n == 1:
        return 1.0
    if n % 2 == 0:
        return 0.0
    num = abs(np.sin(n * np.pi * alpha_p / 2))
    den = abs(np.sin(np.pi * alpha_p / 2))
    return num / (n * den)


# Collect per-file harmonic data (exclude 20 RPS — low fundamental SNR)
chan_ratio_files = {h: [] for h in HARMONICS_ANALYZE}
chan_per_files = {h: {'c1': [], 'c2': [], 'c3': []} for h in HARMONICS_ANALYZE}
sig_over_chan_files = {h: [] for h in HARMONICS_ANALYZE}
files_used = []
for path in sorted(glob.glob(os.path.join(DATA, "*.CSV"))):
    name = os.path.basename(path)
    lbl_rps = float(name.replace("RPS.CSV", ""))
    if lbl_rps == 20:
        continue
    t, c1, c2, c3 = load(path)
    f0 = f_elec_fft(t, c1)
    h1 = harmonic_amps(t, c1, f0)
    h2 = harmonic_amps(t, c2, f0)
    h3 = harmonic_amps(t, c3, f0)
    hs = harmonic_amps(t, c1 + c2 + c3, f0)
    h_chan = (h1 + h2 + h3) / 3
    for i, h in enumerate(HARMONICS_ANALYZE):
        chan_ratio_files[h].append(h_chan[i] / h_chan[0])
        chan_per_files[h]['c1'].append(h1[i] / h1[0])
        chan_per_files[h]['c2'].append(h2[i] / h2[0])
        chan_per_files[h]['c3'].append(h3[i] / h3[0])
        sig_over_chan_files[h].append(hs[i] / h_chan[i] if h_chan[i] > 0 else 0)
    files_used.append(name)

print()
print("=" * 100)
print("HARMONIC SPECTRUM — BEMF n-th / 1st (channel-mean across high-SNR files)")
print(f"Files ({len(files_used)}): {', '.join(files_used)}    [20 RPS excluded — low SNR]")
print("=" * 100)
print(f"  {'n':>3}  {'measured (%)':>14}   {'α_p=0.68':>9} {'α_p=0.75':>9} {'α_p=0.85':>9} {'α_p=0.95':>9}    notes")
print(f"  {'':>3}  {'mean ± std':>14}   {'(raw magnet, no winding filter)':>40}")
print("  " + "─" * 100)
for h in HARMONICS_ANALYZE:
    arr = np.array(chan_ratio_files[h]) * 100
    m, s = arr.mean(), arr.std()
    if h == 1:
        meas = "100.00 (ref)"
        preds = "{:>9} {:>9} {:>9} {:>9}".format("100", "100", "100", "100")
        note = "fundamental"
    elif h % 2 == 0:
        meas = f"{m:.2f} ± {s:.2f}"
        preds = "{:>9} {:>9} {:>9} {:>9}".format("0", "0", "0", "0")
        note = "(zero by symmetry — measured > 0 indicates rotor/magnet asymmetry)"
    else:
        meas = f"{m:.2f} ± {s:.2f}"
        p1 = magnet_only_ratio(0.68, h) * 100
        p2 = magnet_only_ratio(0.75, h) * 100
        p3 = magnet_only_ratio(0.85, h) * 100
        p4 = magnet_only_ratio(0.95, h) * 100
        preds = f"{p1:>9.2f} {p2:>9.2f} {p3:>9.2f} {p4:>9.2f}"
        note = "TRIPLEN — also see Σ test" if h % 3 == 0 else ""
    print(f"  {h:>3}  {meas:>14}   {preds}    {note}")

print()
print("=" * 100)
print("Σ-CHANNEL TEST — |Σ_n| / |chan_n|  (probing-convention + triplen-content diagnostic)")
print("  Phase-to-Y + non-zero triplen: ratio ≈ 3 at n=3,9    (in-phase across channels)")
print("  Line-to-line probing:          ratio ≈ 0 at all n    (line-line voltages sum to 0)")
print("  Non-triplen at balanced 3-ph:  ratio ≈ 0             (cancel by 120° separation)")
print("=" * 100)
print(f"  {'n':>3}  {'|Σ_n|/|chan_n|':>15}   {'triplen?':>9}    interpretation")
print("  " + "─" * 100)
for h in HARMONICS_ANALYZE:
    arr = np.array(sig_over_chan_files[h])
    m = arr.mean()
    is_trip = (h % 3 == 0)
    label = "Y" if is_trip else "N"
    if h == 1:
        interp = "(fundamental — Σ ≈ 0 confirms balanced wiring)"
    elif is_trip:
        interp = "→ phase-to-Y + real triplen content" if m > 1.5 else "→ no/low triplen content (probing AMBIGUOUS — see note)"
    else:
        interp = "(balanced cancel as expected)"
    print(f"  {h:>3}  {m:>13.2f}×    {label:>9}    {interp}")

print()
print("=" * 100)
print("PER-CHANNEL ASYMMETRY — max/min of B_n/B_1 across channels (n=3,5,7)")
print("  Symmetric magnets & winding: ratio = 1.00")
print("=" * 100)
for h in [3, 5, 7]:
    c1m = np.array(chan_per_files[h]['c1']).mean() * 100
    c2m = np.array(chan_per_files[h]['c2']).mean() * 100
    c3m = np.array(chan_per_files[h]['c3']).mean() * 100
    vals = [c1m, c2m, c3m]
    print(f"  n={h}:  c1={c1m:>5.2f}%   c2={c2m:>5.2f}%   c3={c3m:>5.2f}%   max/min = {max(vals)/min(vals):.2f}")

# alpha_p inference from B_3/B_1, treating two physically plausible branches.
# magnet_only_ratio(alpha_p, 3) has a zero at alpha_p = 2/3.
# Plausible coverage range for this winding class: alpha_p in [0.69, 0.81].
b3_meas = np.mean(chan_ratio_files[3])
alphas = np.linspace(0.50, 1.00, 2001)
predictions_raw = np.array([magnet_only_ratio(a, 3) for a in alphas])
# Below-zero branch (alpha_p < 2/3) and above-zero branch
mask_lo = alphas < 2.0 / 3.0
mask_hi = alphas > 2.0 / 3.0
diff_lo = np.abs(predictions_raw[mask_lo] - b3_meas)
diff_hi = np.abs(predictions_raw[mask_hi] - b3_meas)
alpha_lo = alphas[mask_lo][np.argmin(diff_lo)]
alpha_hi = alphas[mask_hi][np.argmin(diff_hi)]

print()
print("=" * 100)
print("α_p INFERENCE — from measured B_3/B_1, magnet-only model")
print("=" * 100)
print(f"  Measured B_3/B_1 (channel mean):    {b3_meas*100:.2f}%")
print(f"  Nominal α_p = 0.75:                 raw magnet B_3/B_1 = {magnet_only_ratio(0.75, 3)*100:.2f}%")
print(f"  Conservative α_p = 0.68:            raw magnet B_3/B_1 = {magnet_only_ratio(0.68, 3)*100:.2f}%")
print()
print("  magnet_only_ratio(α_p, 3) is non-monotonic (zero at α_p = 2/3 ≈ 0.667).")
print("  Two α_p values give the measured B_3/B_1 in the physical range [0.5, 1.0]:")
print(f"    Low branch:   α_p ≈ {alpha_lo:.3f}    (just below the zero)")
print(f"    High branch:  α_p ≈ {alpha_hi:.3f}    (rising toward full coverage)")
print()
print("  The 5th and 7th harmonics disambiguate (zeros at α_p = 0.80 and 6/7=0.857):")
print(f"    Predicted B_5/B_1 at α_p={alpha_lo:.3f}:  {magnet_only_ratio(alpha_lo, 5)*100:.2f}%")
print(f"    Predicted B_5/B_1 at α_p={alpha_hi:.3f}:  {magnet_only_ratio(alpha_hi, 5)*100:.2f}%")
print(f"    Measured B_5/B_1:                          {np.mean(chan_ratio_files[5])*100:.2f}%")
print()
print(f"    Predicted B_7/B_1 at α_p={alpha_lo:.3f}:  {magnet_only_ratio(alpha_lo, 7)*100:.2f}%")
print(f"    Predicted B_7/B_1 at α_p={alpha_hi:.3f}:  {magnet_only_ratio(alpha_hi, 7)*100:.2f}%")
print(f"    Measured B_7/B_1:                          {np.mean(chan_ratio_files[7])*100:.2f}%")
print()
print("  Caveat: 12N14P concentrated winding partially filters higher harmonics.")
print("  Both inferred α_p values are LOWER BOUNDS — the true raw-magnet α_p is somewhat")
print("  higher when the winding-factor attenuation k_w(n)/k_w(1) is restored.")
