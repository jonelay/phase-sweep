"""Cross-validation framework for comparing model and measured results.

Computes deltas on shared quantities between any two results for the
same motor. Tolerance thresholds come in three tiers — model-to-model,
model-to-published, model-to-measured — with per-dataset overrides
taking priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from phasesweep.measured import BoundRef, CurveRef, KeyMapping
from phasesweep.sweep_types import RunResult

# ---------------------------------------------------------------------------
# Tolerance tables
# ---------------------------------------------------------------------------

TOLERANCES_MODEL_TO_MODEL: dict[str, float] = {
    "fundamental": 1.0,
    "thd_pct": 10.0,
    "backemf_fundamental": 1.0,
    "tau_mtpa": 5.0,
    "k_T": 5.0,
    "k_T_rms": 5.0,
}

TOLERANCES_MODEL_TO_MEASURED: dict[str, float] = {
    "fundamental": 5.0,
    "thd_pct": 20.0,
    "backemf_fundamental": 5.0,
    "tau_mtpa": 10.0,
    "k_T": 10.0,
    "k_T_rms": 10.0,
    "L_d": 15.0,
    "L_q": 15.0,
    "R_s": 15.0,
}

TOLERANCES_MODEL_TO_PUBLISHED: dict[str, float] = {
    "fundamental": 3.0,
    "backemf_fundamental": 5.0,
    "tau_mtpa": 7.0,
    "k_T": 7.0,
    "k_T_rms": 7.0,
    "thd_pct": 15.0,
}

DEFAULT_TOLERANCE_PCT = 10.0


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonRow:
    quantity: str
    model_a: str
    val_a: float
    model_b: str
    val_b: float
    delta: float
    rel_pct: float
    tol_pct: float
    passed: bool
    comparison_type: Literal["delta", "bound", "curve"] = "delta"
    extrapolated: bool = False


def _pick_tolerance(
    quantity: str,
    source_a: str,
    source_b: str,
    overrides: dict[str, float] | None = None,
    tol_a: dict[str, float] | None = None,
    tol_b: dict[str, float] | None = None,
) -> float:
    # Tier 1: per-dataset tolerances (widest wins if both present)
    vals = []
    if tol_a and quantity in tol_a:
        vals.append(tol_a[quantity])
    if tol_b and quantity in tol_b:
        vals.append(tol_b[quantity])
    if vals:
        return max(vals)
    # Tier 2: caller-supplied overrides
    if overrides and quantity in overrides:
        return overrides[quantity]
    # Tier 3: source-pair table
    sources = {source_a, source_b}
    if sources == {"computed"}:
        table = TOLERANCES_MODEL_TO_MODEL
    elif "published" in sources:
        table = TOLERANCES_MODEL_TO_PUBLISHED
    else:
        table = TOLERANCES_MODEL_TO_MEASURED
    return table.get(quantity, DEFAULT_TOLERANCE_PCT)


def _scalar_quantities(metrics: dict[str, Any] | None) -> dict[str, float]:
    # Keys prefixed with _ are comparison metadata, not physical quantities.
    if not metrics:
        return {}
    return {k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not k.startswith("_")}


# ---------------------------------------------------------------------------
# Extended comparison resolvers
# ---------------------------------------------------------------------------

def _get_metadata(metrics: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not metrics:
        return {}
    return metrics.get(key, {})


def _resolve_bound_compare(
    measured: RunResult, computed: RunResult,
) -> list[ComparisonRow]:
    meta = _get_metadata(measured.metrics, "_bound_compare")
    if not meta or not computed.metrics:
        return []
    rows: list[ComparisonRow] = []
    for meas_key, ref_dict in meta.items():
        ref = BoundRef.from_dict(ref_dict)
        meas_val = measured.metrics.get(meas_key)
        comp_val = computed.metrics.get(ref.computed_key)
        if meas_val is None or comp_val is None:
            continue
        if ref.relation == "gte":
            passed = comp_val >= meas_val
        elif ref.relation == "lte":
            passed = comp_val <= meas_val
        elif ref.relation == "gt":
            passed = comp_val > meas_val
        else:  # lt
            passed = comp_val < meas_val
        # margin: positive = passing direction
        if ref.relation in ("gte", "gt"):
            margin = (comp_val - meas_val) / abs(meas_val) * 100 if abs(meas_val) > 1e-12 else 0.0
        else:
            margin = (meas_val - comp_val) / abs(meas_val) * 100 if abs(meas_val) > 1e-12 else 0.0
        rows.append(ComparisonRow(
            quantity=meas_key,
            model_a=measured.model,
            val_a=meas_val,
            model_b=computed.model,
            val_b=comp_val,
            delta=comp_val - meas_val,
            rel_pct=margin,
            tol_pct=0.0,
            passed=passed,
            comparison_type="bound",
        ))
    return rows


def _resolve_curve_compare(
    measured: RunResult, computed: RunResult,
    tolerances: dict[str, float] | None = None,
) -> list[ComparisonRow]:
    meta = _get_metadata(measured.metrics, "_curve_compare")
    if not meta or not computed.metrics:
        return []
    rows: list[ComparisonRow] = []
    for meas_key, ref_dict in meta.items():
        ref = CurveRef.from_dict(ref_dict)
        meas_val = measured.metrics.get(meas_key)
        curve_x = computed.metrics.get(ref.curve_x)
        curve_y = computed.metrics.get(ref.curve_y)
        if meas_val is None or curve_x is None or curve_y is None:
            continue
        x_arr = np.asarray(curve_x, dtype=float)
        y_arr = np.asarray(curve_y, dtype=float)
        extrapolated = False
        if ref.extract == "interp":
            if ref.at_x is None:
                continue
            extrapolated = bool(ref.at_x < x_arr.min() or ref.at_x > x_arr.max())
            comp_val = float(np.interp(ref.at_x, x_arr, y_arr))
        elif ref.extract == "max":
            comp_val = float(y_arr.max())
        elif ref.extract == "min":
            comp_val = float(y_arr.min())
        else:  # rms
            comp_val = float(np.sqrt(np.mean(y_arr**2)))
        delta = comp_val - meas_val
        ref_val = abs(meas_val) if abs(meas_val) > 1e-12 else abs(comp_val)
        rel_pct = abs(delta) / ref_val * 100 if ref_val > 1e-12 else 0.0
        tol = _pick_tolerance(meas_key, measured.source, computed.source, tolerances,
                              tol_a=measured.tolerances, tol_b=computed.tolerances)
        rows.append(ComparisonRow(
            quantity=meas_key,
            model_a=measured.model,
            val_a=meas_val,
            model_b=computed.model,
            val_b=comp_val,
            delta=delta,
            rel_pct=rel_pct,
            tol_pct=tol,
            passed=rel_pct <= tol,
            comparison_type="curve",
            extrapolated=extrapolated,
        ))
    return rows


def _resolve_key_mapping(
    measured: RunResult, computed: RunResult,
    tolerances: dict[str, float] | None = None,
) -> list[ComparisonRow]:
    meta = _get_metadata(measured.metrics, "_key_mapping")
    if not meta or not computed.metrics:
        return []
    rows: list[ComparisonRow] = []
    for meas_key, ref_dict in meta.items():
        ref = KeyMapping.from_dict(ref_dict)
        meas_val = measured.metrics.get(meas_key)
        comp_val = computed.metrics.get(ref.computed_key)
        if meas_val is None or comp_val is None:
            continue
        if not isinstance(meas_val, (int, float)) or not isinstance(comp_val, (int, float)):
            continue
        delta = comp_val - meas_val
        ref_val = abs(meas_val) if abs(meas_val) > 1e-12 else abs(comp_val)
        rel_pct = abs(delta) / ref_val * 100 if ref_val > 1e-12 else 0.0
        tol = _pick_tolerance(meas_key, measured.source, computed.source, tolerances,
                              tol_a=measured.tolerances, tol_b=computed.tolerances)
        rows.append(ComparisonRow(
            quantity=meas_key,
            model_a=measured.model,
            val_a=meas_val,
            model_b=computed.model,
            val_b=comp_val,
            delta=delta,
            rel_pct=rel_pct,
            tol_pct=tol,
            passed=rel_pct <= tol,
        ))
    return rows


# ---------------------------------------------------------------------------
# DiagnosisSummary
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisSummary:
    delta_rows: list[ComparisonRow]
    bound_rows: list[ComparisonRow]
    curve_rows: list[ComparisonRow]

    @property
    def delta_pass(self) -> bool:
        return all(r.passed for r in self.delta_rows)

    @property
    def bounds_pass(self) -> bool:
        return all(r.passed for r in self.bound_rows)

    @property
    def curves_pass(self) -> bool:
        return all(r.passed for r in self.curve_rows)

    @property
    def all_pass(self) -> bool:
        return self.delta_pass and self.bounds_pass and self.curves_pass

    @property
    def total(self) -> int:
        return len(self.delta_rows) + len(self.bound_rows) + len(self.curve_rows)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_results(
    a: RunResult,
    b: RunResult,
    tolerances: dict[str, float] | None = None,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    handled: set[str] = set()

    # Determine which result carries comparison metadata (measured/published side)
    # and which is the computed counterpart. Metadata can be on either side.
    for measured, computed in [(a, b), (b, a)]:
        if not measured.metrics:
            continue
        # Precedence: bound > curve > key_mapping > direct
        for row in _resolve_bound_compare(measured, computed):
            if row.quantity not in handled:
                rows.append(row)
                handled.add(row.quantity)
        for row in _resolve_curve_compare(measured, computed, tolerances):
            if row.quantity not in handled:
                rows.append(row)
                handled.add(row.quantity)
        for row in _resolve_key_mapping(measured, computed, tolerances):
            if row.quantity not in handled:
                rows.append(row)
                handled.add(row.quantity)

    # Direct delta comparison on remaining shared scalar keys
    scalars_a = _scalar_quantities(a.metrics)
    scalars_b = _scalar_quantities(b.metrics)
    shared = sorted(set(scalars_a) & set(scalars_b) - handled)

    for q in shared:
        va, vb = scalars_a[q], scalars_b[q]
        delta = vb - va
        ref = abs(va) if abs(va) > 1e-12 else abs(vb)
        rel_pct = abs(delta) / ref * 100.0 if ref > 1e-12 else 0.0
        tol = _pick_tolerance(q, a.source, b.source, tolerances,
                              tol_a=a.tolerances, tol_b=b.tolerances)
        rows.append(ComparisonRow(
            quantity=q,
            model_a=a.model,
            val_a=va,
            model_b=b.model,
            val_b=vb,
            delta=delta,
            rel_pct=rel_pct,
            tol_pct=tol,
            passed=rel_pct <= tol,
        ))
    return rows


def compare_all(
    results: list[RunResult],
    tolerances: dict[str, float] | None = None,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            rows.extend(compare_results(a, b, tolerances))
    return rows


# ---------------------------------------------------------------------------
# Diagnostic pattern
# ---------------------------------------------------------------------------

def diagnose(
    results: list[RunResult],
    tolerances: dict[str, float] | None = None,
) -> str:
    computed = [r for r in results if r.source == "computed"]
    measured = [r for r in results if r.source in ("measured", "published")]

    if len(computed) < 2 and not measured:
        return "insufficient data for diagnosis"

    models_agree = True
    if len(computed) >= 2:
        rows = compare_all(computed, tolerances)
        if rows and not all(r.passed for r in rows):
            models_agree = False

    if not measured:
        return "models agree" if models_agree else "models disagree (no measured data)"

    model_vs_meas: dict[str, bool] = {}
    for c in computed:
        rows = []
        for m in measured:
            rows.extend(compare_results(c, m, tolerances))
        if rows:
            model_vs_meas[c.model] = all(r.passed for r in rows)

    any_match = any(model_vs_meas.values()) if model_vs_meas else False
    all_match = all(model_vs_meas.values()) if model_vs_meas else False

    if models_agree and all_match:
        return "validated"
    if models_agree and not any_match:
        return "models agree but disagree with measured — check motor config or measurement"
    if not models_agree and model_vs_meas.get("fem", False):
        return "FEM matches measured — analytical assumptions may be violated"
    if not models_agree and model_vs_meas.get("analytical", False):
        return "analytical matches measured — check FEM mesh or material settings"
    if not models_agree and not any_match:
        return "models disagree and none match measured — multiple problems likely"
    return "mixed agreement — inspect per-quantity results"


def diagnose_detailed(
    results: list[RunResult],
    tolerances: dict[str, float] | None = None,
) -> DiagnosisSummary:
    all_rows = compare_all(results, tolerances)
    return DiagnosisSummary(
        delta_rows=[r for r in all_rows if r.comparison_type == "delta"],
        bound_rows=[r for r in all_rows if r.comparison_type == "bound"],
        curve_rows=[r for r in all_rows if r.comparison_type == "curve"],
    )


def format_diagnosis(summary: DiagnosisSummary) -> str:
    if summary.total == 0:
        return "no comparable quantities"
    parts = []
    if summary.delta_rows:
        parts.append(f"{len(summary.delta_rows)} delta")
    if summary.bound_rows:
        parts.append(f"{len(summary.bound_rows)} bound")
    if summary.curve_rows:
        parts.append(f"{len(summary.curve_rows)} curve")
    counts = ", ".join(parts)

    if summary.all_pass:
        msg = f"validated ({counts} comparisons)"
    elif not summary.bounds_pass:
        failed = [r for r in summary.bound_rows if not r.passed]
        details = "; ".join(
            f"computed {r.model_b} {r.quantity}={r.val_b:.4g} vs {r.val_a:.4g}"
            for r in failed
        )
        msg = f"BOUND FAILURE: {details}"
    else:
        parts_fail = []
        if not summary.delta_pass:
            n = sum(1 for r in summary.delta_rows if not r.passed)
            parts_fail.append(f"{n}/{len(summary.delta_rows)} delta")
        if not summary.curves_pass:
            n = sum(1 for r in summary.curve_rows if not r.passed)
            parts_fail.append(f"{n}/{len(summary.curve_rows)} curve")
        qualifier = "bounds satisfied, " if summary.bound_rows else ""
        msg = f"partial — {qualifier}{' + '.join(parts_fail)} comparisons fail"

    if any(r.extrapolated for r in summary.curve_rows):
        msg += " [WARNING: extrapolated curve comparison]"
    return msg


# ---------------------------------------------------------------------------
# Text table formatting
# ---------------------------------------------------------------------------

def format_table(rows: list[ComparisonRow]) -> str:
    if not rows:
        return "(no shared quantities to compare)"

    lines = [
        f"{'Quantity':<25} {'Model A':<16} {'Val A':>12} "
        f"{'Model B':<16} {'Val B':>12} {'Δ%':>8} {'Tol%':>6} {'':>4}",
        "-" * 105,
    ]
    for r in rows:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.quantity:<25} {r.model_a:<16} {r.val_a:>12.4g} "
            f"{r.model_b:<16} {r.val_b:>12.4g} {r.rel_pct:>7.1f}% "
            f"{r.tol_pct:>5.0f}% {tag:>4}"
        )
    return "\n".join(lines)
