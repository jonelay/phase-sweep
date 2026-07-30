"""Per-motor parameter calibration.

Bounded least squares over cross-validation residuals: selected Motor
parameters are adjusted (through the perturbation transform, so bounds
rejection and derived-field invalidation apply) to close model-vs-measured
disagreement on caller-selected quantities. Residuals are signed relative
errors weighted by each row's resolved tolerance, so a residual
of 1.0 means "at tolerance".

Identifiability guards refuse circular fits (fitted param derived from the
target dataset), under-determined fits (params >= residual rows), and
zero-sensitivity params; near-collinear sensitivities are reported, never
silently absorbed.

The inner loop is restricted to fast (circuit/analytical-tier) models;
FEM verifies the calibrated motor once at the end (explore-then-validate).

Class-level correction factors are a deliberate non-goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from phasesweep.crossval import DEFAULT_TOLERANCE_PCT, ComparisonRow, compare_results
from phasesweep.measured import BoundRef, CurveRef, KeyMapping
from phasesweep.motor import Motor
from phasesweep.perturbation import PerturbParam, perturb_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult

FITTABLE_PARAMS: tuple[PerturbParam, ...] = (
    "OD", "gap", "L_stk", "B_rem", "k_w", "psi_f", "R_s", "L_d", "L_q",
    "B_core",
)

# psi_f and B_rem are interderivable: fitting one against a
# dataset the other was derived from is the same echo, and fitting both
# is degenerate by construction.
_INTERDERIVABLE: frozenset[str] = frozenset({"psi_f", "B_rem"})

# Fractional-delta bounds; caller overrides per param.
DEFAULT_BOUND: tuple[float, float] = (-0.3, 0.3)

_COLLINEAR_COS = 0.99
_REJECT_RESIDUAL = 1e3


@dataclass(frozen=True)
class FittedParam:
    param: str
    initial: float
    final: float
    delta: float
    stderr: float | None  # indicative (Jacobian-based), param units

    def to_dict(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "initial": self.initial,
            "final": self.final,
            "delta": self.delta,
            "stderr_indicative": self.stderr,
        }


@dataclass(frozen=True)
class CalibrationRecord:
    """Traceability artifact for one calibration."""

    motor_name: str
    source_config_id: str
    calibrated_config_id: str
    params: tuple[FittedParam, ...]
    quantities: tuple[str, ...]
    models: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    residuals_before: tuple[dict[str, Any], ...]
    residuals_after: tuple[dict[str, Any], ...]
    model_versions: dict[str, int]
    optimizer: dict[str, Any]
    warnings: tuple[str, ...]
    closed: bool  # every residual row within tolerance after the fit
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "motor_name": self.motor_name,
            "source_config_id": self.source_config_id,
            "calibrated_config_id": self.calibrated_config_id,
            "params": [p.to_dict() for p in self.params],
            "quantities": list(self.quantities),
            "models": list(self.models),
            "dataset_ids": list(self.dataset_ids),
            "residuals_before": list(self.residuals_before),
            "residuals_after": list(self.residuals_after),
            "model_versions": self.model_versions,
            "optimizer": self.optimizer,
            "warnings": list(self.warnings),
            "closed": self.closed,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        import json
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(frozen=True)
class CalibrationResult:
    motor: Motor
    record: CalibrationRecord


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def _param_value(motor: Motor, param: str) -> float | None:
    """Current value of a fittable param (the quantity the delta scales)."""
    if param == "OD":
        return motor.geometry.r_outer if motor.geometry is not None else None
    if param == "gap":
        if motor.geometry is None:
            return None
        return abs(motor.geometry.r_stator - motor.geometry.r_magnet)
    val = getattr(motor, param)
    return val if isinstance(val, (int, float)) else None


def _apply_deltas(
    motor: Motor, params: tuple[str, ...], deltas: np.ndarray,
) -> Motor | None:
    m: Motor | None = motor
    for param, delta in zip(params, deltas):
        if m is None:
            return None
        m = perturb_motor(m, param, float(delta))  # type: ignore[arg-type]
    return m


def _guard_params(motor: Motor, params: tuple[str, ...]) -> None:
    for p in params:
        if p not in FITTABLE_PARAMS:
            raise ValueError(
                f"unknown fittable param {p!r} — choose from {FITTABLE_PARAMS}"
            )
    if len(set(params)) != len(params):
        raise ValueError(f"duplicate params in {params}")
    if _INTERDERIVABLE <= set(params):
        raise ValueError(
            "psi_f and B_rem are interderivable: fit one, never both"
        )
    for p in params:
        if _param_value(motor, p) is None:
            raise ValueError(
                f"Motor '{motor.name}': param {p!r} is not set — cannot fit it"
            )


def _guard_circular(
    measured: list[RunResult], params: tuple[str, ...],
) -> None:
    """Refuse fitting a param against a dataset it was derived from.

    The echo problem: the fit would 'validate' the derivation
    against itself. psi_f/B_rem count as one equivalence class.
    """
    for r in measured:
        derived = set((r.metrics or {}).get("_derived_params", ()))
        if derived & _INTERDERIVABLE:
            derived |= _INTERDERIVABLE
        clash = sorted(set(params) & derived)
        if clash:
            raise ValueError(
                f"circular fit refused: param(s) {clash} were derived from "
                f"dataset {r.config.dataset_id!r} (derived_params tag) — "
                f"fitting them against it is an echo, not a calibration"
            )


# ---------------------------------------------------------------------------
# Residual evaluation
# ---------------------------------------------------------------------------

def _auto_models(measured: list[RunResult], quantities: tuple[str, ...]) -> tuple[str, ...]:
    """Fast computed models producing the computed-side keys the selected
    quantities resolve to (key_mapping / curve_compare / direct)."""
    needed: set[str] = set()
    for r in measured:
        metrics = r.metrics or {}
        km = metrics.get("_key_mapping", {})
        cc = metrics.get("_curve_compare", {})
        bc = metrics.get("_bound_compare", {})
        for q in quantities:
            if q in km:
                needed.add(KeyMapping.from_dict(km[q]).computed_key)
            elif q in cc:
                ref = CurveRef.from_dict(cc[q])
                needed.update({ref.curve_x, ref.curve_y})
            elif q in bc:
                needed.add(BoundRef.from_dict(bc[q]).computed_key)
            else:
                needed.add(q)
    # thermal_duty needs a duty_profile the loop's RunConfig never carries.
    models = tuple(sorted(
        name for name, info in MODEL_REGISTRY.items()
        if info.source == "computed" and info.cost == "fast"
        and name != "thermal_duty"
        and needed & info.produces
    ))
    if not models:
        raise ValueError(
            f"no fast computed model produces any of {sorted(needed)} — "
            f"the v1 inner loop excludes slow models (FEM verifies at the end)"
        )
    return models


def _run_models(
    motor: Motor, models: tuple[str, ...], n_theta: int,
) -> list[RunResult]:
    results = []
    for name in models:
        info = MODEL_REGISTRY[name]
        assert info.fn is not None
        config = RunConfig(motor=motor, model=name, n_theta=n_theta)
        metrics = info.fn(config)
        results.append(RunResult(
            config=config, model=name, status="OK",
            metrics=metrics, elapsed_s=0.0,
        ))
    return results


def _dataset_labels(measured: list[RunResult]) -> list[str]:
    """One unique label per dataset — the residual-row key needs it so two
    datasets sharing a test_type and quantity cannot overwrite each other."""
    labels = []
    for i, m in enumerate(measured):
        label = m.config.dataset_id or f"dataset-{i}"
        if label in labels:
            label = f"{label}#{i}"
        labels.append(label)
    return labels


def _row_key(dataset: str, row: ComparisonRow) -> tuple[str, str, str, str]:
    return (dataset, row.quantity, row.model_a, row.model_b)


def _fit_rows(
    motor: Motor, measured: list[RunResult],
    quantities: tuple[str, ...], models: tuple[str, ...], n_theta: int,
) -> dict[tuple[str, str, str, str], ComparisonRow]:
    """Equality-style comparison rows for the selected quantities.

    Bound rows carry no equality semantics and skipped rows no comparand,
    so neither can be a least-squares residual.
    """
    computed = _run_models(motor, models, n_theta)
    rows: dict[tuple[str, str, str, str], ComparisonRow] = {}
    for label, m in zip(_dataset_labels(measured), measured):
        for c in computed:
            for row in compare_results(m, c):
                if row.quantity not in quantities:
                    continue
                if row.comparison_type in ("bound", "skipped"):
                    continue
                rows[_row_key(label, row)] = row
    return rows


def _residual(row: ComparisonRow) -> float:
    """Signed relative error in tolerance units (1.0 = at tolerance)."""
    tol = row.tol_pct if row.tol_pct > 0 else DEFAULT_TOLERANCE_PCT
    ref = max(abs(row.val_a), 1e-12)
    return (row.delta / ref * 100.0) / tol


def _row_dict(row: ComparisonRow) -> dict[str, Any]:
    return {
        "quantity": row.quantity,
        "measured_model": row.model_a,
        "computed_model": row.model_b,
        "measured": row.val_a,
        "computed": row.val_b,
        "rel_pct": row.rel_pct,
        "tol_pct": row.tol_pct,
        "passed": row.passed,
    }


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def calibrate(
    motor: Motor,
    measured: list[RunResult],
    params: tuple[str, ...] | list[str],
    quantities: tuple[str, ...] | list[str],
    *,
    models: tuple[str, ...] | list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_theta: int = 360,
) -> CalibrationResult:
    """Fit `params` on `motor` against measured/published `measured` results.

    Raises ValueError on identifiability failures (circular fit,
    under-determination, zero sensitivity, missing residual rows).
    Never mutates or overwrites the source motor — the calibrated Motor
    is returned alongside a CalibrationRecord.
    """
    from scipy.optimize import least_squares

    params = tuple(params)
    quantities = tuple(quantities)
    _guard_params(motor, params)
    _guard_circular(measured, params)
    for name, (lo, hi) in (bounds or {}).items():
        if name not in params:
            raise ValueError(
                f"bound for {name!r} matches no fitted param {params}"
            )
        if not lo < hi:
            raise ValueError(
                f"bound for {name!r}: ({lo}, {hi}) — lower must be < upper"
            )

    model_tuple = tuple(models) if models else _auto_models(measured, quantities)
    for name in model_tuple:
        info = MODEL_REGISTRY.get(name)
        if info is None or info.source != "computed" or info.fn is None:
            raise ValueError(f"{name!r} is not a runnable computed model")
        if info.validate is not None:
            info.validate(motor)

    rows0 = _fit_rows(motor, measured, quantities, model_tuple, n_theta)
    if not rows0:
        raise ValueError(
            f"no equality-style comparison rows for quantities {quantities} "
            f"between models {model_tuple} and the given datasets — nothing to fit"
        )
    if len(params) > len(rows0):
        raise ValueError(
            f"under-determined fit refused: {len(params)} param(s) against "
            f"{len(rows0)} independent residual row(s)"
        )
    exactly_determined = len(params) == len(rows0)

    keys = sorted(rows0)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        m = _apply_deltas(motor, params, x)
        if m is None:
            return np.full(len(keys), _REJECT_RESIDUAL)
        try:
            rows = _fit_rows(m, measured, quantities, model_tuple, n_theta)
        except ValueError:
            return np.full(len(keys), _REJECT_RESIDUAL)
        return np.array([
            _residual(rows[k]) if k in rows else _REJECT_RESIDUAL for k in keys
        ])

    # Zero-sensitivity pre-check: a param with no effect on the selected
    # quantities would silently ride along at its initial value.
    r0 = residual_fn(np.zeros(len(params)))
    step = 1e-3
    jac0 = np.empty((len(keys), len(params)))
    for j in range(len(params)):
        x = np.zeros(len(params))
        x[j] = step
        jac0[:, j] = (residual_fn(x) - r0) / step
    dead = [params[j] for j in range(len(params))
            if np.linalg.norm(jac0[:, j]) < 1e-9]
    if dead:
        raise ValueError(
            f"param(s) {dead} have no sensitivity to quantities "
            f"{quantities} under models {model_tuple} — refusing to fit"
        )

    lb = np.array([(bounds or {}).get(p, DEFAULT_BOUND)[0] for p in params])
    ub = np.array([(bounds or {}).get(p, DEFAULT_BOUND)[1] for p in params])
    x0 = np.clip(np.zeros(len(params)), lb, ub)
    clamped = [p for p, x in zip(params, x0) if x != 0.0]
    res = least_squares(residual_fn, x0, bounds=(lb, ub))

    motor_cal = _apply_deltas(motor, params, res.x)
    if motor_cal is None:  # pragma: no cover — bounds keep deltas feasible
        raise RuntimeError("optimizer returned an infeasible parameter vector")
    motor_cal = dc_replace(motor_cal, name=f"{motor.name} [calibrated]")
    rows_after = _fit_rows(motor_cal, measured, quantities, model_tuple, n_theta)

    warnings_list = _collinearity_warnings(params, res.jac)
    if clamped:
        warnings_list.append(
            f"param(s) {clamped}: bounds exclude a zero delta — initial "
            f"guess clamped to the nearest bound"
        )
    if exactly_determined:
        warnings_list.append(
            f"exactly determined: {len(params)} param(s) against "
            f"{len(keys)} residual row(s) — the fit is an inversion with no "
            f"redundancy; indicative uncertainty unavailable"
        )
    at_bound = [
        p for p, x, lo, hi in zip(params, res.x, lb, ub)
        if np.isclose(x, lo, atol=1e-9) or np.isclose(x, hi, atol=1e-9)
    ]
    if at_bound:
        warnings_list.append(
            f"param(s) {at_bound} finished at a bound — the data wants more "
            f"correction than the allowed range permits"
        )

    stderr = _indicative_stderr(res.jac, res.fun)
    fitted = tuple(
        FittedParam(
            param=p,
            initial=float(_param_value(motor, p) or 0.0),
            final=float(_param_value(motor_cal, p) or 0.0),
            delta=float(x),
            stderr=(
                None if stderr is None
                else float(stderr[j] * (_param_value(motor, p) or 0.0))
            ),
        )
        for j, (p, x) in enumerate(zip(params, res.x))
    )

    record = CalibrationRecord(
        motor_name=motor.name,
        source_config_id=motor.config_id,
        calibrated_config_id=motor_cal.config_id,
        params=fitted,
        quantities=quantities,
        models=model_tuple,
        dataset_ids=tuple(
            r.config.dataset_id or "unknown" for r in measured),
        residuals_before=tuple(
            {"dataset": k[0], **_row_dict(rows0[k])} for k in keys),
        residuals_after=tuple(
            {"dataset": k[0], **_row_dict(rows_after[k])}
            for k in keys if k in rows_after),
        model_versions={
            name: MODEL_REGISTRY[name].version for name in model_tuple},
        optimizer={
            "method": "scipy.optimize.least_squares (trf)",
            "cost_initial": float(0.5 * np.sum(r0**2)),
            "cost_final": float(res.cost),
            "nfev": int(res.nfev),
            "status": int(res.status),
            "message": str(res.message),
            "success": bool(res.success),
            "bounds": {p: [float(lo), float(hi)]
                       for p, lo, hi in zip(params, lb, ub)},
        },
        warnings=tuple(warnings_list),
        closed=all(rows_after[k].passed for k in keys if k in rows_after)
        and len(rows_after) == len(keys),
        timestamp=datetime.now().isoformat(),
    )
    return CalibrationResult(motor=motor_cal, record=record)


def _collinearity_warnings(
    params: tuple[str, ...], jac: np.ndarray,
) -> list[str]:
    out: list[str] = []
    if len(params) < 2:
        return out
    norms = np.linalg.norm(jac, axis=0)
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            if norms[i] < 1e-12 or norms[j] < 1e-12:
                continue
            cos = abs(float(jac[:, i] @ jac[:, j]) / (norms[i] * norms[j]))
            if cos > _COLLINEAR_COS:
                out.append(
                    f"near-collinear sensitivities: {params[i]} vs "
                    f"{params[j]} (|cos| = {cos:.4f}) — the fit cannot "
                    f"separate them; fix one externally"
                )
    return out


def _indicative_stderr(
    jac: np.ndarray, residuals: np.ndarray,
) -> np.ndarray | None:
    """Jacobian-based 1-sigma estimate in fractional-delta units.

    Indicative only: residual weights are tolerances, not measurement
    uncertainties, so this is a conditioning statement, not a confidence
    interval.
    """
    m, n = jac.shape
    if m <= n:
        return None
    dof = m - n
    s_sq = float(residuals @ residuals) / dof
    try:
        cov = np.linalg.pinv(jac.T @ jac) * s_sq
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


# ---------------------------------------------------------------------------
# Calibrated TOML output (new file, never silent overwrite)
# ---------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, str):
        import json
        return json.dumps(v)  # JSON string escapes are valid TOML basic strings
    if isinstance(v, bool):  # pragma: no cover — no bool motor fields today
        return "true" if v else "false"
    return repr(v)


def _toml_section(name: str, fields: dict[str, Any]) -> list[str]:
    present = {k: v for k, v in fields.items() if v is not None}
    if not present:
        return []
    lines = [f"[{name}]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in present.items()]
    lines.append("")
    return lines


def write_motor_toml(
    motor: Motor, path: Path, *, header: str | None = None,
) -> None:
    """Serialize a Motor to the load_motor TOML schema (exact round-trip).

    Refuses to overwrite an existing file: calibrated motors go to NEW
    files by design.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} exists — calibrated motors are written to new files, "
            f"never over an existing one"
        )
    lines: list[str] = []
    if header:
        lines += [f"# {ln}" for ln in header.splitlines()]
        lines.append("")

    geo = motor.geometry
    lines += _toml_section("motor", {
        "name": motor.name,
        "topology": geo.topology if geo is not None else None,
    })
    lines += _toml_section("circuit", {
        "n_p": motor.n_p,
        "R_s": motor.R_s,
        "L_d": motor.L_d,
        "L_q": motor.L_q,
        "psi_f": motor.psi_f,
        "J": motor.J,
        "I_rated": motor.I_rated,
    })
    lines += _toml_section("winding", {
        "N": motor.N,
        "k_w": motor.k_w,
        "coils_series": motor.coils_series,
    })
    if geo is not None:
        lines += _toml_section("geometry", {
            "r_outer": geo.r_outer,
            "r_stator": geo.r_stator,
            "r_magnet": geo.r_magnet,
            "r_rotor": geo.r_rotor,
            "r_inner": geo.r_inner,
            "r_ag": geo.r_ag,
            "n_slots": geo.n_slots,
            "slot_depth": geo.slot_depth,
            "slot_width_ratio": geo.slot_width_ratio,
            "slot_opening_width": geo.slot_opening_width,
            "back_iron_thickness": geo.back_iron_thickness,
            "alpha_p": motor.alpha_p,
            "L_stk": motor.L_stk,
        })
    lines += _toml_section("materials", {
        "B_rem": motor.B_rem,
        "mu_r_pm": motor.mu_r_pm,
        "mu_r_fe": motor.mu_r_fe,
        "alpha_Br": motor.alpha_Br,
        "B_knee": motor.B_knee,
        "alpha_B_knee": motor.alpha_B_knee,
    })
    lines += _toml_section("drive", {
        "U_DC": motor.drive.U_DC,
        "MAX_I_S": motor.drive.MAX_I_S,
        "W_REF": motor.drive.W_REF,
        "I_LIMIT": motor.drive.I_LIMIT,
    })
    lines += _toml_section("thermal", {
        "winding_temp_limit": motor.winding_temp_limit,
        "ambient_temp": motor.ambient_temp,
        "r_th": motor.r_th,
        "insulation_class": motor.insulation_class,
        "thermal_time_constant": motor.thermal_time_constant,
        "magnet_temp": motor.magnet_temp,
    })
    lines += _toml_section("iron", {
        "k_h": motor.k_h,
        "k_e": motor.k_e,
        "alpha_fe": motor.alpha_fe,
        "m_core": motor.m_core,
        "B_core": motor.B_core,
    })
    path.write_text("\n".join(lines))
