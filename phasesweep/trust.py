"""Headline metrics and trust caveats for one result.

Caveats are derived at read time from provenance keys the models already
emit (``envelope_source``, ``op_source``, ``machine_model``, ...) plus the
model's registered ``limits``, so stored records gain them without a
recompute. No hand-maintained physics table: a rule fires on what the
result says about itself. When a model's outputs change, update its
``headline``/``units`` and the rules here (physics-change skill).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from phasesweep.registry import MODEL_REGISTRY, headline_key
from phasesweep.solver_params import B_IRON_SATURATION_T

__all__ = [
    "RULES",
    "Caveat",
    "Rule",
    "assess",
    "assess_record",
    "headline",
    "headline_record",
    "measured_available",
]

Severity = Literal["info", "warn"]


@dataclass(frozen=True)
class Caveat:
    """One reason to read a result with care. ``basis`` names the
    triggering key=value, or ``limits`` for a model-inherent omission."""

    code: str
    severity: Severity
    text: str
    basis: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    """Fires when ``metrics[key]`` (or ``config[key]`` when
    ``on_config``) satisfies ``when``. ``text`` may format ``{value}``
    and any metric key."""

    code: str
    models: frozenset[str]
    key: str
    when: Callable[[Any], bool]
    severity: Severity
    text: str
    on_config: bool = False
    # When given and true for the result's metrics, the caveat is raised
    # to warn and `escalated_text` (formatted like `text`) is appended.
    escalate: Callable[[Mapping[str, Any]], bool] | None = None
    escalated_text: str = ""


def _eq(v: object) -> Callable[[Any], bool]:
    return lambda x: x == v


_PMSM_CIRCUIT = frozenset({"rated_torque", "stall_torque"})
_OP = frozenset({"operating_point", "motor_thermal"})
_SIM = frozenset({"drive_sim", "drive_sim_two_mass"})
_IM_FEM = frozenset({"im_fem_harmonic", "im_fem_locked_rotor"})

RULES: tuple[Rule, ...] = (
    Rule("constant_l_envelope", frozenset({"torque_speed"}), "envelope_source",
         _eq("analytical"), "warn",
         "constant-L envelope: saturation is ignored, so high-current torque "
         "and power are optimistic; a [flux_map] gives the saturated envelope"),
    Rule("constant_l_mtpa", _PMSM_CIRCUIT, "mtpa_source", _eq("analytical"), "info",
         "constant-L MTPA point (no [flux_map] covering this current)"),
    Rule("stall_saturation", frozenset({"stall_torque"}), "saturation_warning",
         _eq(True), "warn",
         "I_stall is {saturation_ratio:.3g}× I_rated; constant-L torque this far "
         "past rated current is likely optimistic"),
    Rule("constant_l_operating_point", _OP, "op_source", _eq("analytical"), "info",
         "constant-L operating point (no [flux_map])"),
    Rule("constant_l_machine", _SIM, "machine_model", _eq("constant_L"), "info",
         "constant-L machine model (no [flux_map])"),
    Rule("backemf_from_psi_f",
         frozenset({"analytical", "analytical_axial", "fem_axial"}),
         "backemf_source", _eq("psi_f"), "info",
         "backemf_fundamental comes from the motor's stated psi_f, not the "
         "field solution; agreement with a back-EMF measurement it was "
         "fitted to is by construction"),
    Rule("k_end_heuristic", frozenset({"analytical"}), "k_end_source",
         _eq("heuristic"), "info",
         "end-effect factor k_end is a heuristic, not a computed 3D value"),
    # stored as 1.0 when Newton fell back to Picard; absent otherwise
    Rule("newton_fell_back", frozenset({"fem"}), "newton_fell_back", bool, "info",
         "nonlinear Newton solve fell back to Picard iteration"),
    Rule("linear_iron", frozenset({"fem"}), "nonlinear", _eq(False), "info",
         "linear iron (nonlinear=false): saturation is not modelled",
         on_config=True,
         escalate=lambda m: isinstance(m.get("b_iron_max"), (int, float))
         and m["b_iron_max"] > B_IRON_SATURATION_T,
         escalated_text=f"; b_iron_max = {{b_iron_max:.2f}} T exceeds the "
                        f"{B_IRON_SATURATION_T} T saturation threshold, so the "
                        "field is optimistic; rerun with nonlinear=true"),
    Rule("im_linear_iron", _IM_FEM, "iron_model", _eq("linear"), "info",
         "linear iron: saturation is not modelled"),
    Rule("im_ideal_ring", _IM_FEM, "ring_model", _eq("ideal"), "info",
         "ideal end ring"),
    Rule("im_no_skew", _IM_FEM, "skew_model", _eq("none"), "info",
         "rotor skew not modelled"),
    Rule("budget_from_rated_current", frozenset({"thermal_duty"}), "budget_source",
         _eq("rated_current"), "info",
         "S1 loss budget inferred from rated current, not a thermal resistance"),
)


def _fmt(text: str, value: Any, metrics: Mapping[str, Any]) -> str:
    try:
        return text.format(value=value, **metrics)
    except (KeyError, ValueError, TypeError, IndexError):
        return text


def assess(
    model: str, status: str, metrics: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> list[Caveat]:
    """Caveats for one result; empty for non-OK or non-computed results.

    ``config`` is the run's RunConfig fields as a mapping (only rules
    with ``on_config`` read it)."""
    info = MODEL_REGISTRY.get(model)
    if status != "OK" or info is None or info.source != "computed":
        return []
    metrics = metrics or {}
    config = config or {}
    out: list[Caveat] = []
    for r in RULES:
        if model not in r.models:
            continue
        src = config if r.on_config else metrics
        if r.key not in src:
            continue
        v = src[r.key]
        if not r.when(v):
            continue
        basis = f"{r.key}={json_scalar(v)}"
        if r.escalate is not None and r.escalate(metrics):
            out.append(Caveat(r.code, "warn",
                              _fmt(r.text + r.escalated_text, v, metrics), basis))
        else:
            out.append(Caveat(r.code, r.severity, _fmt(r.text, v, metrics), basis))
    out.extend(Caveat("limits", "info", text, "limits") for text in info.limits)
    return out


def measured_available(n_records: int, config_key: str, motor_name: str) -> Caveat:
    """Info caveat for a computed result whose motor has measured or
    published records in the store: points the reader at the comparison
    instead of leaving them to recall a number from memory."""
    plural = "s" if n_records != 1 else ""
    return Caveat(
        code="measured_data_available", severity="info",
        text=(f"{n_records} measured/published record{plural} exist for this motor; "
              f"validation_summary(motor={config_key!r}) compares them with this result"),
        basis=f"store: {n_records} measured/published record{plural} for {motor_name!r}",
    )


def json_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def headline(model: str, metrics: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """``[{key, value, unit}]`` for the model's headline entries present
    in ``metrics``; ``max:k`` takes the maximum of a numeric list."""
    info = MODEL_REGISTRY.get(model)
    if info is None or not metrics:
        return []
    out: list[dict[str, Any]] = []
    for entry in info.headline:
        k = headline_key(entry)
        if k not in metrics or metrics[k] is None:
            continue
        v = metrics[k]
        if entry.startswith("max:"):
            nums = [x for x in v if isinstance(x, (int, float)) and x == x] \
                if isinstance(v, list) else []
            if not nums:
                continue
            v = max(nums)
        out.append({"key": entry, "value": v, "unit": info.units.get(entry, "")})
    return out


def _record_model(rec: Mapping[str, Any]) -> str:
    from phasesweep.sweep_types import resolve_model
    try:
        return resolve_model(rec)
    except KeyError:
        return ""


def assess_record(rec: Mapping[str, Any]) -> list[Caveat]:
    """`assess` over a stored result-store record."""
    return assess(_record_model(rec), rec.get("status", "ERROR"),
                  rec.get("metrics"), rec.get("config"))


def headline_record(rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """`headline` over a stored result-store record."""
    if rec.get("status") != "OK":
        return []
    return headline(_record_model(rec), rec.get("metrics"))
