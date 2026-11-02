"""Headline metrics and provenance-derived caveats (phasesweep.trust)."""

from __future__ import annotations

import pytest

from phasesweep.registry import MODEL_REGISTRY, headline_key
from phasesweep.trust import RULES, assess, assess_record, headline, headline_record

COMPUTED = {k: v for k, v in MODEL_REGISTRY.items() if v.source == "computed"}



# -- registry guards -----------------------------------------------------------

@pytest.mark.parametrize("key", sorted(COMPUTED))
def test_computed_model_has_headline(key):
    info = COMPUTED[key]
    assert 2 <= len(info.headline) <= 5, key


@pytest.mark.parametrize("key", sorted(COMPUTED))
def test_headline_keys_are_produced(key):
    info = COMPUTED[key]
    for entry in info.headline:
        assert headline_key(entry) in info.produces, (key, entry)


@pytest.mark.parametrize("key", sorted(COMPUTED))
def test_units_cover_headline(key):
    info = COMPUTED[key]
    assert set(info.units) == set(info.headline), key


@pytest.mark.parametrize("key", sorted(COMPUTED))
def test_limits_are_sentences(key):
    for text in COMPUTED[key].limits:
        assert text and not text.endswith(".") and len(text) < 200, (key, text)


def test_measured_models_have_no_trust_metadata():
    for key, info in MODEL_REGISTRY.items():
        if info.source == "measured":
            assert not info.headline and not info.limits, key


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.code)
def test_rule_keys_are_produced(rule):
    for model in rule.models:
        info = MODEL_REGISTRY[model]
        if rule.on_config:
            assert rule.key in info.hash_fields, (rule.code, model)
        else:
            assert rule.key in info.produces, (rule.code, model)


def test_rule_codes_unique():
    codes = [r.code for r in RULES]
    assert len(codes) == len(set(codes))


# -- one synthetic record per rule ----------------------------------------------

_TRIGGERS = {
    "constant_l_envelope": ("torque_speed", {"envelope_source": "analytical"}, None),
    "constant_l_mtpa": ("rated_torque", {"mtpa_source": "analytical"}, None),
    "stall_saturation": ("stall_torque",
                         {"saturation_warning": True, "saturation_ratio": 4.25}, None),
    "constant_l_operating_point": ("operating_point", {"op_source": "analytical"}, None),
    "constant_l_machine": ("drive_sim", {"machine_model": "constant_L"}, None),
    "backemf_from_psi_f": ("analytical", {"backemf_source": "psi_f"}, None),
    "k_end_heuristic": ("analytical", {"k_end_source": "heuristic"}, None),
    "newton_fell_back": ("fem", {"newton_fell_back": True}, None),
    "linear_iron": ("fem", {}, {"nonlinear": False}),
    "im_linear_iron": ("im_fem_harmonic", {"iron_model": "linear"}, None),
    "im_ideal_ring": ("im_fem_locked_rotor", {"ring_model": "ideal"}, None),
    "im_no_skew": ("im_fem_harmonic", {"skew_model": "none"}, None),
    "budget_from_rated_current": ("thermal_duty", {"budget_source": "rated_current"}, None),
}


def test_every_rule_has_a_trigger_case():
    assert set(_TRIGGERS) == {r.code for r in RULES}


@pytest.mark.parametrize("code", sorted(_TRIGGERS))
def test_rule_fires(code):
    model, metrics, config = _TRIGGERS[code]
    cav = {c.code: c for c in assess(model, "OK", metrics, config)}
    assert code in cav
    rule = next(r for r in RULES if r.code == code)
    assert cav[code].severity == rule.severity
    assert cav[code].basis.startswith(f"{rule.key}=")


def test_rule_silent_on_other_value():
    assert not [c for c in assess("torque_speed", "OK", {"envelope_source": "flux_map"})
                if c.code != "limits"]
    assert not [c for c in assess("fem", "OK", {}, {"nonlinear": True})
                if c.code != "limits"]


@pytest.mark.parametrize(("b", "sev"), [(1.5, "info"), (2.06, "warn")])
def test_linear_iron_escalates_past_saturation(b, sev):
    (c,) = [c for c in assess("fem", "OK", {"b_iron_max": b}, {"nonlinear": False})
            if c.code == "linear_iron"]
    assert c.severity == sev and c.basis == "nonlinear=false"
    assert ("2.06 T exceeds" in c.text) == (sev == "warn")


def test_saturation_text_carries_ratio():
    (c,) = [c for c in assess("stall_torque", "OK",
                              {"saturation_warning": True, "saturation_ratio": 4.25})
            if c.code == "stall_saturation"]
    assert "4.25×" in c.text and c.basis == "saturation_warning=true"


def test_limits_on_every_ok_result():
    lim = [c for c in assess("fem", "OK", {}) if c.code == "limits"]
    assert [c.text for c in lim] == list(MODEL_REGISTRY["fem"].limits)


@pytest.mark.parametrize("status", ["ERROR", "TIMEOUT"])
def test_non_ok_gets_nothing(status):
    assert assess("torque_speed", status, {"envelope_source": "analytical"}) == []


def test_measured_and_unknown_get_nothing():
    assert assess("resistance_test", "OK", {"R_s": 0.1}) == []
    assert assess("no_such_model", "OK", {}) == []


# -- headline ---------------------------------------------------------------------

def test_headline_max_of_list_and_units():
    h = headline("torque_speed", {"tau_curve_peak": [3.0, 9.5, 1.0], "p_max_peak": 100.0})
    by = {x["key"]: x for x in h}
    assert by["max:tau_curve_peak"]["value"] == 9.5
    assert by["max:tau_curve_peak"]["unit"] == MODEL_REGISTRY["torque_speed"].units[
        "max:tau_curve_peak"]
    assert "base_speed_peak" not in by  # absent key skipped


def test_headline_order_follows_registry():
    info = MODEL_REGISTRY["analytical"]
    metrics = {headline_key(e): 1.0 for e in info.headline}
    assert [x["key"] for x in headline("analytical", metrics)] == list(info.headline)


def test_record_wrappers():
    rec = {"model": "torque_speed", "status": "OK",
           "metrics": {"envelope_source": "analytical", "tau_curve_peak": [1.0, 2.0]},
           "config": {}}
    assert "constant_l_envelope" in {c.code for c in assess_record(rec)}
    assert headline_record(rec)[0]["value"] == 2.0
    assert headline_record({**rec, "status": "ERROR"}) == []
    # legacy run_type key resolves
    assert assess_record({**rec, "model": None, "run_type": "torque_speed"})


# -- real model output -------------------------------------------------------------

def test_prius_torque_speed_carries_envelope_warning():
    """The case that motivated the feature: Prius 2004 constant-L envelope
    peaks near 905 N·m against ~340 N·m measured (ORNL/TM-2004/185)."""
    from phasesweep.machines.configs import load_motor
    from phasesweep.sweep_types import RunConfig

    motor = load_motor("motors/prius_2004.toml")
    cfg = RunConfig(motor=motor, model="torque_speed")
    metrics = MODEL_REGISTRY["torque_speed"].fn(cfg)
    assert metrics["envelope_source"] == "analytical"
    cav = assess("torque_speed", "OK", metrics)
    (warn,) = [c for c in cav if c.code == "constant_l_envelope"]
    assert warn.severity == "warn" and warn.basis == "envelope_source=analytical"
    peak = next(h for h in headline("torque_speed", metrics)
                if h["key"] == "max:tau_curve_peak")
    assert peak["value"] > 340.0 and peak["unit"] == "N·m"
