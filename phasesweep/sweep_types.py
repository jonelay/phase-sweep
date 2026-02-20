"""Type-safe, validated configuration and result objects for motor parameter sweeps."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from phasesweep.configs import FullMotorConfig

Status = Literal["OK", "TIMEOUT", "ERROR"]


@dataclass(frozen=True)
class MotorSweepConfig:
    """Immutable, validated motor sweep configuration.

    Covers both motulator simulation and NGSolve FEM runs via a single type.
    FEM-specific fields (n_slots, j_s, n_theta, maxh) are ignored by sim_runner;
    sim-specific fields (load_torque, load_time, t_stop) are ignored by fem_runner.
    """

    # Core motor params (both runners)
    n_p: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float
    J: float

    # Simulation params
    load_torque: float = 3.0
    load_time: float = 1.2
    t_stop: float = 1.8

    # FEM params
    n_slots: int = 0
    j_s: float = 0.0
    n_theta: int = 360
    maxh: float = 0.04
    nonlinear: bool = False

    # Winding params (forwarded to solve_field_fem)
    N: int = 50
    k_w: float = 0.966
    L_stk: float = 0.10

    def __post_init__(self) -> None:
        if not (1 <= self.n_p <= 20):
            raise ValueError(f"n_p={self.n_p} outside range [1, 20]")
        if not (1e-4 <= self.R_s <= 100.0):
            raise ValueError(f"R_s={self.R_s} outside range [1e-4, 100]")
        if not (1e-6 <= self.L_d <= 1.0):
            raise ValueError(f"L_d={self.L_d} outside range [1e-6, 1.0]")
        if not (1e-6 <= self.L_q <= 1.0):
            raise ValueError(f"L_q={self.L_q} outside range [1e-6, 1.0]")
        if not (1e-4 <= self.psi_f <= 10.0):
            raise ValueError(f"psi_f={self.psi_f} outside range [1e-4, 10.0]")
        if not (1e-6 <= self.J <= 100.0):
            raise ValueError(f"J={self.J} outside range [1e-6, 100.0]")
        if not (0 <= self.n_slots <= 200):
            raise ValueError(f"n_slots={self.n_slots} outside range [0, 200]")

    def to_motor_config(self) -> FullMotorConfig:
        """Convert to MotorConfig-compatible dict for build_sim()."""
        return {
            "n_p": self.n_p, "R_s": self.R_s, "L_d": self.L_d, "L_q": self.L_q,
            "psi_f": self.psi_f, "J": self.J, "n_slots": self.n_slots, "j_s": self.j_s,
            "N": self.N, "k_w": self.k_w, "L_stk": self.L_stk,
        }

    @property
    def config_id(self) -> str:
        """Deterministic hash for deduplication and resume."""
        key = (
            f"{self.n_p}_{self.R_s:.6f}_{self.L_d:.3e}_{self.L_q:.3e}_"
            f"{self.psi_f:.6f}_{self.J:.6f}_"
            f"{self.load_torque:.3f}_{self.load_time:.3f}_{self.t_stop:.3f}_"
            f"{self.n_slots}_{self.j_s:.6f}_{self.n_theta}_{self.maxh:.4f}_"
            f"{self.nonlinear}_{self.N}_{self.k_w:.4f}_{self.L_stk:.4f}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_p": self.n_p,
            "R_s": self.R_s,
            "L_d": self.L_d,
            "L_q": self.L_q,
            "psi_f": self.psi_f,
            "J": self.J,
            "load_torque": self.load_torque,
            "load_time": self.load_time,
            "t_stop": self.t_stop,
            "n_slots": self.n_slots,
            "j_s": self.j_s,
            "n_theta": self.n_theta,
            "maxh": self.maxh,
            "nonlinear": self.nonlinear,
            "N": self.N,
            "k_w": self.k_w,
            "L_stk": self.L_stk,
            "config_id": self.config_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MotorSweepConfig:
        return cls(
            n_p=d["n_p"],
            R_s=d["R_s"],
            L_d=d["L_d"],
            L_q=d["L_q"],
            psi_f=d["psi_f"],
            J=d["J"],
            load_torque=d.get("load_torque", 3.0),
            load_time=d.get("load_time", 1.2),
            t_stop=d.get("t_stop", 1.8),
            n_slots=d.get("n_slots", 0),
            j_s=d.get("j_s", 0.0),
            n_theta=d.get("n_theta", 360),
            maxh=d.get("maxh", 0.04),
            nonlinear=d.get("nonlinear", False),
            N=d.get("N", 50),
            k_w=d.get("k_w", 0.966),
            L_stk=d.get("L_stk", 0.10),
        )


@dataclass
class SweepResult:
    """Result from a single sweep configuration run.

    status:
      OK      — completed successfully, metrics populated
      TIMEOUT — process killed after timeout, metrics None
      ERROR   — exception in subprocess, error_msg populated

    metrics keys vary by run_type:
      sim: t_settle, i_ss, speed_droop, tau_peak
      fem: peak_Br, fundamental, thd_pct, sh_pct
    """

    config: MotorSweepConfig
    run_type: Literal["sim", "fem"]
    status: Status
    metrics: dict[str, Any] | None
    elapsed_s: float
    error_msg: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: str = "v1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "run_type": self.run_type,
            "status": self.status,
            "metrics": self.metrics,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SweepResult:
        return cls(
            config=MotorSweepConfig.from_dict(d["config"]),
            run_type=d.get("run_type", "sim"),
            status=d["status"],
            metrics=d.get("metrics"),
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
            timestamp=d.get("timestamp", ""),
            schema_version=d.get("schema_version", "v1.0"),
        )
