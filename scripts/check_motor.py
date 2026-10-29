"""Thin shim — the implementation lives in phasesweep.check_motor.

Prefer: phasesweep-check-motor motors/my_motor.toml
"""
from phasesweep.check_motor import main

if __name__ == "__main__":
    main()
