"""PMDC/BLDC Motor Design Demo — thin entry point."""

from phasesweep.configs import CONFIGS
from phasesweep.sim import (
    run_all, run_sweep, run_field_fem, run_field_fem_slotted,
    run_armature_decomposition, run_slot_sweep,
)
from phasesweep.plots import (
    plot_results, plot_sweep, plot_field_polar, plot_field_comparison,
    plot_field_slotted, plot_cross_section, plot_armature_reaction,
    plot_slot_sweep,
)

if __name__ == "__main__":
    results = run_all()
    plot_results(results)

    print("\n--- Parameter sweep ---")
    grid = run_sweep()
    plot_sweep(grid)

    print("\n--- Field cross-section (analytical) ---")
    plot_field_polar(CONFIGS)

    print("\n--- FEM field comparison ---")
    fem_results, harmonics = run_field_fem()
    plot_field_comparison(CONFIGS, fem_results, harmonics)

    print("\n--- Slotted FEM field + winding currents ---")
    slotted_results, slotted_harmonics = run_field_fem_slotted()
    plot_field_slotted(CONFIGS, slotted_results, fem_results, slotted_harmonics)

    print("\n--- 2D cross-section field map (slotted) ---")
    plot_cross_section(CONFIGS)

    print("\n--- Armature reaction decomposition ---")
    theta, comps, harms, cname, Q, n_p = run_armature_decomposition()
    plot_armature_reaction(theta, comps, harms, cname, Q, n_p)

    print("\n--- Slot count sweep ---")
    sweep_res, n_p_sw, cname_sw = run_slot_sweep()
    plot_slot_sweep(sweep_res, n_p_sw, cname_sw)
