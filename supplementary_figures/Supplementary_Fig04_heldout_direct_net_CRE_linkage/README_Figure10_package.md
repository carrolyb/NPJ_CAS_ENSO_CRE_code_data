# Figure 10 Package

This directory is the packaged archive for the Figure 10 version that keeps the simplified 1x3 held-out scatter layout while re-evaluating ENSO phase-episode robustness under a `+/-0.5 C` monthly Nino3.4 definition.

Scientific role
- Figure 10 shows held-out diagnostic linkage between cloud-structure metrics and regional direct daytime Net CRE variability across WP, CP, and EP.
- The displayed scatter panels use the fixed 248-month purged blocked cross-validation outputs from the approved all-month Step10-1B-2A chain.
- The `+/-0.5 C` ENSO definition is applied only to the phase-episode robustness bookkeeping that accompanies the removed auxiliary panels.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: package-local copies of the fixed Step10 inputs, the Nino3.4 index, and read-only v1 references.
- `04_key_results/`: plot-data table and derived `degC05` phase-episode robustness tables.
- `05_notes/`: caption, method notes, and input manifest.

Important notes
- The plotting style follows the `Figure10 ... v2 copy.py` simplified 1x3 scatter layout.
- Held-out scatter values and panel annotations remain tied to the fixed all-month blocked-CV outputs.
- The `degC05` update changes only the ENSO phase-episode robustness summaries retained in the caption and method notes.

Main files
- Final figure: `01_final_figure/Figure10_heldout_direct_diagnostic_linkage_degC05.png`
- Plotting script: `02_plotting_script/make_figure10_heldout_direct_diagnostic_linkage_degC05.py`
- Plot data: `04_key_results/Figure10_degC05_plot_data.csv`
- Pooled phase-episode held-out skill: `04_key_results/Figure10_degC05_pooled_phase_episode_heldout_skill.csv`
- Phase-episode bootstrap increment table: `04_key_results/Figure10_degC05_phase_episode_cluster_bootstrap_skill_increment.csv`
- Method notes: `05_notes/Figure10_degC05_method_and_plot_checks.txt`
