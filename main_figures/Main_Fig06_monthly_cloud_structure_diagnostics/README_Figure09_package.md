# Figure 09 Package

This directory is the packaged archive for the Figure 09 version that keeps the all-month monthly representativeness scatterplots but evaluates ENSO direction robustness using a `+/-0.5 C` Nino3.4 definition.

Scientific role
- Figure 09 shows monthly representativeness of cloud-structure diagnostics for cloud-type-resolved daytime Net CRE pathways across WP, CP, and EP.
- The displayed scatter points and fitted lines use the full 248-month anomaly series in each region.
- The `+/-0.5 C` ENSO definition is applied only to the sensitivity audit that checks whether the all-month relationship signs remain consistent within ENSO-only months.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: package-local copies of the fixed Step09-3A / Step09-3B inputs plus the Nino3.4 index.
- `04_key_results/`: packaged plot-data table and derived `degC05` ENSO sensitivity table.
- `05_notes/`: caption, method notes, and input manifest.

Important notes
- The plotting style follows the `Figure09 ... v2 copy.py` 3x3 scatter-layout presentation.
- Regression coefficients and bootstrap confidence intervals are reused from the fixed Step09-3B all-month audit.
- The `degC05` update changes only the ENSO sensitivity bookkeeping, not the plotted 248-month scatter data.

Main files
- Final figure: `01_final_figure/Figure09_monthly_diagnostic_representativeness_degC05.png`
- Plotting script: `02_plotting_script/make_figure09_monthly_diagnostic_representativeness_degC05.py`
- Plot data: `04_key_results/Figure09_degC05_plot_data.csv`
- ENSO sensitivity summary: `04_key_results/Figure09_degC05_ENSO_subset_direction_sensitivity.csv`
- Method notes: `05_notes/Figure09_degC05_method_and_plot_checks.txt`
