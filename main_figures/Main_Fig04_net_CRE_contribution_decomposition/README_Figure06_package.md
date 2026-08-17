# Figure 06 Package

This directory is the packaged archive for the Figure 06 version that uses a `+/-0.5 C` Nino3.4 ENSO definition.

Scientific role
- Figure 06 shows the Net CRE pathway decomposition into occurrence, CRE adjustment, and total contributions across WP, CP, and EP.
- Displayed values are El Nino minus La Nina pathway differences.
- ENSO months are defined from `nino34_anom` with El Nino `>= +0.5 C` and La Nina `<= -0.5 C`.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: package-local pointers to monthly chain inputs and ENSO index data.
- `04_key_results/`: plot-data table and summarized group/regional/direct values.
- `05_notes/`: caption, method notes, and input manifest.

Important notes
- The plotting style follows the `Figure06 ... copy.py` four-panel layout.
- Occurrence values are cross-checked against the packaged `Figure05_degC05` results.
- Direct regional Net values are cross-checked against the packaged `Figure02_degC05` results.

Main files
- Final figure: `01_final_figure/Figure06_Net_pathway_decomposition_degC05.png`
- Plotting script: `02_plotting_script/make_figure06_Net_pathway_decomposition_degC05.py`
- Plot data: `04_key_results/Figure06_degC05_plot_data.csv`
- Method notes: `05_notes/Figure06_degC05_method_and_plot_checks.txt`
