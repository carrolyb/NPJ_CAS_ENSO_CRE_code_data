# Figure 07 Package

This directory is the packaged archive for the Figure 07 version that uses a `+/-0.5 C` Nino3.4 ENSO definition.

Scientific role
- Figure 07 shows cloud-group-resolved daytime SW, LW, and Net total pathways across WP, CP, and EP.
- Displayed values are El Nino minus La Nina pathway differences.
- ENSO months are defined from `nino34_anom` with El Nino `>= +0.5 C` and La Nina `<= -0.5 C`.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: package-local pointers to monthly chain inputs and ENSO index data.
- `04_key_results/`: plot-data table and group SW/LW/Net summaries.
- `05_notes/`: caption, method notes, and input manifest.

Important notes
- The plotting style follows the `Figure07 ... copy.py` 1x3 heatmap layout.
- Net rows are cross-checked against the packaged `Figure06_degC05` group totals.
- The script keeps the `SW + LW = Net` closure as a hard validation gate.

Main files
- Final figure: `01_final_figure/Figure07_SW_LW_Net_total_pathways_degC05.png`
- Plotting script: `02_plotting_script/make_figure07_SW_LW_Net_total_pathways_degC05.py`
- Plot data: `04_key_results/Figure07_degC05_plot_data.csv`
- Method notes: `05_notes/Figure07_degC05_method_and_plot_checks.txt`
