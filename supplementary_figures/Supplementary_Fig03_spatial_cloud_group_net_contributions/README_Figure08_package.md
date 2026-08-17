# Figure 08 Package

This directory is the packaged archive for the Figure 08 version that uses a `+/-0.5 C` Nino3.4 ENSO definition.

Scientific role
- Figure 08 shows spatial Net total-contribution pathways for the four dominant cloud groups across the tropical Pacific.
- Displayed values are El Nino minus La Nina pathway differences.
- ENSO months are defined from `nino34_anom` with El Nino `>= +0.5 C` and La Nina `<= -0.5 C`.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: package-local pointers to the gridded monthly source file and ENSO index data.
- `04_key_results/`: final plot-input NetCDF, bootstrap NetCDF, and summary tables.
- `05_notes/`: caption, method notes, and input manifest.

Important notes
- The plotting style follows the `Figure08 ... v2 copy.py` four-panel map layout.
- Pointwise stippling is recomputed under the `0.5C` ENSO definition using a 12-month moving-block bootstrap.
- Regional sign/significance checks are compared against the packaged `Figure07_degC05` Net pathways.

Main files
- Final figure: `01_final_figure/Figure08_spatial_Net_total_pathways_degC05.png`
- Plotting script: `02_plotting_script/make_figure08_spatial_Net_total_pathways_degC05.py`
- Final plot input: `04_key_results/Figure08_degC05_final_plot_input.nc`
- Method notes: `05_notes/Figure08_degC05_method_and_plot_checks.txt`
