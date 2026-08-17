# Figure 04 Package

This directory is the packaged archive for the Figure 04 redraw that follows the `copy.py` layout style and is grouped under the current `degC05` figure-organizing workflow.

Scientific role
- Figure 04 shows contribution-consistent climatological daytime conditional CRE kernels for the 42 CERES cloud types.
- Panels (a)–(c) show TP SW, LW, and Net kernels; panels (d)–(f) show Net kernels for WP, CP, and EP.
- The plotted kernel values are climatological `CRE0 = mean(Q) / mean(CF)` quantities and do not numerically depend on the ENSO threshold definition.

Directory structure
- `01_final_figure/`: final figure files.
- `02_plotting_script/`: package-local plotting script using the Figure04 `copy.py` drawing method.
- `03_input_data/`: packaged candidate plot-input tables.
- `04_key_results/`: plot data exported from the final redraw.
- `05_notes/`: caption, method checks, provenance, and supporting audit records.

Important notes
- `degC05` here is a chain label for consistency with the current figure-organization workflow; Figure 04 kernel values themselves are climatological and unchanged by switching ENSO thresholds.
- The CP low-sample cell remains hatched at `cloud_type=6`, `CTP=1000-800`, `tau=60.36-378.65`, `valid_n=17`.
- The plotting style follows [make_fig04_candidate_conditional_CRE_final_v1 copy.py](/Volumes/My%20Book/P3/figure_optimization_workspace/Figures01_10_final_chain_v1/Figure04_conditional_CRE_kernels/02_plotting_script_original/make_fig04_candidate_conditional_CRE_final_v1%20copy.py).

Main files
- Final figure: `01_final_figure/Figure04_conditional_CRE_kernels_degC05.png`
- Plotting script: `02_plotting_script/make_figure04_conditional_CRE_kernels_degC05.py`
- Plot-data export: `04_key_results/Figure04_degC05_plot_data.csv`
- Caption: `05_notes/Figure04_degC05_caption.md`
- Method/check record: `05_notes/Figure04_degC05_method_and_plot_checks.txt`
