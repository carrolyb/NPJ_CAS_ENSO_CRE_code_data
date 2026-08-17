# Figure 01 Package

This directory is the packaged archive for the manuscript's Figure 01 main version.

Scientific role
- Figure 01 main spatial CRE response over the tropical Pacific.
- Latitude range for the map and regional boxes: 15S-15N.
- Main display choice: no significance stippling on the final figure, to keep the spatial pattern visually clean for the opening figure.

Directory structure
- `01_final_figure/`: final manuscript-ready figure files.
- `02_plotting_script/`: plotting script used to generate the figure.
- `03_input_data/`: direct input files used by the plotting script.
- `04_key_results/`: key quantitative outputs for manuscript writing.
- `05_notes/`: methods and packaging notes.

Important notes
- Manuscript text uses the standard spellings `El Niño` and `La Niña`.
- The copied classification CSV keeps the original phase labels `El Nino` and `La Nina` for script compatibility.
- `03_input_data/ceres_monthly_anomalies.nc` is a symbolic link to the canonical processed anomaly dataset, to avoid duplicating a 1.3 GB file for each figure package.

Main files
- Final figure: `01_final_figure/Figure01_main_CRE_lat15_nosig.png`
- Significance reference: `01_final_figure/Figure01_reference_with_significance_lat15.png`
- Plotting script: `02_plotting_script/make_figure01_main_cre_lat15.py`
- Regional summary table: `04_key_results/Figure01_regional_CRE_summary_lat15.csv`
- Key manuscript text: `04_key_results/Figure01_key_numbers_for_manuscript.md`
