# Figure 03 Package

This directory is the packaged archive for the Figure 03 version that uses a `+/-0.5 C` Nino3.4 ENSO definition.

Scientific role
- Figure 03 shows ENSO-associated reorganization of cloud-type occurrence across the tropical Pacific.
- Displayed values are El Nino minus La Nina cloud-fraction anomalies.
- ENSO months are defined from `nino34_anom` with El Nino `>= +0.5 C` and La Nina `<= -0.5 C`.

Directory structure
- `01_final_figure/`: final manuscript-style figure files.
- `02_plotting_script/`: package-local script that regenerates the figure and packaged tables.
- `03_input_data/`: direct inputs used by the package-local script.
- `04_key_results/`: 42-class table, five-group summary, and bootstrap summary.
- `05_notes/`: validation, method notes, input manifest, and supporting check figure.

Important notes
- This package is separate from the later `64/91` full-CF main-chain Figure 03 package.
- Black dots mark 42 cloud-type cells whose 95% moving-block-bootstrap confidence intervals exclude zero.
- `03_input_data/ceres_monthly_regional_anomalies.nc` and `03_input_data/ceres_monthly_regional_climatology.nc` are symbolic links to canonical processed datasets, so the package stays lightweight.

Main files
- Final figure: `01_final_figure/Figure03_cloudtype_occurrence_degC05.png`
- Plotting script: `02_plotting_script/make_figure03_cloudtype_occurrence_degC05.py`
- 42-class plot data: `04_key_results/Figure03_degC05_cloud_type_CF_anomaly_42class.csv`
- Validation summary: `05_notes/Figure03_degC05_validation.txt`
- Key numbers: `04_key_results/Figure03_degC05_key_numbers.md`
