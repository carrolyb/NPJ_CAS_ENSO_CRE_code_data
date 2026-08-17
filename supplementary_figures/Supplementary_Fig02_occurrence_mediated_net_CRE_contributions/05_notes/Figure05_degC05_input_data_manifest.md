# Figure 05 degC05 Input Data Manifest

- `CERES_regional_integrated_candidate_monthly.nc`: linked regional candidate monthly contribution chain for WP/CP/EP.
- `Step08_0D3A_TP_candidate_monthly_contribution.nc`: linked TP monthly contribution chain.
- `Step08_0D2A2a_Figure04_candidate_regional_conditional_CRE.csv`: WP/CP/EP Figure 4 kernel table.
- `Step08_0D3A_Figure04_TP_candidate_conditional_CRE.csv`: TP Figure 4 kernel table.
- `nino34_200207_202302.csv`: local monthly Nino3.4 index file; this package uses `nino34_anom` with `+/-0.5 C`.

Generated outputs
- `Figure05_degC05_cell_occurrence_Net.csv`: deterministic per-cell occurrence contribution table.
- `Figure05_degC05_cell_occurrence_Net_bootstrap.csv`: per-cell bootstrap CI and significance table.
- `Figure05_degC05_final_plot_input.csv`: final plotting table consumed by the redraw script.
- `Figure05_degC05_group_occurrence_summary.csv`: deterministic five-group occurrence summary.
- `Figure05_degC05_regional_occurrence_summary.csv`: deterministic regional sum42 summary.
