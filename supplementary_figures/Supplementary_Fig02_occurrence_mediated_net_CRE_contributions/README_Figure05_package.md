# Figure 05 Package

This directory is the packaged archive for the Figure 05 version that uses the `+/-0.5 C` Nino3.4 ENSO definition and the `copy.py` drawing style.

Scientific role
- Figure 05 shows occurrence-mediated daytime Net CRE contributions associated with ENSO-driven cloud-type reorganization.
- Each cell is `AmountNet = DeltaCF_paired x CRE0_Net`.
- `DeltaCF_paired` is recomputed here from the `nino34_anom` `+/-0.5 C` definition, while `CRE0_Net` is taken from the contribution-consistent climatological Figure 4 kernel chain.

Directory structure
- `01_final_figure/`: final figure files.
- `02_plotting_script/`: package-local preparation and plotting scripts.
- `03_input_data/`: local and linked inputs used by the package-local scripts.
- `04_key_results/`: deterministic, bootstrap, final-input, and exported plot-data tables.
- `05_notes/`: caption, method checks, and preparation notes.

Important notes
- This package truly changes the ENSO definition to `El Nino >= +0.5 C`, `La Nina <= -0.5 C`, so its numbers differ from the older `64/91` chain.
- The plotting layout follows [make_fig05_candidate_occurrence_Net_final_v2 copy.py](/Volumes/My%20Book/P3/figure_optimization_workspace/Figures01_10_final_chain_v1/Figure05_occurrence_mediated_Net/02_plotting_script_original/make_fig05_candidate_occurrence_Net_final_v2%20copy.py).
- The baseline hatched cell remains the same CP low-cloud cell with `valid_n = 17`.

Main files
- Final figure: `01_final_figure/Figure05_occurrence_mediated_Net_degC05.png`
- Preparation script: `02_plotting_script/prepare_figure05_occurrence_Net_degC05.py`
- Plotting script: `02_plotting_script/make_figure05_occurrence_Net_degC05.py`
- Final plot input: `04_key_results/Figure05_degC05_final_plot_input.csv`
- Preparation summary: `05_notes/Figure05_degC05_preparation_summary.txt`
