# Code and Data Package for Submission

Manuscript: Cloud type compensation shapes ENSO-related cloud radiative variability over the tropical Pacific

This package is organized by manuscript figure number. Main figures and supplementary figures are stored separately. Each figure folder contains the available plotting scripts, processed figure-level source data, and notes/check files. Final figure image files are not included in this archive because they are submitted separately with the manuscript.

## Main figures

- `main_figures/Main_Fig01_direct_daytime_CRE_spatial_patterns`: main Fig. 1, spatial patterns of El Nino minus La Nina daytime SW, LW, and net CRE anomalies.
- `main_figures/Main_Fig02_cloud_type_occurrence_anomalies`: main Fig. 2, cloud-type occurrence anomalies across the 42 CERES cloud-type classes.
- `main_figures/Main_Fig03_conditional_CRE_kernels`: main Fig. 3, climatological daytime conditional CRE kernels.
- `main_figures/Main_Fig04_net_CRE_contribution_decomposition`: main Fig. 4, regional net CRE contribution decomposition into occurrence and CRE adjustment components.
- `main_figures/Main_Fig05_SW_LW_net_cloud_group_contributions`: main Fig. 5, SW, LW, and net cloud-group contribution components.
- `main_figures/Main_Fig06_monthly_cloud_structure_diagnostics`: main Fig. 6, monthly diagnostic relationships between cloud-structure metrics and cloud-group net CRE contribution anomalies.

## Supplementary figures

- `supplementary_figures/Supplementary_Fig01_regional_direct_CRE_and_threshold_sensitivity`: Supplementary Fig. 1, regional direct CRE contrasts, SW-LW compensation, and ENSO-threshold sensitivity.
- `supplementary_figures/Supplementary_Fig02_occurrence_mediated_net_CRE_contributions`: Supplementary Fig. 2, occurrence-mediated net CRE contributions based on cloud occurrence anomalies and climatological net CRE kernels.
- `supplementary_figures/Supplementary_Fig03_spatial_cloud_group_net_contributions`: Supplementary Fig. 3, spatial patterns of net total contribution for major cloud groups.
- `supplementary_figures/Supplementary_Fig04_heldout_direct_net_CRE_linkage`: Supplementary Fig. 4, held-out diagnostic linkage between cloud-structure metrics and independently calculated direct net CRE anomalies.

## Contents within each figure folder

- `02_plotting_script/`: Python plotting scripts used to generate the figure.
- `03_input_data_regular_files/`: small regular input files included with the figure package, where applicable.
- `04_key_results/`: processed figure-level source data used for plotting and numerical checks.
- `05_notes/`: captions, method notes, input manifests, and validation/check files.

## Core processing scripts

The folder `core_processing_scripts/` contains the main scripts used to prepare the ENSO index file, process CERES monthly data, construct anomalies, aggregate cloud types, and generate figure products. These scripts document the analysis workflow beyond the figure-level plotting scripts.

## Data sources

The original CERES satellite data and NOAA CPC ENSO index data are publicly available and are not duplicated in this package. Large intermediate CERES monthly NetCDF files are also not included. The processed figure-level source data in this package are intended to support figure reproduction and manuscript-value checks.

CERES FluxByCldTyp-Day Terra-Aqua MODIS Edition 4.1:

https://ceres.larc.nasa.gov/data

CERES product DOI:

https://doi.org/10.5067/Terra-Aqua/CERES/FluxByCldTyp-DAY_L3.004A

NOAA CPC ERSSTv5 monthly Nino indices:

https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii

## ENSO definition

The baseline El Nino and La Nina composites use Nino3.4 anomaly thresholds of +0.5 degrees C and -0.5 degrees C, respectively. Threshold-sensitivity tests use +/-0.75 degrees C and +/-1.0 degrees C.
