# Figure Packages 关键数据整理

整理目的：根据 `figure_packages` 中各图的绘图程序、`04_key_results` 和 `05_notes`，提炼后续文章写作最常用的关键数字。

说明：这里优先整理“可直接写进正文/图注”的结果，没有逐格重复全部 `csv`/`nc` 内容；如需展开到单个 cloud type 或单月数据，可继续回查各包内 `04_key_results`。

## Figure01_main_CRE_lat15_nosig

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure01_main_CRE_lat15_nosig/02_plotting_script/make_figure01_main_cre_lat15.py`

数据口径：2002-07 到 2023-02；ENSO 合成样本为 64 个 El Niño 月、91 个 La Niña 月。

关键数据：
- Tropical Pacific: `DeltaSW = -2.255`, `DeltaLW = 3.215`, `DeltaNet = 0.960 W m^-2`, `CE = 0.824`
- Western Pacific: `DeltaSW = 7.516`, `DeltaLW = -7.858`, `DeltaNet = -0.342 W m^-2`, `CE = 0.978`
- Central Pacific: `DeltaSW = -11.310`, `DeltaLW = 12.343`, `DeltaNet = 1.033 W m^-2`, `CE = 0.956`
- Eastern Pacific: `DeltaSW = -1.370`, `DeltaLW = 3.023`, `DeltaNet = 1.653 W m^-2`, `CE = 0.624`

写作抓手：TP 平均态呈现显著 SW-LW 补偿；CP 振幅最大；EP 补偿效率最低、残余 `Net` 最大。

## Figure01_main_CRE_lat15_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure01_main_CRE_lat15_degC05/02_plotting_script/make_figure01_main_cre_lat15.py`

数据口径：`nino34_anom` 的 `±0.5 C` ENSO 定义；54 个 El Niño 月、85 个 La Niña 月。

关键数据：
- Tropical Pacific: `DeltaSW = -2.569`, `DeltaLW = 3.576`, `DeltaNet = 1.006 W m^-2`, `CE = 0.836`
- Western Pacific: `DeltaSW = 8.129`, `DeltaLW = -8.317`, `DeltaNet = -0.189 W m^-2`, `CE = 0.989`
- Central Pacific: `DeltaSW = -12.343`, `DeltaLW = 13.234`, `DeltaNet = 0.891 W m^-2`, `CE = 0.965`
- Eastern Pacific: `DeltaSW = -1.701`, `DeltaLW = 3.472`, `DeltaNet = 1.772 W m^-2`, `CE = 0.657`

写作抓手：这是后续 `degC05` 链条的直接 CRE 基线，数值与 Figure02 的区域直接响应一致。

## Figure02_direct_regional_CRE_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure02_direct_regional_CRE_degC05/02_plotting_script/make_figure02_direct_regional_cre_degC05.py`

数据口径：`±0.5 C` ENSO；54 个 El Niño 月、85 个 La Niña 月；`2000` 次、`12` 个月 moving-block bootstrap。

关键数据：
- TP `DeltaNet = 1.006`，`95% CI = [0.292, 1.690]`，回归斜率 `0.561`，`95% CI = [0.276, 0.879]`
- WP `DeltaNet = -0.189`，`95% CI = [-0.946, 0.662]`，不显著
- CP `DeltaNet = 0.891`，`95% CI = [0.451, 1.535]`，回归斜率 `0.477`，`95% CI = [0.167, 1.010]`
- EP `DeltaNet = 1.772`，`95% CI = [0.344, 3.151]`，回归斜率 `1.062`，`95% CI = [0.416, 1.591]`

阈值敏感性：
- `0.5 -> 0.75 -> 1.0 C` 时，EP `DeltaNet` 从 `1.772 -> 2.457 -> 4.018 W m^-2`，增幅最明显
- CP 对阈值较稳健：`0.891 -> 0.827 -> 1.069 W m^-2`

## Figure03_cloudtype_occurrence_fullCF

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure03_cloudtype_occurrence_fullCF/02_plotting_script/make_figure03_cloudtype_occurrence_fullCF.py`

数据口径：formal 主版本；full valid CF record；64 个 El Niño 月、91 个 La Niña 月；`2000` 次、`12` 个月 bootstrap。

关键数据：
- 显著 42-class cell 数：`TP = 26`, `WP = 32`, `CP = 36`, `EP = 26`
- 共享色标上限：`vmax = 0.025`
- WP: `low = +0.0314`, `thin high = -0.0463`, `thick anvil = -0.0358`, `deep convective = -0.0116`
- CP: `low = -0.0581`, `thin high = +0.0677`, `thick anvil = +0.0430`, `deep convective = +0.0173`
- EP: `low = -0.0167` 且不显著，`thin high = +0.0181`, `thick anvil = +0.0112`, `deep convective = +0.0054`

写作抓手：WP 与 CP 呈现低云和高云组的反相重组，是后续 pathway 解释的结构基础。

## Figure03_cloudtype_occurrence_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure03_cloudtype_occurrence_degC05/02_plotting_script/make_figure03_cloudtype_occurrence_degC05.py`

数据口径：`±0.5 C` ENSO；54 个 El Niño 月、85 个 La Niña 月；`2000` 次、`12` 个月 bootstrap。

关键数据：
- 显著 42-class cell 数：`TP = 25`, `WP = 32`, `CP = 35`, `EP = 23`
- TP 五组 `DeltaCF`: `low = -0.0189`, `mid = +0.0063`, `thin high = +0.0208`, `thick anvil = +0.0107`, `deep convective = +0.0055`
- WP 五组 `DeltaCF`: `low = +0.0344`, `mid = -0.0137`, `thin high = -0.0488`, `thick anvil = -0.0396`, `deep convective = -0.0132`
- CP 五组 `DeltaCF`: `low = -0.0862`, `mid = +0.0217`, `thin high = +0.0867`, `thick anvil = +0.0701`, `deep convective = +0.0295`
- EP 五组 `DeltaCF`: `low = -0.0268` 不显著，`mid = +0.0103` 不显著，`thin high = +0.0259`, `thick anvil = +0.0164`, `deep convective = +0.0083`

## Figure04_conditional_CRE_kernels_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure04_conditional_CRE_kernels_degC05/02_plotting_script/make_figure04_conditional_CRE_kernels_degC05.py`

数据口径：气候态 daytime conditional CRE kernel，数值本身不依赖 ENSO 阈值。

关键数据：
- TP 最强 SW kernel: deep convective `cloud_type 42`, `-252.568 W m^-2 CF^-1`
- TP 最强 LW kernel: deep convective `cloud_type 42`, `+165.972 W m^-2 CF^-1`
- TP 最强 Net kernel: low cloud `cloud_type 6`, `-212.751 W m^-2 CF^-1`
- WP 区域最强 Net kernel: `cloud_type 18` mid-level cloud, `-184.395 W m^-2 CF^-1`
- CP 区域最强 Net kernel: `cloud_type 18` mid-level cloud, `-196.050 W m^-2 CF^-1`
- EP 区域最强 Net kernel: `cloud_type 6` low cloud, `-214.321 W m^-2 CF^-1`
- 基线低样本 hatch 仅 1 格：CP `cloud_type 6`, `CTP=1000-800`, `tau=60.36-378.65`, `valid_n = 17`

写作抓手：深对流云控制最强 SW 和 LW kernel，极厚低云/中云则给出最强负 Net kernel。

## Figure05_occurrence_mediated_Net_degC05

绘图程序：
- `/Volumes/My Book/P3/figure_packages/Figure05_occurrence_mediated_Net_degC05/02_plotting_script/prepare_figure05_occurrence_Net_degC05.py`
- `/Volumes/My Book/P3/figure_packages/Figure05_occurrence_mediated_Net_degC05/02_plotting_script/make_figure05_occurrence_Net_degC05.py`

数据口径：`AmountNet = DeltaCF_paired x CRE0_Net`；`DeltaCF_paired` 基于 `±0.5 C` ENSO 定义重算。

区域总和：
- TP `Occurrence Net = 0.490 W m^-2`
- WP `Occurrence Net = -0.253 W m^-2`
- CP `Occurrence Net = 0.849 W m^-2`
- EP `Occurrence Net = 0.728 W m^-2`

每区主导物理组：
- TP: low cloud `+0.815 W m^-2`
- WP: low cloud `-1.358 W m^-2`
- CP: low cloud `+1.920 W m^-2`
- EP: low cloud `+1.209 W m^-2`

写作抓手：Figure05 的 Net occurrence contribution 在三个东向区域主要由 low-cloud 组主导；WP 则表现为负贡献。

## Figure06_Net_pathway_decomposition_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure06_Net_pathway_decomposition_degC05/02_plotting_script/make_figure06_Net_pathway_decomposition_degC05.py`

区域合计：
- WP: `occurrence = -0.253`, `adjustment = +0.554`, `total = +0.301 W m^-2`, total 不显著，adjustment 显著
- CP: `occurrence = +0.849`, `adjustment = +0.180`, `total = +1.029 W m^-2`, `95% CI = [0.158, 1.916]`
- EP: `occurrence = +0.728`, `adjustment = +0.177`, `total = +0.905 W m^-2`, total 不显著

与 Figure02 直接 CRE 对照：
- WP direct `-0.189 W m^-2`，不显著
- CP direct `+0.891 W m^-2`，显著
- EP direct `+1.772 W m^-2`，显著

每区主导总 pathway 组：
- WP: low cloud `-1.372 W m^-2`, `95% CI = [-1.853, -0.895]`
- CP: low cloud `+1.835 W m^-2`, `95% CI = [1.486, 2.056]`
- EP: low cloud `+1.050 W m^-2`，但不显著

写作抓手：CP 的净增暖主要由 low-cloud occurrence pathway 推动；WP 存在明显的 occurrence 与 adjustment 抵消。

## Figure07_SW_LW_Net_total_pathways_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure07_SW_LW_Net_total_pathways_degC05/02_plotting_script/make_figure07_SW_LW_Net_total_pathways_degC05.py`

`Net` 主导组：
- WP: low cloud `-1.372 W m^-2`, `95% CI = [-1.853, -0.895]`
- CP: low cloud `+1.835 W m^-2`, `95% CI = [1.486, 2.056]`
- EP: low cloud `+1.050 W m^-2`，不显著

`SW` 振幅最大组均为 thick anvil：
- WP `+5.058 W m^-2`, `95% CI = [3.323, 6.792]`
- CP `-6.395 W m^-2`, `95% CI = [-8.034, -4.670]`
- EP `-1.593 W m^-2`, `95% CI = [-2.968, -0.302]`

`LW` 振幅最大组也均为 thick anvil：
- WP `-3.965 W m^-2`, `95% CI = [-5.402, -2.548]`
- CP `+5.602 W m^-2`, `95% CI = [4.221, 6.878]`
- EP `+1.321 W m^-2`, `95% CI = [0.329, 2.389]`

写作抓手：thick-anvil 组控制了最大 SW/LW 振幅，但区域 Net 主导项仍主要落在 low-cloud 组。

## Figure08_spatial_Net_total_pathways_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure08_spatial_Net_total_pathways_degC05/02_plotting_script/make_figure08_spatial_Net_total_pathways_degC05.py`

空间显著性覆盖率：
- low cloud: `2850/4800` 个格点显著，覆盖率 `0.594`
- thin high cloud: `2621/4800`，覆盖率 `0.546`
- thick anvil cloud: `2063/4800`，覆盖率 `0.430`
- deep convective cloud: `2031/4800`，覆盖率 `0.423`

空间-区域一致性：
- 与 Figure07 对照的 `12/12` 个“region x physical_group”记录均 `sign_consistent = True`
- `12/12` 个记录也均 `significance_consistent = True`
- 估计差值非常小，通常在 `0.02 W m^-2` 以内

写作抓手：Figure08 的空间积分结果与 Figure07 区域汇总严格一致，说明空间图与区域 pathway 解释是闭合的。

## Figure09_monthly_diagnostic_representativeness_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure09_monthly_diagnostic_representativeness_degC05/02_plotting_script/make_figure09_monthly_diagnostic_representativeness_degC05.py`

数据口径：散点和拟合均基于全部 `248` 个月；ENSO 稳健性审计用 `54 + 85 = 139` 个 ENSO 子样本月。

全部 9 条关系在全月样本中均显著，核心数字如下：
- WP HCCF -> high-cloud pathway: `R2 = 0.411`, slope `-16.070`
- WP HCTB -> high-cloud pathway: `R2 = 0.637`, slope `-44.144`, 相对 HCCF 的 `delta adjusted R2 = +0.372`
- WP LCSP -> low-cloud pathway: `R2 = 0.912`, slope `+40.364`
- CP HCCF -> high-cloud pathway: `R2 = 0.192`, slope `-6.831`
- CP HCTB -> high-cloud pathway: `R2 = 0.411`, slope `-37.862`, `delta adjusted R2 = +0.536`
- CP LCSP -> low-cloud pathway: `R2 = 0.916`, slope `+30.481`
- EP HCCF -> high-cloud pathway: `R2 = 0.252`, slope `-9.108`
- EP HCTB -> high-cloud pathway: `R2 = 0.223`, slope `-27.414`, `delta adjusted R2 = +0.490`
- EP LCSP -> low-cloud pathway: `R2 = 0.905`, slope `+79.564`

ENSO 子样本稳健性：
- 9 条关系全部 `sign_consistent_allmonth_vs_ENSO_subset = True`

写作抓手：LCSP 对 low-cloud pathway 的代表性最强；HCTB 在三大区域都提供了超出 HCCF 的额外高云结构信息。

## Figure10_heldout_direct_diagnostic_linkage_degC05

绘图程序：`/Volumes/My Book/P3/figure_packages/Figure10_heldout_direct_diagnostic_linkage_degC05/02_plotting_script/make_figure10_heldout_direct_diagnostic_linkage_degC05.py`

数据口径：图中 held-out scatter 继承 all-month blocked-CV 输出；ENSO 稳健性检验采用 `±0.5 C` 定义下的 `20` 个 phase episodes、`139` 个 ENSO 月。

phase-episode held-out pooled skill：
- WP: `R2(M1) = -0.064`, `R2(M3) = 0.614`, `delta R2 = +0.678`; `RMSE` 改善 `-0.735`; `MAE` 改善 `-0.590`
- CP: `R2(M1) = 0.097`, `R2(M3) = 0.554`, `delta R2 = +0.457`; `RMSE` 改善 `-0.398`; `MAE` 改善 `-0.330`
- EP: `R2(M1) = 0.847`, `R2(M3) = 0.870`, `delta R2 = +0.023`; `RMSE` 改善 `-0.081`; `MAE` 改善 `-0.094`

bootstrap 稳健性：
- WP `delta R2` `95% CI = [0.477, 0.878]`
- CP `delta R2` `95% CI = [0.287, 0.610]`
- EP `delta R2` `95% CI = [-0.00012, 0.0517]`，边界性最强

逐 episode 稳定性：
- 三个区域都是 `fraction_events_delta_adjusted_R2_positive = 1.0`
- 三个区域都是 `fraction_events_delta_RMSE_negative = 1.0`
- 三个区域都是 `fraction_events_HCTB_coefficient_sign_consistent = 1.0`

写作抓手：M3 = `LCSP + HCCF + HCTB` 在 WP 和 CP 的 held-out 技能增益最强，EP 仍为正增益但幅度明显更弱。

## 最值得正文优先引用的几组数字

- Figure01/Figure02: CP 与 EP 的 direct daytime `Net CRE` 正异常最强，分别约 `0.9` 和 `1.8 W m^-2`
- Figure03: WP `low cloud` 增加而 `thin high / thick anvil / deep convective` 减少；CP 则相反
- Figure05/Figure06/Figure07: low-cloud pathway 主导区域 `Net`，尤其 CP `+1.83 ~ +1.92 W m^-2`
- Figure07: thick-anvil 组给出最大 SW/LW 振幅，但 Net 主导组不是 thick anvil，而是 low cloud
- Figure09: LCSP 与 low-cloud pathway 的 `R2` 在三大区域均约 `0.90+`
- Figure10: 加入 HCTB 后，held-out `delta R2` 在 WP 和 CP 分别提升约 `+0.68` 与 `+0.46`
