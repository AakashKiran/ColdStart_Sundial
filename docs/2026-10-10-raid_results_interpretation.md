# RAID on Intel SKU Cold-Start Forecasting: Results and Interpretation

**Date:** 2026-10-11  
**Experiment:** Aggregate Intel SKU demand, RAID architecture notebook  
**Purpose:** Explain the current results, separate dataset limitations from implementation issues, and record what can and cannot be concluded.

## 1. Experimental setting

The Intel dataset contains 86 SKU demand series. The current split uses **51 training SKUs** (all 27 SKUs ineligible for launch-time evaluation plus 24 eligible SKUs) and **35 held-out test SKUs**. The forecast window is 13 weeks from each target SKU's launch. The target is aggregate weekly `Customer Orders` across distribution centers. Retrieval metadata are Product Offering, Generation, and ASP Group.

The target SKU's own history is unavailable at launch. Donors must have launched strictly before the target and must have usable pre-origin history.

This is a **SKU-held-out cold-start simulation, not strict historical backtesting**: training labels for some training SKUs can occur later in calendar time than a test SKU's launch. The test SKU's labels are held out, and donor histories are cut off before its launch.

The evaluation contains 35 SKUs × 13 weeks = 455 forecast points. These are repeated observations within 35 products, not 455 independent launches.

## 2. Observed results

The notebook reported:

| Metric | Result |
|---|---:|
| MAE | 29,334.79 |
| RMSE | 34,072.77 |
| WAPE | 170.40% |
| sMAPE | 118.16% |
| Test SKUs | 35 |
| Forecast window | 13 weeks |

For the inspected test case **SKU-A-5** (launch origin week 158):

| Quantity | Result |
|---|---:|
| Usable donors | 14 |
| Actual mean over 13 weeks | 19,371 |
| Base forecast mean | 22,557 |
| Final forecast mean | 22,502 |
| Actual maximum | 115,334 |
| Base forecast range | 10,285–34,248 |
| Final forecast range | 10,693–34,351 |
| Gate sigmoid | 0.0541 |

The forecast missed a large spike: actual demand reached 115,334, while the forecast remained below about 34,400. This is an illustrative case, not proof that every SKU behaves the same way.

The earlier script produced predictions clustered around approximately 20 units. The notebook implementation now predicts on a plausible demand scale, but overall MAE, RMSE, and WAPE are worse. Correct numerical scale is an implementation sanity check, not evidence of forecasting quality.

## 3. Main limitation: only a small, narrow training collection is available

The RAID paper is designed for a much broader collection of products and demand trajectories. Here, training is restricted to **51 SKU series**. These cannot represent the full range of product life cycles, demand magnitudes, launch patterns, and occasional large spikes that may occur among held-out SKUs.

Notebook training-only statistics:

- Mean demand: approximately **45,238**
- Median demand: approximately **22,994**
- Maximum demand: approximately **664,203**
- Global standard-deviation scale used for demand normalization: approximately **61,291**

The training mean being different from the test mean is not, by itself, evidence of a coding error. With so few training SKUs, the two partitions can contain different product mixes and demand scales. A model trained on this limited collection cannot be expected to learn the broad cross-product variability available in a large, diverse dataset.

**Faculty interpretation:** the train–test mean difference is expected under this dataset and split because the model sees only 51 training series, rather than the large and diverse collection assumed by the paper. This is a data-availability limitation and cannot be eliminated simply by changing the architecture.

This does not mean every related issue is unavoidable. The limited number and diversity of SKUs are inherent to the dataset; some avoidable train/test mismatch can still be reduced by making training episodes more launch-aligned. That adjustment cannot, however, manufacture missing product diversity.

## 4. Why launch-time forecasts can be too high

### 4.1 Training episodes include mature stages

The notebook creates training episodes at multiple origins across training SKUs' observed life cycles. Many episodes can therefore represent established or mature demand rather than only the first weeks after launch.

At test time, every forecast starts at the target SKU's launch. Early windows may contain low demand, zeros, or a ramp-up. The training distribution can differ from the test distribution even though both come from the same dataset.

This is a **training/evaluation protocol mismatch** layered on top of the small-data limitation. It can be investigated by restricting training origins to early life-cycle weeks or conditioning explicitly on weeks since launch. Those should be labelled as separate variants.

### 4.2 Donor history may not describe a launch curve

The current donor representation uses the 13 weeks immediately before the target's calendar-time origin. For a target launching at week t, that is approximately weeks t−13 through t−1, provided the donor has a complete history in that window.

Those observations describe the donor's stage at that calendar time, not necessarily its first 13 weeks after launch. If the donor is mature, its normalized shape may carry little information about how a newly launched SKU ramps up.

Using each donor's first 13 weeks after its own launch could be a useful **launch-aligned donor variant**, but it changes the current design and should be evaluated as a separate experiment.

### 4.3 Large spikes are difficult to infer from metadata alone

In the SKU-A-5 example, actual demand reaches 115,334 but the forecast stays below 35,000. Metadata similarity and a small donor set do not necessarily reveal which new SKU will have a much larger scale or a sharp spike.

This is consistent with limited training diversity and regression toward more typical demand levels. The example alone does not identify one specific architectural bug.

## 5. The diffusion branch currently contributes very little

The gate sigmoid in the inspected example is **0.054**, close to the value implied by its initial bias of −3:

\[
\sigma(-3) \approx 0.047.
\]

Under the current notebook objective, diffusion loss trains the diffusion/context path and base loss trains the base forecasting path. The gate does not receive a direct learning signal from a final-forecast reconstruction loss. It therefore remains near its initial low value and strongly suppresses the diffusion correction.

For SKU-A-5, the final forecast differs only modestly from the base forecast. The current output is effectively **base forecast plus a small gated correction**; this run does not demonstrate that diffusion improves accuracy.

This is an implementation/objective limitation of the current experiment. Because some details in the paper's displayed objective and fast sampling description are underspecified, it should be described as a limitation of this paper-guided implementation, not as a proven flaw in the original RAID method.

## 6. Interpreting the metrics

- **MAE = 29,334.79:** average absolute error is about 29.3k demand units per SKU-week.
- **RMSE = 34,072.77:** large misses receive greater weight, so spikes that are substantially underforecast can increase RMSE.
- **WAPE = 170.40%:** the sum of absolute errors exceeds the sum of actual demand across evaluated observations. This is a strong warning that aggregate accuracy is poor in the current run.
- **sMAPE = 118.16%:** intermittent demand makes this metric tricky. Under the usual formula, predicting a positive value when actual demand is zero contributes 2 (or 200% when expressed as a percentage) for that observation. Zeros and near-zeros can therefore inflate sMAPE and make it behave differently from MAE or WAPE.

The earlier script's sMAPE was approximately 195.72%, while the notebook's was approximately 118.16%. That improvement does not outweigh the deterioration in MAE, RMSE, and WAPE; the metrics measure different error characteristics. Baselines are required before claiming that any score is competitive.

## 7. What this run establishes—and what it does not

### Supported by this run

- The notebook prepares pseudo-cold-start episodes and trains the implemented model on the available Intel data.
- Demand scaling avoids the extreme numerical-scale issue seen in the earlier script run.
- The model produces forecasts on a plausible demand scale.
- In the inspected case, the gate is small and the final forecast is close to the base forecast.
- Aggregate errors remain high, and the inspected launch contains a large spike that the forecast misses.

### Not established by this run

- That RAID beats simple launch-curve baselines.
- That diffusion improves accuracy.
- That GAT-based retrieval beats a simpler metadata-neighbor method.
- That the results reproduce the paper's reported performance.
- That the model generalizes reliably beyond this split and seed.
- That all errors are caused by one factor, whether architecture or data.

## 8. Recommended diagnostic experiments

1. **Add launch-aligned baselines:** global mean launch curve; category/group mean launch curve where enough training examples exist; and k-nearest-neighbor launch analog using metadata embeddings.
2. **Compare base-only and current RAID output** on the same 35 test SKUs. Report MAE, RMSE, WAPE, sMAPE, and median per-SKU error.
3. **Test launch-aligned training episodes:** use only early post-launch origins, or condition on weeks since launch. Label these as variants.
4. **Test launch-aligned donor histories:** compare donors' first 13 weeks after their own launches with the current calendar-window design.
5. **Run a labelled gate ablation:** compare the current objective with a variant that provides a clearly specified learning signal for the gate. Do not label the variant as the exact original method.
6. **Stabilize checkpoint selection:** lowest stochastic training loss can be noisy. Use a fixed validation subset of training SKUs with fixed evaluation seeds, or predeclare a fixed epoch count. Do not tune against the 35 test SKUs.
7. **Repeat across seeds and report uncertainty:** report per-SKU errors and, where feasible, bootstrap confidence intervals over SKUs.
8. **Evaluate probabilistic forecasts only after validating sampling:** then report interval coverage and CRPS alongside point metrics.

## 9. Overall conclusion

The current experiment is best treated as a **first paper-guided implementation and diagnostic baseline**, not a successful reproduction of RAID's reported performance.

The central limitation is that this Intel benchmark supplies only 86 product series, with 51 used for training under the chosen split. That is a narrow basis for learning the variation in product life cycles and demand scales that a broad product dataset can provide. A train–test mean difference is plausible and expected; it should not automatically be diagnosed as a code bug.

A second issue is that training episodes and donor histories can represent mature demand while test windows begin at launch. The diffusion gate is also nearly closed under the current objective, so the final output has not demonstrated a meaningful diffusion benefit. These issues should be investigated separately from the unavoidable scarcity of training products.

**Conclusion supported by current evidence:** the implementation runs and produces scaled forecasts, but current accuracy is weak and the diffusion contribution is not demonstrated. Baselines and controlled ablations are required before making comparative claims.
