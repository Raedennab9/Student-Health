# Student Health Risk — Kaggle Playground Series S6E7

Competition notebook for [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7):
predict `health_condition` ∈ {`at-risk`, `unhealthy`, `fit`} for college students, scored by
**balanced accuracy**.

## Notebook

[`student-health-risk-ensemble.ipynb`](student-health-risk-ensemble.ipynb) — upload to Kaggle as a
notebook attached to the competition dataset and run all cells; it writes `submission.csv`.

Expected input paths (auto-detected, first match wins):

```
/kaggle/input/competitions/playground-series-s6e7/{train,test,sample_submission}.csv
/kaggle/input/playground-series-s6e7/{train,test,sample_submission}.csv
data/{train,test,sample_submission}.csv        # local
```

## Approach

The target is heavily imbalanced (`at-risk` ≈ 86%, `unhealthy` ≈ 8%, `fit` ≈ 6%) while balanced
accuracy weights all three classes equally, so the pipeline is built around class balance end to end:

1. **Feature engineering** — per-column missing indicators + row missing count (missingness is
   informative in this synthetic data), physiologically sensible ratios/interactions
   (calories-per-step, exercise intensity, activity score, BMI×HR), BMI category bins. NaNs are kept
   native for the tree models.
2. **Models** — LightGBM, XGBoost and CatBoost, each trained class-balanced with 5-fold stratified
   CV and early stopping; out-of-fold probabilities collected for honest tuning.
3. **Blend** — ensemble weights optimized on OOF log-loss (with an equal-weight fallback, whichever
   scores better after step 4).
4. **Decision-rule tuning** — balanced accuracy is maximized by `argmax(w ⊙ p)` rather than
   `argmax(p)`; per-class multipliers `w` are grid-searched (coarse → fine) on OOF predictions only.
   This step is reliably worth several ×0.001 of balanced accuracy over raw argmax.

All tuning happens on out-of-fold predictions, so the reported CV estimate is honest and has tracked
the public leaderboard closely in this competition.
