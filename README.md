# Student Health Risk — Kaggle Playground Series S6E7

Competition notebooks for [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7):
predict `health_condition` ∈ {`at-risk`, `unhealthy`, `fit`} for college students, scored by
**balanced accuracy**.

## Notebooks

| Version | File | Result |
|---|---|---|
| **v2 (current)** | [`student-health-risk-v2.ipynb`](student-health-risk-v2.ipynb) | expected CV/LB ≈ 0.950+ |
| v1 | [`student-health-risk-ensemble.ipynb`](student-health-risk-ensemble.ipynb) | **LB 0.94924** (CV 0.94944) |

Upload to Kaggle attached to the competition dataset, select **GPU T4 x2**, internet **off**, and
Run All; the notebook writes `submission.csv`. All libraries (LightGBM, XGBoost, CatBoost, PyTorch)
are pre-installed on the Kaggle image. Every model probes the GPU once and falls back to CPU
automatically. v2 runtime ≈ 2.5–3 h on T4 x2.

Input paths are auto-detected (first match wins):

```
/kaggle/input/competitions/playground-series-s6e7/{train,test,sample_submission}.csv
/kaggle/input/playground-series-s6e7/{train,test,sample_submission}.csv
data/{train,test,sample_submission}.csv        # local
```

## Why this design

The target is heavily imbalanced (`at-risk` ≈ 86%, `unhealthy` ≈ 8%, `fit` ≈ 6%) while balanced
accuracy weights all three classes equally. The v1 run confirmed CV tracks LB within ~2e-4, so all
decisions are made on out-of-fold predictions of real train rows.

1. **Class-balanced training** everywhere (class/sample weights), so minority recall isn't sacrificed.
2. **Per-class multiplier tuning** — balanced accuracy is maximized by `argmax(w ⊙ p)`, not
   `argmax(p)`; `w` is grid-searched (log-spaced coarse → fine) on OOF. Worth ≈ **+0.02** over raw
   argmax: the single biggest lever.
3. **CatBoost anchors the blend** (v2). On the v1 GPU run CatBoost was the best single model
   (tuned OOF 0.94920 vs 0.94854 LightGBM / 0.94754 XGBoost) *and* ~35× faster than LightGBM on GPU
   (101 s vs 3640 s for 5 folds) — v2 averages 3 CatBoost seeds and adds cat-heavy blend candidates.
4. **Hard-rule masks** (v2). The synthetic target is rule-generated with mineable hard bounds —
   train contains zero `fit` above BMI 26.00, zero `unhealthy` below BMI 19.85, none `at-risk` above
   BMI 30.83, no `fit` below 3.66 h sleep, no `unhealthy` above 102.5 bpm. Bounds are re-derived
   from train at runtime and impossible classes zeroed before the tuned argmax.
5. **Pseudo-labeling with auto-fallback** (v2). Confident test predictions (max blended prob ≥ 0.90)
   join each fold's training part for a second round; validation rows stay pure train. Offline
   simulation on a 200k/300k split: +0.0010–0.0014 for single models, ~neutral for the blend — so
   round 2 ships only if it beats round 1 on OOF.
6. **Neural corrector, candidate-gated** (v2). A small embedding MLP may join the blend at a capped
   share (0–15%, selected on OOF). Measured offline: tree-heavy blends win and the NN's best share
   was 0% — the gate keeps it only if full-scale training disagrees.
7. **Missingness features** — per-column NA indicators + row NA count (missingness is informative),
   NaNs kept native for the trees; plus ratio features and `stress×activity`, `stress×sleep_quality`
   crosses (stress and activity dominate the label rules).

## Offline validation (200k-row simulation, tuned balanced accuracy)

| Candidate | Round 1 | Round 2 (+pseudo) |
|---|---|---|
| LightGBM | 0.94676 | 0.94780 |
| XGBoost | 0.94569 | 0.94705 |
| CatBoost (2 seeds) | 0.94791 | 0.94773 |
| Equal blend | 0.94851 | 0.94845 |
| Equal blend + masks + fine tuning | **0.94881** | — |

At full 690k scale (v1 run) the same pipeline pieces scored CV 0.94944 / LB 0.94924; v2's additions
(CatBoost seeds, masks, richer candidate grid, guarded pseudo round) target the 0.952x cluster.
The public-LB ceiling sits at 0.95238 = 20/21 — the Bayes limit of the noisy rule-based target.
