# Will it be a match? Predicting mutual interest after a 4-minute speed date

Machine Learning course project by **Kacper Kwak**.

> **Quick preview without running code:** The complete, rendered report (in Polish) is available for direct download in the **[Releases](../../releases)** section on the right (`Speed_Dating_ML.html`). You can download it and open it in any web browser—there is no need to clone the repository, download the dataset, or re-run the Quarto build to view all analyses, interactive plots, and results.

## Overview & Dataset

This project investigates what actually drives mutual romantic interest ("a match") during short, 4-minute speed-dating interactions, comparing what people *claim* they look for in a partner against what governs their actual decisions.

The dataset originates from the classic **Columbia Business School Speed Dating Experiment** conducted by Raymond Fisman and Sheena Iyengar between 2002 and 2004:
- **Scale & Structure:** 8,378 individual date evaluations collected across 21 distinct speed-dating sessions (`wave`), involving 551 unique participants. Because each interaction is recorded from both perspectives, the raw records are restructured into **4,189 unique date pairs** (one row per date combining features of both the woman and the man).
- **Target Variable:** Mutual match (`match = 1`), occurring when both participants independently indicate "yes" (`dec = 1` and `dec_o = 1`) to seeing each other again. Matches represent **16.5%** of all pairs (690 matches out of 4,189 dates), creating a notable class imbalance.
- **Collected Information:**
  - *Pre-date baseline:* Demographics (age, field of study, race, career goals), lifestyle and dating habits, self-reported interest ratings across 17 activities (e.g., sports, art, movies, clubbing), self-perceived attribute scores, and stated partner preferences (100-point allocation across 6 traits: attractiveness, sincerity, intelligence, fun, ambition, and shared interests).
  - *Engineered pair-level features:* Age differences, shared field of study or race, cosine similarity and correlation of stated interests, and alignment between one person's self-ratings and the partner's stated preferences.
  - *Post-date evaluations:* Immediate post-interaction ratings given by each partner on the 6 core traits, overall liking (`like`), and perceived probability of reciprocation (`prob`).

## Key Results

| Scenario | Features | Best model (chosen by CV) | Test ROC AUC | Test PR AUC |
|---|---|---|---|---|
| A | pre-date only (demographics, stated preferences, interests, self-ratings, pair features) | Logistic regression (L1) | ≈ 0.60 | ≈ 0.24 |
| B | A + ratings given after the date | XGBoost | ≈ 0.83 | ≈ 0.54 |

Random baseline: ROC AUC = 0.5, PR AUC ≈ 0.16 (share of matches).

- **Pre-date questionnaires barely predict a match.** They tell the model who is generally popular or easy-going, but almost nothing about whether two specific people click.
- **Post-date impressions carry the signal.** Overall liking, attractiveness, fun, shared interests and the perceived chance of a "yes" from the partner matter most. A low rating from either side effectively vetoes a match.
- **Stated preferences ≠ revealed preferences.** Women say intelligence matters most, men say attractiveness matters a lot. In actual decisions, attractiveness is 1.5–2.5× more important than stated for both genders, while intelligence and sincerity hardly matter. People who claim a trait is important do not react to it more strongly than others.
- **Gender differences:** attractiveness weighs significantly more for men and intelligence slightly more for women. Women are more selective (36.5% vs 47.4% "yes").
- **Methodology matters:** a random train/test split inflates XGBoost's pre-date ROC AUC from 0.60 to 0.74, because the same people end up in both sets. All validation is therefore grouped by session (`wave`).

## Methods

- Pair-level dataset (one row per date, features of both people), removal of leaking variables (`dec`, `dec_o`, post-event surveys, IDs)
- Train/test split and 5-fold CV grouped by session (`StratifiedGroupKFold`), so no person appears in both training and evaluation data
- Models: logistic regression, decision tree, linear SVM, RBF SVM, random forest, XGBoost, tuned with `RandomizedSearchCV` (PR AUC objective)
- Class imbalance: class weights vs SMOTE vs undersampling (inside folds only, `imblearn.Pipeline`), plus F1-optimal threshold chosen on out-of-fold predictions
- Bootstrap confidence intervals for test metrics
- Interpretability: SHAP (global, grouped, waterfall), 1D/2D partial dependence, ceteris paribus profiles, and a supplementary logistic model of individual decisions with cluster-robust SEs

## Data & Reproducibility

Speed Dating Experiment (Fisman & Iyengar, Columbia Business School, 2002–2004), available on Kaggle:
<https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment>.

If you want to reproduce the analysis from source (`projekt.qmd`), download `Speed Dating Data.csv` and place it in `data/` (the raw dataset is not included in the repository).

```bash
pip install -r requirements.txt
quarto render projekt.qmd      # ~15 min on 2 CPU cores (hyperparameter search)
```

Requires [Quarto](https://quarto.org) ≥ 1.4 and Python ≥ 3.10. All random seeds are fixed (`random_state = 42`).
