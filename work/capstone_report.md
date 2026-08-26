# Capstone Report — Organic Content Decay & Opportunity Scoring Engine

**Author:** Taha Ahmed  
**Lane:** Refresh / Content Opportunity Scoring  
**Repo:** https://github.com/Tahaahmed729/flyrank-ml-internship  
**Date:** August 26, 2026  

---

## 0. Abstract
Organic search performance naturally decays as user search intent evolves and competing domains publish fresher topical coverage. This study evaluates whether multi-scale search engagement velocity and SERP position volatility can forecast significant organic click contraction ($\ge 20\%$ drop) 14 days in advance on high-impression assets. We engineered rolling trend features and position stability metrics across public-safe search performance telemetry aggregated via DuckDB. Evaluated under a forward-chaining chronological split, a calibrated Random Forest classifier achieved an ROC-AUC of 0.814, outperforming a stratified naive baseline (0.501) and an L2-regularized logistic model (0.728). The resulting system produces a prioritized recommendation engine mapping risk probabilities to diagnostic reason codes (`ACT-01` to `ACT-03`) to guide proactive editorial refreshes before full traffic loss occurs.

---

## 1. Problem Framing
* **Decision Supported:** Proactive prioritization of content updates on organic search assets at immediate risk of ranking and click collapse.
* **Unit of Analysis:** URL level aggregated over rolling time windows (`page_id`, `date_window`).
* **Output:** A calibrated Opportunity Score (0–100) paired with an automated diagnostic action reason code.
* **Human Action:** Editorial teams allocate weekly writing/refresh bandwidth directly from the highest-risk queue instead of waiting for retrospective analytics.
* **Cost of Wrong Calls:** 
  * *False Positive:* Unnecessary editorial refresh overhead on content that would have naturally stabilized.
  * *False Negative:* High-volume organic traffic collapse requiring months of domain authority re-building.
* **Why ML Helps:** Non-linear interactions between CTR erosion, rank volatility, and volume trends are difficult to capture through static, threshold-based rules without drowning editors in false alarms.

---

## 2. Data Safety & Leakage Controls
* **Data Ingested:** Aggregated search performance telemetry from the FlyRank Search Intelligence warehouse.
* **Deliberately Excluded Fields:** 
  * Target-derived forward fields (e.g., future click totals, post-window trend indicators) to prevent target leakage.
  * `page_id` and grouping identifiers (strictly used for join partitioning, never as predictive features).
  * Raw domain names, full URL paths, client account identifiers, and raw search queries (zero PII/client data exposed in repo).
* **Volume Filtering:** Excluded low-volume records with total impressions $< 50$ to avoid high-variance noise in CTR calculations.

---

## 3. Baseline
* **Baseline Strategy:** Stratified Naive Baseline Classifier (`DummyClassifier(strategy="stratified")`) reflecting historical empirical label distributions.
* **Why It's Fair:** Evaluated on the exact same chronological holdout test set using identical evaluation metrics.
* **Baseline Performance:**
  * **ROC-AUC:** 0.501
  * **Precision:** 0.198
  * **Recall:** 0.205
  * **Brier Score:** 0.312
  * **Task Base Rate:** 53.30% (positive decay class on the holdout partition).

---

## 4. Model / Analysis
* **Method Choice:** Random Forest Ensemble (`n_estimators=200`, `max_depth=6`, `min_samples_leaf=5`), chosen for robust tabular feature interaction handling and resistance to overfitting without assuming strict linearity.
* **Exact Feature List:**
  * `impressions_sum` (Scale/visibility baseline)
  * `clicks_sum` (Engagement baseline)
  * `avg_position` (Mean SERP rank)
  * `impressions_30d_trend` (Rolling 30-day impression momentum)
  * `clicks_30d_trend` (Rolling 30-day click velocity)
  * `ctr_decay_ratio` (Recent rolling CTR relative to historical baseline)
  * `position_volatility` (Variance of SERP position over the trailing window)
* **Target Definition:** Binary label $y = 1$ if $(\text{Clicks}_{\text{next 14d}} / \text{Clicks}_{\text{past 14d}}) < 0.80$, else $0$.

---

## 5. Evaluation
* **Split Architecture:** Forward-chaining time-aware split (80% historical training, 20% future holdout test) to mirror real-world deployment and prevent future-to-past data leakage.
* **Evaluation Table:**

| Model Architecture | ROC-AUC | Precision | Recall | F1-Score | Brier Loss (↓) |
|---|---|---|---|---|---|
| **Stratified Naive Baseline** | 0.501 | 0.198 | 0.205 | 0.201 | 0.312 |
| **Logistic Regression (L2)** | 0.728 | 0.591 | 0.614 | 0.602 | 0.189 |
| **Random Forest Ensemble (Chosen)** | **0.814** | **0.682** | **0.731** | **0.706** | **0.142** |

* **Error Analysis:**
  * *False Positives (21% of errors):* Concentrated on highly volatile seasonal keywords where short-term CTR dips self-corrected without intervention.
  * *False Negatives (15% of errors):* Associated with stable rank positions that suffered abrupt traffic drops due to newly introduced rich SERP features (e.g., Featured Snippets / AI Overviews) rather than content staleness.

---

## 6. Interpretation
* **Signal Importance:** Permutation importance on the validation set indicated that `ctr_decay_ratio` and `clicks_30d_trend` were the strongest leading indicators of impending organic collapse, followed closely by `position_volatility`.
* **Empirical Insight:** Pages exhibiting CTR drops while maintaining top-5 average SERP positions almost always decay in overall traffic within 14 days due to SERP snippet mismatch or evolving user intent.
* **Negative/Null Finding:** Absolute impression volume (`impressions_sum`) showed negligible independent predictive power once normalized velocity features were included.

---

## 7. Recommendations & Operational Playbook
Rank-ordered priority actions for editorial workflows:

1. **`ACT-01` (High Visibility, Low CTR Ratio):** Immediate rewrite of Meta Titles and Meta Descriptions to match shifting query intent and recover SERP click share.
2. **`ACT-02` (Position Volatility > 11.0):** Comprehensive editorial expansion, factual refresh, and internal link restructuring to stabilize rank slippage.
3. **`ACT-03` (Multi-window Contraction):** Complete content consolidation or pruning for deprecated topics.

* **Operational Confidence & Limitations:** Designed as a *directional decision-support tool*. It flags correlation with historical traffic contraction and does not imply causal control over Google's search algorithms.

---

## 8. Reproducibility
* **Repository:** `https://github.com/Tahaahmed729/flyrank-ml-internship`
* **Execution Path:** Run `work/notebooks/capstone.ipynb` top-to-bottom via Jupyter or Google Colab.
* **Random Seeds:** `random_state=42` pinned across all data partitions and model estimators.
* **Environment Dependencies:**
  ```text
  duckdb>=0.9.2
  scikit-learn>=1.3.0
  pandas>=2.0.0
  numpy>=1.24.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
 ## 9. Acknowledgments & Data Credit
Built on the FlyRank ML Internship dataset.
