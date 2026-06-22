# Bitcoin Predictability 
How much of daily Bitcoin behaviour is actually predictable from its own price history? This collaborative project answers that question honestly by building models using two distinct feature engineering philosophies, and then stress-testing them hard enough that a weak signal can't masquerade as a strong one.

**TL;DR:** Return *magnitude* is effectively unpredictable (best R² ≈ 0). Next-day *direction* carries a faint, walk-forward-confirmed edge (XGBoost ROC-AUC ≈ 0.55–0.58 on held-out data, ~0.53 averaged across 2019–2024) — but that edge does **not** survive becoming a tradeable strategy (+3% vs +25% buy-and-hold over the test window). The result is a clean demonstration of weak-form market efficiency and of valuing disciplined validation over a vanity metric.

---

## Why this project

It's easy to fit a model to crypto prices and report a number that looks good. It's much harder to show whether that number means anything. As a team, our design philosophy is the opposite of the usual "throw 200 features at a deep net" approach. We explored this through two parallel tracks:

- **Track 1: Parsimony over complexity** — A handcrafted set of 15 economically motivated features, each mapping to a property the data actually exhibits.
- **Track 2: Consensus over intuition** — A statistically rigorous set of 23 features, surviving strict statistical tests and a multi-model voting system to prevent single-algorithm bias.
- **Shared discipline** — Validation over the headline metric (walk-forward out-of-sample checks) and strict leakage discipline (trailing features, stationary inputs, chronological splits).

## Data

Daily OHLCV + market cap for Bitcoin, **2010-07-17 → 2024-05-22** (~5,059 rows). The models use *only* the asset's own market history — no on-chain, macro, or sentiment data — so the question is specifically about self-predictability.

> The notebook expects a CSV at `bitcoin_2010-07-17_2024-05-23.csv` with columns `Start, Open, High, Low, Close, Volume, Market Cap`. Update `CSV_PATH` at the top of the notebook to point to your local copy.

## Method

### 1. Exploratory data analysis
Establishes the statistical character of the series *before* any feature is engineered, so the feature design is motivated rather than arbitrary. Key findings:
- Returns are **stationary** (ADF p ≈ 0) but **near-random** — weak, short-lived autocorrelation.
- **Fat tails** (kurtosis ≈ 17) — extreme days far more common than a normal distribution implies.
- **No** significant weekday or month seasonality (ANOVA p > 0.05).
- Squared returns autocorrelate strongly — **volatility clusters**, even when returns don't.

These point to a clear prior: magnitude ≈ unpredictable, direction possibly faint, volatility the most persistent property.

### 2. Dual Feature Engineering Approaches
To thoroughly explore the predictability of the asset, we engineered two distinct feature sets. 

#### Approach 1: The CORE Feature Set (15 features)
A deliberately small, interpretable set. Every feature is trailing-only (no look-ahead) and price enters as ratios to stay stationary across the 2010→2024 range. Collinear duplicates are dropped (e.g. one stochastic, `rsi_14` over `rsi_30`).

| Family | Features | Rationale |
|---|---|---|
| **Momentum** | `ret_1`, `ret_3`, `ret_7`, `ret_lag1` | short-horizon trend / persistence |
| **Mean-reversion** | `rsi_14`, `stoch_k`, `bb_pctb`, `zscore_30` | strongest single-feature correlations in EDA |
| **Volume / flow** | `mfi_14`, `rel_volume` | money flow + relative participation |
| **Volatility** | `vol_30`, `parkinson` | the one property that clusters |
| **Trend / regime** | `close_ma200`, `bull` | long-run regime context |
| **Interaction** | `ret3_x_vol7` | the only cross-term with real EDA correlation |

*Finding:* Screening confirms no single feature exceeds ~0.07 |corr| and importances are near-uniform — the signal is weak and *aggregate*, proving why parsimony is safe.

#### Approach 2: The Two-Stage Consensus Approach (23 features)
Rather than relying on human intuition, this pipeline subjects raw features to strict statistical tests followed by a multi-model voting system, resulting in a robust, algorithm-agnostic set of 23 core features.

**Phase A: Statistical Filtering**
| Pipeline Phase | Core Mechanism | Rationale |
| :--- | :--- | :--- |
| **1. Redundancy** | Pearson Correlation (<0.90) | Drops highly collinear duplicates. |
| **2. Time-Series** | ADF Test & Cointegration | Ensures time-series integrity and stationarity. |
| **3. Info Gain** | Mutual Information | Screens for standalone, non-linear predictive value. |

**Phase B: Multi-Model Consensus Voting System**
To eliminate the structural biases of any single ML algorithm, we employed a strict democratic voting mechanism:
* **Diverse Evaluation:** Features are tested across 6 distinct ML architectures to prevent single-model bias.
* **Voting Mechanism:** Each model awards exactly one vote to its top 20 most important features.
* **Consensus Threshold:** A candidate feature must secure $\ge$ 3 out of 6 possible votes to survive.
* **Robust Outcome:** This distills the data into 23 core features, guaranteeing a final signal that is resilient, transferable, and highly resistant to overfitting.

### 3. Models and targets
A leakage-safe chronological split (train ≤ 2023-01-31, 1-day embargo, test Feb–Jul 2023) and one consistent scorecard across:

- **Return magnitude** (regression) — LinearRegression, RandomForest, XGBoost
- **Next-day direction** (classification) — XGBoost, CatBoost
- **Forward 30-day volatility** and a **7-day return horizon** (regression)
- **LSTM** baselines (PyTorch) for both next-day regression and direction

### 4. Direction validation battery
The edge is small, so each check asks: *is this real and usable, or noise?*

- **Walk-forward validation** — train on all prior data, test each year 2019–2024 separately.
- **Feature importance** — which families drive the prediction.
- **Confusion matrix** — is the model just predicting UP and tracking market drift?
- **Threshold testing** — does accuracy rise on high-confidence predictions? (informative probabilities)
- **Trading evaluation** — long when p > 0.55 else cash, vs buy-and-hold.
- **7-day horizon** — does signal strengthen further out?
- **Bull/bear regime models** — train and test separately on `Close > MA200` vs not.
- **SHAP explainability** — signed, directional attributions for the direction model.

## Key results

| Target | Outcome |
|---|---|
| Return magnitude (next-day) | R² ≤ 0 — effectively unpredictable |
| Next-day direction | ROC-AUC ≈ 0.55–0.58 held-out; ~0.53 mean across 2019–2024 walk-forward |
| Trading the direction signal | +3% vs +25% buy-and-hold — edge does not survive as a strategy |
| Forward 30-day volatility | strongest R² of any target — volatility persists, returns don't |
| Longer horizons | modest improvement in 30-day directional accuracy — weak medium-term info |

**Conclusion:** Bitcoin returns are difficult to predict, consistent with weak/semi-strong market efficiency. A compact, well-motivated feature set extracts only modest, regime-stable predictive power. Parsimony improves interpretability and reduces overfitting risk. The honest negative result — an edge that's statistically detectable but economically unusable — is the point.

## Repository
