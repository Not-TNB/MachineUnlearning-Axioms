# MachineUnlearning-Axioms — Project Guide

## Context
Premier League match prediction for a **betting competition** (10k virtual GBP).
Dataset: `epl_final.csv` — 9,380 matches, 2000/01–2024/25, 22 columns.
Goal: predict match outcome (H/D/A), then simulate a betting strategy on the holdout set.

## Golden Rules

### Data leakage — the most important thing
**NEVER** use in-game stats as input features. These columns are off-limits as features:
- `HomeShots`, `AwayShots`, `HomeShotsOnTarget`, `AwayShotsOnTarget`
- `HomeCorners`, `AwayCorners`
- `HomeFouls`, `AwayFouls`
- `HomeYellowCards`, `AwayYellowCards`, `HomeRedCards`, `AwayRedCards`
- `HalfTimeHomeGoals`, `HalfTimeAwayGoals`, `HalfTimeResult`
- `FullTimeHomeGoals`, `FullTimeAwayGoals` (these ARE the target)

You can use rolling averages of these stats computed from **past matches only**.

### Temporal split — never shuffle
Always split by date:
- Train: 2000/01 – 2017/18
- Validation: 2018/19 – 2021/22
- Test (holdout): 2022/23 – 2024/25

When computing rolling features, **sort by date first** and only look backwards.

## File Structure
All modelling work goes in `.ipynb` notebooks. One notebook per approach/experiment.

```
epl_final.csv                        # raw data, never modify
data_cleaning.ipynb                  # existing cleaning
NeuralNetwork/
    01_feature_engineering.ipynb     # rolling stats, ELO, encoding
    02_baseline_mlp.ipynb            # simple dense network
    03_deeper_mlp.ipynb              # more layers / different widths
    04_regularisation_study.ipynb    # dropout vs L2 vs batch norm
    05_embeddings.ipynb              # team embeddings approach
    06_elo_features.ipynb            # ELO rating as a feature
    07_betting_simulation.ipynb      # ROI / Kelly criterion / flat stake
LogisticRegression/
    logistic_regression.ipynb
RandomForest/
    (TBD)
```

## We Get Marked on Approaches — Try Everything

The whole point is to document and compare multiple strategies. For each experiment, record:
- What changed vs. the previous approach
- Train/val accuracy and loss
- Test set ROI from betting simulation
- What worked, what didn't, why

### Architectures to try (neural network)
1. **Shallow MLP** — 1 hidden layer (64 units)
2. **Deep MLP** — 4+ layers (256 → 128 → 64 → 32)
3. **Wide MLP** — 2 layers but very wide (512 → 256)
4. **Bottleneck** — narrow middle layer (128 → 32 → 128 → 3)
5. **Team embeddings** — embed team IDs, concatenate with stats features
6. **Residual connections** — skip connections between dense blocks

### Regularisation strategies to compare
- Dropout only (rates: 0.2, 0.3, 0.5)
- L2 weight decay only
- Batch normalisation only
- Combinations of the above
- Early stopping patience values

### Feature sets to compare
- Minimal: last-5-match win rate + home/away form only
- Medium: rolling goals, shots, points over 5 and 10 matches
- Full: all of the above + head-to-head + ELO ratings

### Betting strategies to compare
- Flat stake (1% of bank per bet)
- Bet only when max predicted probability > threshold (0.5, 0.55, 0.6)
- Kelly criterion staking
- Home-win-only strategy as baseline
- Never bet on draws (draws are historically hardest to predict)

## Tech Stack
- **TensorFlow/Keras** for all neural network models
- **pandas / numpy** for feature engineering
- **scikit-learn** for preprocessing (StandardScaler, label encoding)
- **matplotlib / seaborn** for plots inside notebooks
- Work exclusively in `.ipynb` files inside the relevant folder

## Feature Engineering Reference

Rolling stats (computed per team, looking backwards only):
- `home_win_rate_5`, `home_win_rate_10` — win rate last 5/10 matches
- `home_goals_scored_avg_5` — goals scored per game, last 5
- `home_goals_conceded_avg_5` — goals conceded per game, last 5
- Same set for away team
- `home_home_win_rate_5` — home team's win rate in home games specifically
- `away_away_win_rate_5` — away team's win rate in away games specifically
- `h2h_home_wins`, `h2h_draws`, `h2h_away_wins` — head-to-head last 5 meetings
- `home_elo`, `away_elo` — ELO ratings at time of match
- `elo_diff` — home ELO minus away ELO
