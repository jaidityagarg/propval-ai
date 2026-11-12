# Accuracy report

Data: 54,465 cleaned listings, 9,717 projects, 146 suburbs of the Mumbai Metropolitan Region.

Method: 5-fold cross-validation grouped by project, so every test flat belongs to a project the model never saw in training. Settings were chosen on half the projects (calibration) and all numbers below are on the other half (test).

## Engine vs simple baseline (test projects)

| Method | Median error | Within ±10% | Within ±20% |
|---|---|---|---|
| Baseline: suburb median rate × area | 16.1% | 33.2% | 58.8% |
| ML model only | 11.9% | 43.5% | 71.3% |
| ML model + same-project comparables (full engine) | 10.3% | 49.2% | 74.1% |

## Traffic light: does confidence mean anything? (test projects)

| Light | Share of flats | Median error | Within ±10% | Within ±20% | 80% range actually covers |
|---|---|---|---|---|---|
| GREEN | 10.3% | 4.8% | 76.9% | 92.7% | 78.2% |
| AMBER | 38.5% | 8.5% | 56.0% | 81.5% | 80.0% |
| RED | 51.2% | 14.0% | 38.5% | 64.7% | 79.4% |

## By evidence available (test projects)

| Situation | Median error | Within ±10% | Typical 80% range (±) | Range actually covers |
|---|---|---|---|---|
| 3+ other listings in same project | 9.6% | 51.2% | 24.7% | 80.5% |
| New / unknown project | 13.2% | 40.4% | 27.2% | 75.3% |

## What the location data adds (ablation, ML model only)

| Model | Median error | Within ±10% |
|---|---|---|
| Full model | 11.9% | 43.5% |
| without location & distance features | 30.9% | 19.2% |
| without rail-station features | 11.8% | 43.7% |
| without distance-to-hub features | 11.9% | 43.8% |

Caveat: these are listing (asking) prices from about 2022, not registered sale prices. The same test method should be rerun on registered transactions (IGR) and the lender's own valuation reports before production use.
