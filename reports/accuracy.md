# Accuracy report

Data: 54,465 cleaned listings, 9,717 projects, 146 suburbs of the Mumbai Metropolitan Region.

Method: 5-fold cross-validation grouped by project, so every test flat belongs to a project the model never saw in training. Settings were chosen on half the projects (calibration) and all numbers below are on the other half (test).

## Engine vs simple baseline (test projects)

| Method | Median error | Within ±10% | Within ±20% |
|---|---|---|---|
| Baseline: suburb median rate × area | 16.1% | 33.2% | 58.8% |
| ML model only | 11.9% | 43.5% | 71.4% |
| ML model + same-project comparables (full engine) | 10.3% | 49.1% | 74.0% |

## Traffic light: does confidence mean anything? (test projects)

| Light | Share of flats | Median error | Within ±10% | Within ±20% | 80% range actually covers |
|---|---|---|---|---|---|
| GREEN | 8.7% | 4.6% | 77.6% | 92.6% | 79.4% |
| AMBER | 43.8% | 8.5% | 55.9% | 81.6% | 79.5% |
| RED | 47.5% | 14.5% | 37.5% | 63.5% | 79.4% |

## By evidence available (test projects)

| Situation | Median error | Within ±10% | Typical 80% range (±) | Range actually covers |
|---|---|---|---|---|
| 3+ other listings in same project | 9.6% | 51.2% | 24.1% | 80.5% |
| New / unknown project | 13.4% | 39.8% | 26.5% | 74.7% |

## What the location data adds (ablation, ML model only)

| Model | Median error | Within ±10% |
|---|---|---|
| Full model | 11.9% | 43.5% |
| without location & distance features | 30.9% | 19.2% |
| without distance-to-hub features | 11.8% | 43.8% |

Caveat: these are listing (asking) prices from about 2022, not registered sale prices. The same test method should be rerun on registered transactions (IGR) and the lender's own valuation reports before production use.
