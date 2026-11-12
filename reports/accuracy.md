# Accuracy report

Data: 84,821 cleaned listings, 14,543 projects, 146 suburbs of the Mumbai Metropolitan Region.

Method: 5-fold cross-validation grouped by project, so every test flat belongs to a project the model never saw in training. Settings were chosen on half the projects (calibration) and all numbers below are on the other half (test).

## Engine vs simple baseline (test projects)

| Method | Median error | Within ±10% | Within ±20% |
|---|---|---|---|
| Baseline: suburb median rate × area | 17.3% | 30.8% | 56.4% |
| ML model only | 11.9% | 43.3% | 71.5% |
| ML model + same-project comparables (full engine) | 10.4% | 48.7% | 75.0% |

## Traffic light: does confidence mean anything? (test projects)

| Light | Share of flats | Median error | Within ±10% | Within ±20% | 80% range actually covers |
|---|---|---|---|---|---|
| GREEN | 9.4% | 5.3% | 73.0% | 92.3% | 75.3% |
| AMBER | 40.7% | 8.8% | 55.3% | 82.2% | 80.2% |
| RED | 49.9% | 13.7% | 38.7% | 65.8% | 80.2% |

## By evidence available (test projects)

| Situation | Median error | Within ±10% | Typical 80% range (±) | Range actually covers |
|---|---|---|---|---|
| 3+ other listings in same project | 9.8% | 50.5% | 24.2% | 80.4% |
| New / unknown project | 13.1% | 40.6% | 28.2% | 76.9% |

## What the location data adds (ablation, ML model only)

| Model | Median error | Within ±10% |
|---|---|---|
| all features (same settings, for comparison) | 11.9% | 43.1% |
| without location & distance features | 32.3% | 17.7% |
| without rail-station features | 12.0% | 43.0% |
| without building map position (suburb centre only) | 12.1% | 42.6% |
| without building age, bathrooms, furnishing | 12.0% | 42.9% |
| without distance-to-hub features | 12.0% | 43.2% |

## By data source (test projects, full engine)

| Source | Flats | Median error | Within ±10% | 80% range covers |
|---|---|---|---|---|
| Source 1: Mumbai House Prices (76k) | 26,104 | 10.2% | 49.1% | 80.6% |
| Source 2: Mumbai House Price Data (70k) | 15,144 | 10.7% | 47.9% | 78.1% |

Source 2 prices were 1.086× source 1 for the same project and BHK, so they were scaled to source 1's level before training. 76.6% of listings have a building-level map position.

Caveat: these are listing (asking) prices from about 2022, not registered sale prices. The same test method should be rerun on registered transactions (IGR) and the lender's own valuation reports before production use.
