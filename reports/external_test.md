# External test on a second listings source (model 1.1, before the two sources were merged)

Source: Kaggle 'Mumbai House Price Data (70k entries)' (kevinnadar22). The engine was trained only on the first source. Listings that look like the same flat in both sources were removed. Values are compared at the data's own price level (before the 2026 price update).

| Group | Flats tested | Median error | Within ±10% | Median bias | 80% range covers |
|---|---|---|---|---|---|
| Buildings the engine knows | 2,500 | 16.8% | 32.3% | -9.7% | 60.8% |
| Buildings the engine has never seen | 2,500 | 15.7% | 34.2% | -5.9% | 71.2% |
| All | 5,000 | 16.2% | 33.2% | -7.9% | 66.0% |

## By traffic light

| Light | Flats | Median error | Within ±10% | Median bias | 80% range covers |
|---|---|---|---|---|---|
| Green | 197 | 12.7% | 39.6% | -6.1% | 41.6% |
| Amber | 1,417 | 12.3% | 41.9% | -6.9% | 65.6% |
| Red | 3,386 | 18.2% | 29.2% | -8.8% | 67.5% |

What we learned: the second source asks about 8% more for the same buildings (it is newer), so model 1.1 was 8% low on it, and its ranges covered 66% instead of 80%. Model 2.0 is trained on both sources, with the second scaled to the first one's price level.
