# PropVal AI – Digital Property Valuation Engine

Prototype for the **Hero Housing Fintech Challenge, Problem Statement 1** (AI-Powered Digital Property Valuation Engine for India).

Enter a flat's details. The app returns:
- an estimated market value with an **80% price range**
- a **confidence traffic light**: GREEN means desktop valuation with no site visit, AMBER means one site visit, RED means full physical valuation
- the **main factors** behind the value and the **comparable listings** used
- **lending support**: the collateral value to use, the maximum loan under RBI loan-to-value caps, RBI valuation-report rules, and a flag if a valuer's figure looks inflated
- a downloadable **valuation report**

Everything is open source and free to run.

## Files

| File | What it does |
|---|---|
| `app.py` | The web app (Streamlit) |
| `engine.py` | Cleaning, features, valuation, confidence and lending logic |
| `train.py` | Trains and tests the model, writes `artifacts/` and `reports/accuracy.md` |
| `data/Mumbai_House_Prices.csv` | Listing data (Kaggle "Mumbai House Prices", c. 2022) |
| `data/regions.csv` | Approximate coordinates and zone for each suburb |
| `artifacts/` | Trained models and comparables database, already built |
| `reports/accuracy.md` | Accuracy test results |

## Run it on your computer

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
python train.py               # optional: retrain (about 1 minute)
```

## Put it online for free (GitHub + Streamlit Community Cloud)

1. Create a new **public** repository on github.com, for example `propval-ai`.
2. Upload all the files and folders from this folder: on the repository page, choose **Add file → Upload files** and drag them in. Keep the folder structure (`data/`, `artifacts/`, `reports/`).
3. Go to **share.streamlit.io** and sign in with GitHub.
4. Click **Create app → Deploy a public app from GitHub**. Pick your repository, branch `main`, and main file `app.py`.
5. Click **Deploy**. After a few minutes you get a link like `https://propval-ai.streamlit.app` to share with the judges.

The free tier puts an app to sleep when nobody has used it for a while, so open the link yourself shortly before judging.

Hugging Face Spaces also works: create a Space with the Docker → Streamlit template and upload the same files.

## How accurate is it?

See `reports/accuracy.md`. The test is strict: flats are scored on projects the model never saw during training.

## Data sources (cite these in the submission)

- Mumbai House Prices dataset, Kaggle (listing prices, c. 2022).
- Suburb coordinates: approximate centroids. To refine them, look each suburb up on OpenStreetMap (openstreetmap.org) and update `data/regions.csv`, then run `python train.py` again.
- Suburban railway stations: DataMeet Indian Railways open data (github.com/datameet/railways), 112 stations in the Mumbai region, saved in `data/rail_stations.csv`.
- RBI All-India House Price Index, Q4 2025-26 press release (29 May 2026).
- RBI Master Direction for Housing Finance Companies, 2021 (Annex XIV, valuation of properties) and RBI loan-to-value caps for housing loans.

## Settings you can change (in `train.py`)

- `TIME_ADJUSTMENT`: the listings are from about 2022. It is set to 1.159, from the RBI All-India House Price Index (2022-23 = 100, Q4 2025-26 = 115.9). For more precision, use the Mumbai and Thane city indices from RBI's DBIE portal (data.rbi.org.in).
- `GREEN_MAX`, `AMBER_MAX`: range widths for each traffic-light colour. A lender's board would set these.

## Limitations (say these honestly to the judges)

- Asking prices, not registered sale prices.
- Area basis: the dataset doesn't say whether areas are carpet or super built-up. New and resale listings in the same project quote the same areas, and a typical 1BHK is ~610 sq ft, so listings are treated as super built-up. The app converts whatever the user enters (carpet, built-up or super built-up) using typical Mumbai ratios of 1 : 1.20 : 1.35.
- Suburb-level location, not exact addresses.
- Mumbai region only.

The roadmap: replace listings with registered transaction data (Maharashtra IGR) and the lender's own valuation reports; add satellite, flood-risk and metro/infrastructure features; extend city by city using the same pipeline.
