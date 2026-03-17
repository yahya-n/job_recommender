 

## Project Structure

```
job_recommender/
├── app.py            # Flask routes
├── recommender.py    # ML engine (TF-IDF + Cosine Similarity)
├── jobs.csv          # Dataset (32 jobs, 5 categories, 208 skills)
├── requirements.txt
└── templates/
    └── index.html    # Full UI (dark theme, animated cards)
```







# JobMatch AI — HuggingFace-Powered Job Recommendation System

Recommends jobs using **TF-IDF + Cosine Similarity** on real job data loaded live
from the HuggingFace dataset API.

## Dataset
`batuhanmtl/job-skill-set` — 1,170 real job postings with skill sets  
https://huggingface.co/datasets/batuhanmtl/job-skill-set

## How the Data Flows

```
Browser                          Flask Server
  │                                  │
  ├─ GET HF parquet API ──► HuggingFace CDN (direct)
  │         ◄── parquet binary ──────┤
  │                                  │
  ├─ POST /load-dataset ────────────►│
  │   (parquet bytes as body)        ├─ pd.read_parquet()
  │                                  ├─ TF-IDF fit
  │◄── { rows, categories, skills } ─┤
  │                                  │
  ├─ POST /recommend ───────────────►│
  │   { skills, experience, cats }   ├─ cosine_similarity()
  │◄── ranked job results ───────────┤
```

The server never calls HuggingFace — the browser fetches the parquet and
forwards the bytes to Flask, which parses and builds the ML model.

## Setup

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Main UI |
| POST | /load-dataset | Receive parquet bytes, build ML model |
| POST | /recommend | Get ranked job recommendations |
| GET | /job/<id> | Full job details |
| GET | /autocomplete?q=... | Skill suggestions |
| GET | /stats | Dataset stats |

## ML Pipeline

1. **TF-IDF Vectorization** — job_title + category + skills → 8,000-feature vector  
2. **Cosine Similarity** — user skill query vs all job vectors  
3. **Experience boosting** — ±0.12 score delta for level match  
4. **Category boosting** — +0.06 for preferred categories  
5. **Skill breakdown** — matched (green) and missing (red) chips per result