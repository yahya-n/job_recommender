 

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








-----------------

# JobMatch AI — CSV-Based ML Job Recommender

## Input Files
| File | Purpose |
|------|---------|
| `User-data-10000.csv` | 10,000 user profiles for training (hard_skill, soft_skill, candidate_field) |
| `jobs_data.csv` | 11 job categories with required hard/soft skills |

## ML Pipeline
1. **TF-IDF** (5,000 features, bigrams, sublinear_tf) — vectorise all user skill docs
2. **Random Forest** (200 trees, balanced class weights) — train on 80% of user data
3. **Evaluation** — 20% test split, per-class precision/recall/F1 report
4. **Cosine Similarity** — compare user query against job catalogue vectors
5. **Blend** — 60% RF probability + 40% cosine → final ranked score

## Run
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
# Model trains automatically on startup (~15 seconds)
```

## API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/status | Poll training status + KPIs |
| POST | /api/recommend | `{hard_skills, soft_skills}` → ranked jobs |
| GET | /api/autocomplete/hard?q=… | Hard skill suggestions |
| GET | /api/autocomplete/soft?q=… | Soft skill suggestions |
| GET | /api/metrics | Full classification report |