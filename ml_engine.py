"""
ML Engine — TF-IDF + Cosine Similarity job recommender.
Loaded once when the browser delivers the HuggingFace parquet bytes.
"""
import io, ast, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Singleton ─────────────────────────────────────────────
class MLEngine:
    def __init__(self):
        self.df         = None
        self.matrix     = None
        self.vectorizer = None
        self.all_skills = []
        self.categories = []
        self.ready      = False

    # ── Ingest ────────────────────────────────────────────
    def ingest_parquet(self, raw_bytes: bytes) -> dict:
        df = pd.read_parquet(io.BytesIO(raw_bytes))
        return self._load(df)

    def _load(self, df: pd.DataFrame) -> dict:
        self.df = df.reset_index(drop=True)
        self._preprocess()
        self._fit()
        self.ready = True
        return {
            "rows":       len(self.df),
            "categories": self.categories,
            "skills":     len(self.all_skills),
        }

    # ── Pre-process ───────────────────────────────────────
    def _preprocess(self):
        def parse_skills(raw):
            if pd.isna(raw):
                return []
            try:
                out = ast.literal_eval(str(raw))
                if isinstance(out, list):
                    return [s.strip().lower() for s in out if str(s).strip()]
            except Exception:
                pass
            return [s.strip().lower()
                    for s in re.split(r"[,;\n|]", str(raw)) if s.strip()]

        self.df["_skills"] = self.df["job_skill_set"].apply(parse_skills)
        self.df["_skills_str"] = self.df["_skills"].apply(" ".join)

        # combined corpus field
        self.df["_doc"] = (
            self.df["job_title"].fillna("") + " "
            + self.df["category"].fillna("") + " "
            + self.df["_skills_str"]
        )

        skill_universe: set = set()
        for lst in self.df["_skills"]:
            skill_universe.update(lst)
        self.all_skills = sorted(skill_universe)
        self.categories = sorted(self.df["category"].dropna().unique().tolist())

    # ── Fit TF-IDF ────────────────────────────────────────
    def _fit(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["_doc"])

    # ── Recommend ─────────────────────────────────────────
    def recommend(
        self,
        skills: list[str],
        experience: str = "",
        categories: list[str] = None,
        top_n: int = 10,
    ) -> list[dict]:
        if not self.ready or not skills:
            return []

        # Build query document
        query = " ".join(s.strip().lower() for s in skills)
        if categories:
            query += " " + " ".join(categories) * 3   # boost weight

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten().copy()

        tmp = self.df.copy()
        tmp["_score"] = scores

        # ── Experience bonus ──────────────────────────────
        SENIOR_KW = {"senior","sr","director","lead","head","chief","vp",
                     "manager","principal","staff"}
        JUNIOR_KW = {"junior","jr","entry","associate","intern","graduate"}

        if experience:
            lvl = experience.lower()
            def exp_delta(row):
                t = (str(row.get("job_title","")) + " " +
                     str(row.get("job_description",""))).lower()
                words = set(re.findall(r"[a-z]+", t))
                if lvl == "entry":
                    if words & JUNIOR_KW:  return  0.12
                    if words & SENIOR_KW:  return -0.08
                elif lvl == "mid":
                    if words & {"senior","sr","lead"}: return 0.04
                    if words & {"director","chief","vp"}: return -0.04
                elif lvl == "senior":
                    if words & SENIOR_KW:  return  0.12
                return 0
            tmp["_score"] += tmp.apply(exp_delta, axis=1)

        # ── Category boost ────────────────────────────────
        if categories:
            tmp.loc[tmp["category"].isin(categories), "_score"] += 0.06

        # ── Rank & format ─────────────────────────────────
        top = (tmp[tmp["_score"] > 0.005]
               .sort_values("_score", ascending=False)
               .head(top_n))

        user_set = {s.strip().lower() for s in skills}
        results = []
        for _, row in top.iterrows():
            job_skills = row["_skills"]
            job_set    = set(job_skills)
            matched    = sorted(user_set & job_set)
            missing    = sorted(job_set - user_set)
            pct        = round(len(matched) / max(len(job_set), 1) * 100)
            desc       = str(row.get("job_description", ""))

            results.append({
                "job_id":          str(row.get("job_id", "")),
                "category":        str(row.get("category", "")),
                "job_title":       str(row.get("job_title", "")),
                "description":     desc[:350] + "…" if len(desc) > 350 else desc,
                "all_skills":      job_skills,
                "matched_skills":  matched,
                "missing_skills":  missing[:7],
                "match_pct":       pct,
                "score":           round(float(row["_score"]) * 100, 1),
            })
        return results

    # ── Helpers ───────────────────────────────────────────
    def get_job(self, job_id: str) -> dict | None:
        row = self.df[self.df["job_id"].astype(str) == job_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "job_id":      str(r["job_id"]),
            "category":    str(r["category"]),
            "job_title":   str(r["job_title"]),
            "description": str(r.get("job_description", "")),
            "all_skills":  r["_skills"],
        }

    def skill_suggest(self, q: str) -> list[str]:
        q = q.lower()
        return [s for s in self.all_skills if q in s][:14]

    def stats(self) -> dict:
        if not self.ready:
            return {"total_jobs": 0, "categories": {}, "total_skills": 0}
        return {
            "total_jobs":   len(self.df),
            "categories":   self.df["category"].value_counts().to_dict(),
            "total_skills": len(self.all_skills),
        }


# Shared singleton used by Flask
engine = MLEngine()