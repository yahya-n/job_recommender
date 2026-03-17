from flask import Flask, render_template, request, jsonify
from ml_engine import engine

app = Flask(__name__)

# ── Pages ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Dataset ingestion (called by browser after it fetches HF parquet) ──
@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Browser GETs the parquet file from HuggingFace CDN,
    then POSTs the raw bytes here.  Flask parses + fits the ML model.
    """
    raw = request.get_data()
    if not raw:
        return jsonify(error="Empty body"), 400
    try:
        info = engine.ingest_parquet(raw)
        return jsonify(ok=True, **info)
    except Exception as exc:
        return jsonify(error=str(exc)), 500

# ── ML endpoints ──────────────────────────────────────────
@app.route("/api/recommend", methods=["POST"])
def recommend():
    if not engine.ready:
        return jsonify(error="Dataset not loaded yet"), 503
    body       = request.get_json(force=True)
    raw_skills = body.get("skills", "")
    skills     = [s.strip() for s in raw_skills.split(",") if s.strip()]
    if not skills:
        return jsonify(error="Enter at least one skill"), 400

    results = engine.recommend(
        skills     = skills,
        experience = body.get("experience", ""),
        categories = body.get("categories", []),
        top_n      = int(body.get("top_n", 10)),
    )
    return jsonify(results=results, total=len(results))

@app.route("/api/job/<job_id>")
def job_detail(job_id):
    if not engine.ready:
        return jsonify(error="Dataset not loaded"), 503
    job = engine.get_job(job_id)
    return (jsonify(job) if job else (jsonify(error="Not found"), 404))

@app.route("/api/autocomplete")
def autocomplete():
    q = request.args.get("q", "").strip()
    return jsonify(engine.skill_suggest(q) if len(q) >= 2 else [])

@app.route("/api/stats")
def stats():
    return jsonify(engine.stats())

if __name__ == "__main__":
    app.run(debug=True, port=5000)