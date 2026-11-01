# FactProtest India 🇮🇳
**AI-Powered Protest Credibility Checker · Civic Voice Platform**

Built for FORGE Buildathon 2026 by NxtWave Launchpad.

## What it does
- Citizens can browse 10 real Indian protests and submit opinions (Support/Oppose/Neutral)
- Groq AI scores each opinion's credibility (0–100) based on reasoning quality
- Authority Portal for government officials to submit official replies
- AI Civic Reports generate real-time protest sentiment summaries

## Tech Stack
- **Frontend**: Streamlit 1.40
- **AI**: Groq API (openai/gpt-oss-20b)
- **Database**: Supabase
- **Charts**: Plotly

## Live Demo
https://factpotest-india-43fdfkvnzfw6vyuv8vpagk.streamlit.app

## Setup
1. Clone the repo
2. Add `.streamlit/secrets.toml` with your API keys
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
