# AI-Powered Resume Analyzer & Job Matcher

Production-ready Streamlit application that analyzes your resume against a job description using AI (Groq, OpenAI or xAI Grok).

## Features

- PDF resume upload with safe text extraction (`pdfplumber`)
- Paste Job Description
- ATS Match Score (0-100%)
- Matched & Missing Keywords
- Strengths & Weaknesses
- 3-5 Actionable Recommendations
- Clean structured JSON output from the LLM
- Robust error handling (missing API key, bad PDF, invalid JSON)

## Tech Stack

- **Frontend / Backend**: Streamlit
- **PDF Parsing**: pdfplumber
- **AI**: Groq (free) / OpenAI / xAI Grok (all OpenAI-compatible)
- **Deployment**: Render (Free Web Service)

## Local Setup

1. Clone / download this project
2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Render Deployment (100% Free)

1. Push this folder to a GitHub repository.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect your repository.
4. Configure:

| Setting            | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| **Environment**    | Python 3                                                              |
| **Build Command**  | `pip install -r requirements.txt`                                     |
| **Start Command**  | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`   |
| **Instance Type**  | **Free**                                                              |

5. Add Environment Variable:
   - `OPENAI_API_KEY` = your Groq key (`gsk_...`) or OpenAI key
   - optional `LLM_MODEL` = `openai/gpt-oss-120b` (default for Groq)
   - or `XAI_API_KEY` (+ `LLM_MODEL`) for xAI Grok

   Alternatively use **New → Blueprint** – the included `render.yaml` sets everything up.

6. Click **Create Web Service**.

**Note**: Free tier spins down after inactivity. First request after sleep may take 20-40 seconds (cold start).

## Environment Variables

| Variable         | Required | Description                                      |
|------------------|----------|--------------------------------------------------|
| `OPENAI_API_KEY` | Yes*     | OpenAI API key                                   |
| `XAI_API_KEY`    | Yes*     | xAI Grok API key (use only one of the two keys)  |
| `LLM_MODEL`      | No       | Model name (defaults: Groq `openai/gpt-oss-120b`, OpenAI `gpt-4o-mini`) |

*Provide at least one API key.

## License

MIT – Feel free to use and modify.

> **Vercel note:** Streamlit needs a long-running server and WebSockets, so it cannot run on Vercel's serverless platform. Use Render (or Streamlit Community Cloud).
