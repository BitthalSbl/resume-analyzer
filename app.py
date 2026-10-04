"""
AI-Powered Resume Analyzer & Job Matcher
Production-ready Streamlit app for free-tier Render deployment.
Supports: OpenAI + Groq + xAI Grok
"""

import os
import json
import re
from typing import Optional, Dict, Any, Tuple, List

import streamlit as st
import pdfplumber
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

# Load environment variables (local development)
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 8000
# llama-3.3-70b-versatile was retired by Groq on 2026-08-16
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_client() -> Tuple[Optional[OpenAI], str]:
    """
    Initialize an OpenAI-compatible client. Returns (client, model_name).

    Supports OpenAI, Groq (keys start with gsk_) and xAI Grok.
    Secrets are read from environment variables (Render dashboard / .env locally).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")

    api_key = groq_key or openai_key or xai_key
    if not api_key:
        return None, "Not detected"

    model = os.getenv("LLM_MODEL")
    base_url = None

    if groq_key or api_key.startswith("gsk_"):
        base_url = "https://api.groq.com/openai/v1"
        model = model or DEFAULT_GROQ_MODEL
    elif xai_key and not openai_key:
        base_url = "https://api.x.ai/v1"
        # xAI renames/retires models often, so set LLM_MODEL explicitly.
        model = model or os.getenv("XAI_MODEL", "grok-3-mini")
    else:
        model = model or "gpt-4o-mini"

    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2), model


def extract_text_from_pdf(uploaded_file) -> str:
    """Safely extract text from PDF using pdfplumber."""
    try:
        text_parts = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def clean_json_response(raw: str) -> str:
    """Extract pure JSON from an LLM response (handles fences and extra prose)."""
    raw = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        raw = match.group(1).strip()
    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last > first:
        return raw[first:last + 1]
    return raw


def _as_list(value: Any) -> List[str]:
    """Coerce a model-provided value into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [v.strip() for v in re.split(r"[\n,;]+", value) if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _as_score(value: Any) -> int:
    """Turn 75, 75.4, '75%' or '75/100' into an int clamped to 0-100."""
    if isinstance(value, (int, float)):
        num = value
    else:
        m = re.search(r"\d+(?:\.\d+)?", str(value))
        if not m:
            raise ValueError(f"Invalid ats_score in LLM response: {value!r}")
        num = float(m.group())
    return max(0, min(100, int(round(num))))


SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) and career coach.
Analyze the resume against the job description and return ONLY valid JSON.
Do not include any extra text, markdown, or explanations outside the JSON."""

# NOTE: literal braces are doubled ({{ }}) because this string goes through .format()
USER_PROMPT_TEMPLATE = """Resume Text:
\"\"\"
{resume}
\"\"\"

Job Description:
\"\"\"
{jd}
\"\"\"

Return ONLY a valid complete JSON object. No markdown, no extra text.
Use exactly these keys:
{{
  "ats_score": 75,
  "matched_keywords": ["Python", "React"],
  "missing_keywords": ["Docker", "AWS"],
  "strengths": ["Strong Python skills", "Good frontend experience"],
  "weaknesses": ["Missing cloud experience", "No Docker mentioned"],
  "recommendations": [
    "Add Docker and containerization experience",
    "Highlight any cloud deployment projects",
    "Include specific metrics in project descriptions"
  ]
}}

Rules:
- ats_score must be an integer 0-100
- Keep lists short (3-6 items max)
- recommendations must have exactly 3 to 5 items
- Output the full JSON completely
"""


def analyze_resume_with_llm(resume_text: str, job_description: str,
                            client: OpenAI, model: str) -> Dict[str, Any]:
    """Call the LLM and force structured JSON output."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        resume=resume_text[:MAX_RESUME_CHARS],
        jd=job_description[:MAX_JD_CHARS],
    )
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4000,  # headroom: reasoning models spend tokens thinking
    )

    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except BadRequestError as e:
        # Model doesn't support JSON mode -> retry without it,
        # but surface genuine problems like a retired model.
        if "decommissioned" in str(e) or "model_not_found" in str(e):
            raise ValueError(
                f"Model '{model}' is not available. Set LLM_MODEL to a currently "
                f"supported model.\n\n{e}"
            )
        response = client.chat.completions.create(**kwargs)

    raw_content = response.choices[0].message.content or ""
    if not raw_content.strip():
        raise ValueError("The model returned an empty response. Please try again.")

    cleaned = clean_json_response(raw_content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw response:\n{raw_content}")

    required_keys = ["ats_score", "matched_keywords", "missing_keywords",
                     "strengths", "weaknesses", "recommendations"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in LLM response: {key}")

    return {
        "ats_score": _as_score(data["ats_score"]),
        "matched_keywords": _as_list(data["matched_keywords"]),
        "missing_keywords": _as_list(data["missing_keywords"]),
        "strengths": _as_list(data["strengths"]),
        "weaknesses": _as_list(data["weaknesses"]),
        "recommendations": _as_list(data["recommendations"]),
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.title("📄 AI-Powered Resume Analyzer & Job Matcher")
    st.markdown(
        "Upload your resume (PDF) + paste a Job Description → Get ATS score, "
        "matched/missing keywords, strengths, weaknesses & actionable recommendations."
    )

    # Check API key early
    client, active_model = get_client()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(f"**Model:** `{active_model}`")
        st.markdown("---")
        
        if client:
            st.success("✅ API Key detected")
        else:
            st.error("❌ No API Key found")
        
        st.info(
            "Supported providers:\n"
            "- **Groq** (free) → key starts with `gsk_`\n"
            "- **OpenAI** → key starts with `sk-`\n"
            "- **xAI Grok** → use `XAI_API_KEY`"
        )
        st.markdown("---")
        st.caption("Built for production • Clean JSON pipeline • Robust error handling")

    if client is None:
        st.error(
            "❌ **API Key Missing**\n\n"
            "**Local:** create a `.env` file next to `app.py` with "
            "`OPENAI_API_KEY=gsk_your_groq_key_here`, then restart the app.\n\n"
            "**Render:** add `OPENAI_API_KEY` under *Environment* and redeploy."
        )
        st.stop()

    # Main layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Resume (PDF)")
        uploaded_file = st.file_uploader(
            "Choose a PDF resume",
            type=["pdf"],
            help="Only PDF files are supported."
        )

    with col2:
        st.subheader("2. Paste Job Description")
        job_description = st.text_area(
            "Job Description",
            height=220,
            placeholder="Paste the full job description here...",
        )

    analyze_button = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

    if analyze_button:
        if not uploaded_file:
            st.error("Please upload a PDF resume.")
            st.stop()
        if not job_description or len(job_description.strip()) < 50:
            st.error("Please paste a meaningful Job Description (at least 50 characters).")
            st.stop()

        with st.spinner("Extracting text from PDF..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        if len(resume_text) < 100:
            st.error("Could not extract enough text from the PDF. Please use a text-based PDF (not scanned image).")
            st.stop()

        with st.spinner("Running AI analysis... This may take 10-30 seconds."):
            try:
                result = analyze_resume_with_llm(resume_text, job_description, client, active_model)
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()

        # Results
        st.success("✅ Analysis Complete!")

        score = result["ats_score"]
        score_color = "green" if score >= 75 else "orange" if score >= 50 else "red"
        st.markdown(f"### ATS Match Score: :{score_color}[**{score}%**]")
        st.progress(score / 100)

        kcol1, kcol2 = st.columns(2)
        with kcol1:
            st.subheader("✅ Matched Keywords")
            if result["matched_keywords"]:
                st.markdown(", ".join(f"`{kw}`" for kw in result["matched_keywords"]))
            else:
                st.write("None detected")

        with kcol2:
            st.subheader("❌ Missing Keywords")
            if result["missing_keywords"]:
                st.markdown(", ".join(f"`{kw}`" for kw in result["missing_keywords"]))
            else:
                st.write("None critical")

        scol1, scol2 = st.columns(2)
        with scol1:
            st.subheader("💪 Strengths")
            for s in result["strengths"]:
                st.markdown(f"- {s}")

        with scol2:
            st.subheader("⚠️ Weaknesses / Gaps")
            for w in result["weaknesses"]:
                st.markdown(f"- {w}")

        st.subheader("🎯 Actionable Recommendations")
        for i, rec in enumerate(result["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

        with st.expander("🔍 View Raw JSON Output"):
            st.json(result)

        with st.expander("📄 View Extracted Resume Text"):
            st.text_area("Resume Text", resume_text, height=300, disabled=True)


if __name__ == "__main__":
    main()
