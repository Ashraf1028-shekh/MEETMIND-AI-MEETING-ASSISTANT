# ============================================================
# 🎙️ AI Meeting Summarizer — MeetMind
# Groq  → audio transcription (Whisper Large v3 Turbo)
# Gemini → summarization      (gemini-2.0-flash)
# Groq  → fallback summarization (Llama 3.3 70b)
# Firebase → save summaries silently in background
# ============================================================

import streamlit as st
import json
import os
import io
import time
import datetime
from groq import Groq
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align: center; padding: 2.5rem 1rem 1rem; }
.hero h1 {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p { color: #94a3b8; font-size: 1.05rem; font-weight: 300; }

.steps-row {
    display: flex; justify-content: center; gap: 0.5rem;
    margin: 1.4rem 0 1.8rem; flex-wrap: wrap;
}
.step-pill {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 999px; padding: 0.3rem 1rem;
    font-size: 0.8rem; color: #94a3b8;
}
.step-pill.active {
    background: rgba(167,139,250,0.22);
    border-color: #a78bfa; color: #e9d5ff; font-weight: 600;
}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1.7rem;
    margin-bottom: 1.1rem;
}
.card-title { color: #e2e8f0; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.8rem; }
.summary-text { color: #cbd5e1; line-height: 1.8; font-size: 0.96rem; }
.bullet-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.42rem 0; color: #cbd5e1; font-size: 0.94rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.bullet-item:last-child { border-bottom: none; }
.dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 0.48rem; flex-shrink: 0; }
.dot-purple { background: #a78bfa; }
.dot-green  { background: #34d399; }
.transcript-box {
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 1rem 1.2rem; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    line-height: 1.75; max-height: 200px; overflow-y: auto; white-space: pre-wrap;
}
.badge {
    display: inline-block; background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.28); border-radius: 999px;
    padding: 0.18rem 0.7rem; font-size: 0.76rem; color: #93c5fd; margin-top: 0.5rem;
}
.firebase-badge {
    display: inline-block; background: rgba(251,146,60,0.12);
    border: 1px solid rgba(251,146,60,0.28); border-radius: 999px;
    padding: 0.18rem 0.7rem; font-size: 0.76rem; color: #fb923c; margin-top: 0.5rem; margin-left: 0.4rem;
}
.free-banner {
    background: rgba(52,211,153,0.09); border: 1px solid rgba(52,211,153,0.28);
    border-radius: 10px; padding: 0.65rem 1rem; color: #6ee7b7;
    font-size: 0.84rem; margin-bottom: 1rem; text-align: center;
}
.demo-banner {
    background: rgba(251,191,36,0.09); border: 1px solid rgba(251,191,36,0.28);
    border-radius: 10px; padding: 0.65rem 1rem; color: #fcd34d;
    font-size: 0.84rem; margin-bottom: 1rem; text-align: center;
}
.powered-row {
    display: flex; justify-content: center; gap: 0.6rem;
    margin-bottom: 1rem; flex-wrap: wrap;
}
.powered-pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 0.25rem 0.85rem;
    font-size: 0.75rem; color: #94a3b8;
}
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }

div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-family: 'Sora', sans-serif;
    font-weight: 600; font-size: 0.95rem; width: 100%; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; border: none; }
div.stDownloadButton > button {
    background: rgba(52,211,153,0.1); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3); border-radius: 10px;
    font-family: 'Sora', sans-serif; font-weight: 600; width: 100%;
}
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(167,139,250,0.3);
    border-radius: 12px; padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PASTE YOUR KEYS HERE
# ─────────────────────────────────────────────
GROQ_KEY   = "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"   # ← "gsk_xxxx"
GEMINI_KEY = "AIzaSyDzs-GFIa2EFE6MpB8BOWrX3JOm4VyFfLY" # ← "AIzaxxxx"

# Firebase service account JSON — paste the whole dict here
# Get this from Firebase Console → Project Settings → Service Accounts → Generate new private key
FIREBASE_CREDENTIALS = {
  "type": "service_account",
  "project_id": "meetmind-c1bb6",
  "private_key_id": "64af4e742aea73fd23bc59f1bdaa1ac46f933211",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC+iESPkBG7f9Zv\nZDfydak6bNhPIN8iBV9/AZk4bN909PJ/SgsUxlFp5+dY8v19clHg+u1F7qUtEOH4\nCWOLlmD74GET2u6H/RFr6ltAc1LlZNqZASR04zyXOalxkEMuGoXnXhRgY64AefOC\nSNPwTICi4f552Dz5ZoT3c4lv4Jap7r0F7qQKbB1fn+9BYUWB+Cx4+w3Zrj8pBU1C\nuW0uKelx/8xtWp5o2ErHbZ75kCu9o+ZADmDntXx4oOHmuo6JLWC5ortBqJr/gofG\nrMgF8FRgf76je9oxwu9+5HC9igpHPUGDFRM2iemqRQ0y6+KbEjW/pHKbmtpypuMt\nI2d2G1spAgMBAAECggEAB1cPExvI/xjt7SUmfZ34SsKqM6QvzGr6EpXAbhrNof8U\nv4Ad/KsmNEcFk9jqmJRYoKryDJ1kuh96IRjVf6lnq5RpkP8AbPc8HlTtNoaq1SEF\nRds0KRee/DFxvGkDGYnWikMG1G7Ko4SPM7/0AEAmPYAo0k9ZC/+d1D2ceBWmID47\n8wOblJW6EWiGK31OhaUeZhGiujBkPcS/H+SWcf4MIL/0Usz9jt+OVE86gJDlZbhi\ngXKTl+XsKt/eZ2friukuu7yplmvLa0a+qOfPaNH5U+L6xn+W+A9Ia0mbrS+6z216\nEg2yHxczLn9yzutqEKYloKy+rOqHPCeHW9hyY6AEvQKBgQDuu8YJPHyHNq37idfw\nSR6loPND1kYUugetNZapwWTQDcejcmXLA7nkBznQMGajo4oB/ZcZR6IhwoBrS+PU\nVHLClEO/LVP+rx1OmKvBKRfNqDoBiD27T5mfM9vWNTtZlJ0xNQiu94LaH19pAFgk\n2WmQcne6VD1vx2O4foR3z1kHpQKBgQDMUAinw3x8hLMHG4pydFRjnXH/NqARRWTo\ndZpgZ0UZg3DovsvtPU8nAvdEmofB/MTflklxzwU6b/T8n2thHwhfvqrSS7RrvPLh\nWvnsKCttj/YtTbNwd4SMCJ3JUoA4+83enh/Xh6uzU/pSDgBQEJL+s0SqHiBv2Nar\n5iEuW47ONQKBgQCx2VNcGCKRPKv6YyNqj550N2Zi2CNXV4tWf0ChcEChOlLKsfOh\nfUIj3To1nYkpSTEL8JZ7L8FaMyMmS+ALk2n3CSs7JhKX/m7v2Y7ClNF11Wh5Mb0D\naPaCAsk4WGnBf5RnTyIL7rodsxeKCsaflRQulwgY2J74QetJS392HhMuPQKBgGZB\n9GMK+2ZTAEuhZCSuxoHpDmI+09RxF/UkzzRK25HOcJd8zxIZKmGLIXkLosDboymI\nzY+RnrtFx6ccnyDH6GgLRPHmBBIcSq+x2f5xiHLYp8y65AF5QDTaJybZDOX4WWx/\nVc6Iz838TP/+k/d+pCOoU8iVNYTlUWRN1POfzZS1AoGBANRWAX1LMc758MpmJwho\nwAVR4CWLaYzjQ6ZkXi/MsDIKrj6RxCH1s6i7H7ie3q9KeaM8aMozovESxetyWC/u\nz2Ddkxry19BYoQ108DOY4v8UaPU0vTmICTzK/C7q+MwDZjqvIPv6rbtaPe1dCwEK\nrW6JQfG4y6cdJr+QEJ03twGX\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-fbsvc@meetmind-c1bb6.iam.gserviceaccount.com",
  "client_id": "107707560633874668786",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40meetmind-c1bb6.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
 # ← paste dict here, e.g. {"type": "service_account", ...}


# ─────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """Good morning everyone. Thanks for joining the Q3 product review.

Alice: Let's start with the mobile app. We hit 50,000 downloads this month,
which is 20% above our target. However, the crash rate on Android is still high —
around 3.2%. We need to fix that before the next release.

Bob: Agreed. I'll assign two engineers to investigate the crash logs by end of day.
We should have a patch ready by Friday.

Carol: On the marketing side, the TikTok campaign drove most of the installs.
We're planning to double the budget next quarter. I'll send the proposal to finance by Thursday.

Alice: Great. One more thing — pricing for the premium tier. $9.99/month. Thoughts?

Bob: Seems competitive. Let's go with it and revisit after 60 days.

Alice: Perfect. Action items: Bob's team fixes Android crashes by Friday,
Carol sends the budget proposal Thursday, and we'll launch premium pricing next week."""

MOCK_RESULT = {
    "summary": "The Q3 product review covered the mobile app's strong performance (50K downloads, 20% above target), a critical Android crash issue, a successful TikTok campaign with plans to double the budget, and a decision to launch premium pricing at $9.99/month.",
    "key_points": [
        "Mobile app hit 50,000 downloads — 20% above Q3 target",
        "Android crash rate at 3.2%; patch targeted for Friday",
        "TikTok campaign was the primary driver of new installs",
        "Marketing budget for TikTok to be doubled next quarter",
        "Premium tier pricing agreed at $9.99/month, reviewed after 60 days",
    ],
    "action_items": [
        "Bob's team: Investigate Android crash logs and ship patch by Friday",
        "Carol: Send doubled TikTok budget proposal to Finance by Thursday",
        "Product team: Launch premium pricing tier next week",
    ],
}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "transcript": None,
    "result": None,
    "processed_id": None,
    "firebase_ready": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# FIREBASE INIT
# ─────────────────────────────────────────────
def init_firebase():
    """
    Initialize Firebase app once per session.
    Reads credentials from FIREBASE_CREDENTIALS dict or
    FIREBASE_CREDENTIALS_JSON environment variable.
    Returns Firestore client or None if unavailable.
    """
    try:
        # Already initialized
        if firebase_admin._apps:
            return firestore.client()

        # Try hardcoded credentials first
        creds_dict = FIREBASE_CREDENTIALS

        # Try environment variable (for Cloud Run deployment)
        if not creds_dict:
            creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
            if creds_json:
                creds_dict = json.loads(creds_json)

        if not creds_dict:
            return None  # No credentials — silently skip Firebase

        cred = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(cred)
        return firestore.client()

    except Exception:
        return None  # Firebase unavailable — app still works fine


def save_to_firebase(db, transcript: str, result: dict, source: str):
    """
    Save summary to Firebase Firestore silently.
    Collection: meetmind_summaries
    Does nothing if db is None.
    """
    if db is None:
        return

    try:
        db.collection("meetmind_summaries").add({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": source,                          # "audio" or "text"
            "word_count": len(transcript.split()),
            "transcript": transcript,
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", []),
            "action_items": result.get("action_items", []),
        })
    except Exception:
        pass  # Fail silently — never break the user experience


# Initialize Firebase once
db = init_firebase()
st.session_state.firebase_ready = db is not None


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def transcribe_with_groq(api_key: str, audio_bytes: bytes, filename: str) -> str:
    """Transcribe audio using Groq Whisper Large v3 Turbo. Free."""
    client = Groq(api_key=api_key)
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    response = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=buf,
        response_format="text",
    )
    return response


def summarize_with_gemini(api_key: str, transcript: str) -> dict:
    """Summarize using Google Gemini 2.0 Flash. Free."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are an expert meeting analyst. "
        "Return ONLY valid JSON — no markdown fences, no extra text — with this exact structure:\n"
        '{"summary": "...", "key_points": ["...", "..."], "action_items": ["...", "..."]}\n'
        "Be specific. Include owners and deadlines in action items when mentioned.\n\n"
        f"Meeting transcript:\n\n{transcript}"
    )
    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return json.loads(raw)


def summarize_with_groq(api_key: str, transcript: str) -> dict:
    """Summarize using Groq Llama 3.3 70b. Fallback when Gemini quota exceeded. Free."""
    client = Groq(api_key=api_key)
    system_prompt = (
        "You are an expert meeting analyst. "
        "Return ONLY valid JSON — no markdown fences, no extra text — with this exact structure:\n"
        '{"summary": "...", "key_points": ["...", "..."], "action_items": ["...", "..."]}\n'
        "Be specific. Include owners and deadlines in action items when mentioned."
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Meeting transcript:\n\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return json.loads(raw)


def build_download_text(transcript: str, result: dict) -> str:
    """Format results as clean plain-text for download."""
    sep = "=" * 58
    lines = [sep, "  AI MEETING SUMMARY — MeetMind", sep, "",
             "SUMMARY", "-" * 40, result.get("summary", ""), "",
             "KEY POINTS", "-" * 40]
    for i, kp in enumerate(result.get("key_points", []), 1):
        lines.append(f"  {i}. {kp}")
    lines += ["", "ACTION ITEMS", "-" * 40]
    for i, ai in enumerate(result.get("action_items", []), 1):
        lines.append(f"  {i}. {ai}")
    lines += ["", sep, "  FULL TRANSCRIPT", sep, "", transcript]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")

    st.markdown("### 🟣 Groq (transcription + fallback)")
    st.markdown("Free · [console.groq.com](https://console.groq.com)")
    groq_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")

    st.markdown("### 🔵 Gemini (summarization)")
    st.markdown("Free · [aistudio.google.com](https://aistudio.google.com)")
    gemini_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...")

    st.markdown("---")
    st.markdown("**Supported audio:** MP3 · MP4 · WAV · OGG · M4A · WebM")
    st.markdown("**Max size:** 25 MB")
    st.markdown("---")

    # Firebase status
    if st.session_state.firebase_ready:
        st.markdown("🔥 **Firebase:** Connected")
    else:
        st.markdown("🔥 **Firebase:** Not configured")

    st.caption("Groq Whisper · Gemini 2.0 Flash · Llama 3.3 · Firebase")

# Resolve keys
groq_key   = GROQ_KEY   or os.environ.get("GROQ_API_KEY")   or groq_input.strip()   or None
gemini_key = GEMINI_KEY or os.environ.get("GEMINI_API_KEY") or gemini_input.strip() or None
demo_mode  = not groq_key and not gemini_key


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎙️ MeetMind</h1>
  <p>Upload a recording · Get summaries, key points &amp; action items — <strong style="color:#34d399">100% free</strong></p>
</div>
""", unsafe_allow_html=True)

# Progress pills
step = 1
if st.session_state.transcript: step = 2
if st.session_state.result:     step = 3

pills = "".join(
    f'<div class="step-pill {"active" if step >= i else ""}">{label}</div>'
    for i, label in [(1, "① Input"), (2, "② Process"), (3, "③ Results")]
)
st.markdown(f'<div class="steps-row">{pills}</div>', unsafe_allow_html=True)

# Powered by
firebase_pill = '🔥 Firebase Firestore' if st.session_state.firebase_ready else '🔥 Firebase (not configured)'
st.markdown(f"""
<div class="powered-row">
  <div class="powered-pill">🟣 Groq Whisper</div>
  <div class="powered-pill">🔵 Gemini 2.0 Flash</div>
  <div class="powered-pill">🦙 Llama 3.3 fallback</div>
  <div class="powered-pill">{firebase_pill}</div>
</div>
""", unsafe_allow_html=True)

# Status banner
if not demo_mode:
    st.markdown('<div class="free-banner">✅ <strong>Live Mode Active</strong> — Groq + Gemini ready. Firebase saving summaries in background.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="demo-banner">🎭 <strong>Demo Mode</strong> — No API keys detected. Mock output shown. Add keys in the sidebar.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📝 Option A — Paste your transcript</div>', unsafe_allow_html=True)
pasted_text = st.text_area(
    "transcript", height=130,
    placeholder="Paste your meeting notes or transcript here...",
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#475569;font-size:0.85rem;margin:0.2rem 0 0.8rem;">— or —</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">📂 Option B — Upload an audio file</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "audio", type=["mp3", "mp4", "wav", "ogg", "webm", "m4a"],
    label_visibility="collapsed",
)
if uploaded_file:
    st.audio(uploaded_file, format=uploaded_file.type)
    st.markdown(f"**{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────
process_clicked = st.button("🚀 Summarize Meeting")

if process_clicked:
    has_text  = bool(pasted_text and pasted_text.strip())
    has_audio = uploaded_file is not None

    if not has_text and not has_audio:
        st.error("⚠️ Please paste a transcript OR upload an audio file.")
        st.stop()

    process_id = (uploaded_file.name if has_audio else "text") + str(len(pasted_text or ""))

    if process_id == st.session_state.processed_id and st.session_state.result:
        st.info("✅ Already processed — results shown below.")
    else:
        input_source = "audio" if has_audio else "text"

        # ── DEMO MODE ──────────────────────────────────
        if demo_mode:
            with st.spinner("🎭 Simulating transcription…"):
                time.sleep(1.0)
            st.session_state.transcript = pasted_text.strip() if has_text else SAMPLE_TRANSCRIPT
            with st.spinner("🎭 Generating mock summary…"):
                time.sleep(0.8)
            st.session_state.result = MOCK_RESULT

        else:
            # ── STEP 1: Get transcript ──────────────────
            if has_text:
                st.session_state.transcript = pasted_text.strip()
            else:
                if uploaded_file.size > 25 * 1024 * 1024:
                    st.error("❌ File too large. Max 25 MB supported.")
                    st.stop()
                try:
                    with st.spinner("🎧 Transcribing with Groq Whisper…"):
                        transcript = transcribe_with_groq(
                            groq_key, uploaded_file.read(), uploaded_file.name
                        )
                    if not transcript or not transcript.strip():
                        st.error("⚠️ Empty transcript. Check your audio file.")
                        st.stop()
                    st.session_state.transcript = transcript
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                    st.stop()

            # ── STEP 2: Summarize ───────────────────────
            # Try Gemini first, fall back to Groq Llama if quota exceeded
            result = None
            used_fallback = False

            if gemini_key:
                try:
                    with st.spinner("✨ Summarizing with Google Gemini…"):
                        result = summarize_with_gemini(gemini_key, st.session_state.transcript)
                except Exception:
                    # Gemini failed (quota or error) — try Groq fallback
                    used_fallback = True

            if result is None and groq_key:
                try:
                    with st.spinner("🦙 Using Groq Llama fallback…"):
                        result = summarize_with_groq(groq_key, st.session_state.transcript)
                except Exception as e:
                    st.error(f"❌ Summarization failed: {e}")
                    st.stop()

            if result is None:
                st.error("❌ No summarization API available. Add a Groq or Gemini key.")
                st.stop()

            if used_fallback:
                st.info("ℹ️ Gemini quota exceeded — used Groq Llama as fallback automatically.")

            try:
                st.session_state.result = result if isinstance(result, dict) else json.loads(result)
            except Exception:
                st.error("❌ Unexpected output format. Please try again.")
                st.stop()

            # ── STEP 3: Save to Firebase silently ───────
            save_to_firebase(
                db,
                st.session_state.transcript,
                st.session_state.result,
                input_source,
            )

        st.session_state.processed_id = process_id
        st.success("✅ Done! Results below.")


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state.transcript and st.session_state.result:
    transcript = st.session_state.transcript
    result     = st.session_state.result

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Transcript
    word_count = len(transcript.split())
    firebase_saved = '<span class="firebase-badge">🔥 Saved to Firebase</span>' if st.session_state.firebase_ready else ''
    st.markdown(f"""
    <div class="card">
      <div class="card-title">📝 Transcript</div>
      <div class="transcript-box">{transcript}</div>
      <div class="badge">📊 {word_count} words</div>
      {firebase_saved}
    </div>""", unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="card">
      <div class="card-title">📋 Summary</div>
      <div class="summary-text">{result.get("summary", "")}</div>
    </div>""", unsafe_allow_html=True)

    # Key Points + Action Items
    col1, col2 = st.columns(2)
    with col1:
        kp_html = "".join(
            f'<div class="bullet-item"><div class="dot dot-purple"></div><span>{kp}</span></div>'
            for kp in result.get("key_points", [])
        )
        st.markdown(f'<div class="card"><div class="card-title">✅ Key Points</div>{kp_html}</div>', unsafe_allow_html=True)

    with col2:
        ai_html = "".join(
            f'<div class="bullet-item"><div class="dot dot-green"></div><span>{item}</span></div>'
            for item in result.get("action_items", [])
        )
        st.markdown(f'<div class="card"><div class="card-title">🎯 Action Items</div>{ai_html}</div>', unsafe_allow_html=True)

    # Download
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    base = os.path.splitext(uploaded_file.name)[0] if uploaded_file else "meeting"
    st.download_button(
        label="⬇️ Download Summary as TXT",
        data=build_download_text(transcript, result),
        file_name=f"summary_{base}.txt",
        mime="text/plain",
    )
    with st.expander("🔧 Raw JSON (copy-paste friendly)"):
        st.code(json.dumps(result, indent=2), language="json")

elif not pasted_text and not uploaded_file:
    st.markdown("""
    <div style="text-align:center;color:#475569;padding:2rem 0 1rem;">
        ☝️ Paste a transcript above or upload an audio file to get started
    </div>""", unsafe_allow_html=True)
