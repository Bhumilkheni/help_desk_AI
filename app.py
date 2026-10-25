"""HelpDesk AI – WhatsApp-style Streamlit demo.

Run:  streamlit run app.py
Needs (same folder): chatbot.pt, vocab.json, chatbot_core.py  (downloaded from the Colab notebook)
"""

import html
import os
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="HelpDesk AI", page_icon="💬", layout="centered")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.stApp { background: #efeae2; }
header[data-testid="stHeader"] { background: transparent; }
.wa-header { background:#075e54; color:#fff; padding:12px 16px; border-radius:12px;
             display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.wa-avatar { width:42px; height:42px; border-radius:50%; background:#25d366;
             display:flex; align-items:center; justify-content:center; font-size:22px; }
.wa-title { font-weight:600; font-size:17px; line-height:1.2; }
.wa-sub { font-size:12px; opacity:.85; }
.wa-row { display:flex; margin:6px 0; }
.wa-row.user { justify-content:flex-end; }
.wa-bubble { max-width:78%; padding:8px 12px 6px; border-radius:12px; font-size:15px;
             color:#111b21; box-shadow:0 1px 1px rgba(0,0,0,.18); word-wrap:break-word; }
.wa-row.user .wa-bubble { background:#d9fdd3; border-top-right-radius:2px; }
.wa-row.bot  .wa-bubble { background:#ffffff; border-top-left-radius:2px; }
.wa-time { font-size:11px; color:#667781; text-align:right; margin-top:2px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- model
for f in ("chatbot.pt", "vocab.json", "chatbot_core.py"):
    if not os.path.exists(f):
        st.error(
            f"Missing `{f}`. Download chatbot.pt, vocab.json and chatbot_core.py "
            "from the Colab notebook and put them next to app.py."
        )
        st.stop()

from chatbot_core import generate, load_chatbot  # noqa: E402


@st.cache_resource(show_spinner="Loading model…")
def get_bot():
    return load_chatbot("chatbot.pt", "vocab.json", "cpu")


encoder, decoder, vocab = get_bot()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio(
        "Decoding",
        ["greedy", "sample"],
        help="Greedy = most probable word. " "Sample = top-k sampling (more varied).",
    )
    temperature = st.slider(
        "Temperature", 0.3, 1.5, 0.8, 0.1, disabled=(method == "greedy")
    )
    top_k = st.slider("Top-k", 1, 50, 10, disabled=(method == "greedy"))
    if st.button("🗑️ Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()


# ---------------------------------------------------------------- chat state
def now():
    return datetime.now().strftime("%H:%M")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Hello! How can I help you today?", "time": now()}
    ]

prompt = st.chat_input("Type a message")
if prompt and prompt.strip():
    st.session_state.messages.append(
        {"role": "user", "text": prompt.strip(), "time": now()}
    )
    reply = generate(
        encoder,
        decoder,
        vocab,
        prompt,
        "cpu",
        method=method,
        temperature=temperature,
        top_k=top_k,
    )
    st.session_state.messages.append(
        {
            "role": "bot",
            "text": reply or "Sorry, I did not get that. Could you rephrase?",
            "time": now(),
        }
    )

# ---------------------------------------------------------------- render
st.markdown(
    '<div class="wa-header"><div class="wa-avatar">🤖</div><div>'
    '<div class="wa-title">HelpDesk AI</div><div class="wa-sub">online</div></div></div>',
    unsafe_allow_html=True,
)
bubbles = "".join(
    f'<div class="wa-row {m["role"]}"><div class="wa-bubble">{html.escape(m["text"])}'
    f'<div class="wa-time">{m["time"]}</div></div></div>'
    for m in st.session_state.messages
)
st.markdown(bubbles, unsafe_allow_html=True)
