import random
from pathlib import Path

import numpy as np
import streamlit as st

from speech_buddy_core import (
    MODEL_ID,
    WORDS_FILE,
    load_words,
    load_model,
    build_filler_phone_sequences,
    make_expected_pronunciations,
    record_one_attempt,
    detect_phonemes,
    remove_edge_fillers,
    assess,
    make_feedback,
)

st.set_page_config(
    page_title="voAIce Speech Buddy",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: linear-gradient(135deg, #f4f1ff 0%, #eef8ff 55%, #f4fff9 100%); }
.block-container { padding-top: 1.6rem; max-width: 1120px; }
.hero {
  background: linear-gradient(110deg, #6c63d9, #8d7cf6);
  color: white; padding: 1.6rem 1.8rem; border-radius: 24px;
  box-shadow: 0 12px 30px rgba(83, 75, 170, .18); margin-bottom: 1.2rem;
}
.hero h1 { color: white; font-size: 2.3rem; margin: 0; font-weight: 900; }
.hero p { color: #f1efff; margin: .35rem 0 0 0; font-size: 1rem; }
.word-card {
  background: white; border: 2px solid #e7e3ff; border-radius: 22px;
  padding: 1.5rem; text-align: center; box-shadow: 0 8px 24px rgba(60,60,100,.06);
}
.word { font-size: 3rem; font-weight: 900; color: #5148b8; letter-spacing: 1px; }
.small-muted { color: #72718a; font-size: .9rem; }
.result-card {
  background: white; border-radius: 18px; padding: 1.2rem;
  border: 1px solid #e9e7f5; margin: .5rem 0;
}
.pill { display:inline-block; padding:.3rem .7rem; border-radius:20px;
  background:#efedff; color:#5148b8; font-weight:800; font-size:.85rem; }
div.stButton > button {
  border-radius: 14px; font-weight: 800; min-height: 3rem;
}
[data-testid="stMetric"] {
  background: white; padding: 1rem; border-radius: 16px;
  border: 1px solid #e9e7f5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🗣️ voAIce Speech Buddy</h1>
  <p>A playful pronunciation practice demo powered by Wav2Vec2 phoneme recognition</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_engine():
    model, tokenizer, extractor, device = load_model()
    fillers = build_filler_phone_sequences(tokenizer)
    return model, tokenizer, extractor, device, fillers

def initialise_state():
    defaults = {
        "selected_level": None,
        "level_words": [],
        "word_index": 0,
        "results": [],
        "last_result": None,
        "engine_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialise_state()

with st.sidebar:
    st.markdown("## 🎮 Practice menu")
    st.caption("Choose a level to begin.")
    try:
        all_words = load_words()
        levels = sorted({item["level"] for item in all_words})
    except Exception as exc:
        st.error(f"Word list problem: {exc}")
        st.stop()

    chosen = st.selectbox("Practice level", levels, format_func=lambda x: f"Level {x}")
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.write("1. Choose a practice level.")
    st.write("2. Look at the target word.")
    st.write("3. Press the microphone button and say the word.")
    st.write("4. Review the sound comparison and feedback.")
    st.markdown("---")
    st.caption("Audio is processed locally in memory and is not saved or uploaded.")

if st.session_state.selected_level != chosen:
    st.session_state.selected_level = chosen
    st.session_state.level_words = [
        item["target_word"] for item in all_words if item["level"] == chosen
    ]
    random.shuffle(st.session_state.level_words)
    st.session_state.word_index = 0
    st.session_state.results = []
    st.session_state.last_result = None

words = st.session_state.level_words
total = len(words)
idx = st.session_state.word_index

top1, top2, top3 = st.columns(3)
top1.metric("Practice level", f"Level {chosen}")
top2.metric("Words in this level", total)
top3.metric("Words completed", len(st.session_state.results))

st.progress(min((idx / total) if total else 0, 1.0), text=f"Progress: {idx} of {total} words")

if idx < total:
    target = words[idx]
    left, right = st.columns([1.15, .85], gap="large")

    result = st.session_state.last_result

    with left:
        st.markdown(f"""
        <div class="word-card">
          <div class="small-muted">WORD {idx + 1} OF {total}</div>
          <div class="word">{target.capitalize()}</div>
          <div class="small-muted">Look at the word, then say it clearly 🎤</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🎙️ Record & analyse", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading speech model (first run may take a while)..."):
                    model, tokenizer, extractor, device, fillers = get_engine()
                with st.spinner("Listening... say the word after the short pause."):
                    audio = record_one_attempt()
                if audio is None:
                    st.warning("I couldn't capture your voice clearly. Please try again.")
                else:
                    with st.spinner("Checking the speech sounds..."):
                        candidates = make_expected_pronunciations(target, tokenizer)
                        actual = detect_phonemes(
                            audio, model, tokenizer, extractor, device
                        )
                        actual, ignored = remove_edge_fillers(actual, fillers)
                        comparisons = [
                            (candidate, assess(candidate["phones"], actual))
                            for candidate in candidates
                        ]
                        usable = [pair for pair in comparisons if pair[1]["score"] is not None]
                        if usable:
                            selected, result_data = max(usable, key=lambda pair: pair[1]["score"])
                        else:
                            selected = candidates[0]
                            result_data = assess(selected["phones"], actual)

                        record = {
                            "word": target,
                            "score": result_data["score"],
                            "status": result_data["status"],
                            "feedback": make_feedback(target, result_data),
                            "expected": selected["phones"],
                            "actual": actual,
                            "operations": result_data["operations"],
                            "ignored_fillers": ignored,
                        }
                        st.session_state.last_result = record
                        existing = next(
                            (i for i, r in enumerate(st.session_state.results)
                             if r["word"] == target and i == idx), None
                        )
                        if existing is None:
                            st.session_state.results.append(record)
                        else:
                            st.session_state.results[existing] = record
            except Exception as exc:
                st.error(f"Could not analyse this attempt: {type(exc).__name__}: {exc}")

        # Mission information sits underneath the flashcard and recording button.
        st.markdown("---")
        st.markdown("### 🌟 Your mission")
        st.write("Try your best to say the word. You can practise again if you want.")
        st.info("The recording stops automatically after around 2 seconds of silence, or at the recording time limit.")
        st.markdown("**What the system checks**")
        st.write("• Expected pronunciation from CMUdict")
        st.write("• Sounds detected from the microphone")
        st.write("• Possible sound match, substitution, omission or addition")

    with right:
        # Pronunciation result appears beside the target word.
        if result and result["word"] == target:
            st.markdown("### 🏅 Your pronunciation result")
            if result["score"] is None:
                st.warning("Unable to assess this attempt. Try recording again in a quiet place.")
            else:
                st.metric("Practice score", f'{result["score"]:.1f}%')
                if result["status"] == "LIKELY_MATCH":
                    st.success("Sounds matched for this attempt!")
                else:
                    st.warning("Possible sound differences detected")

            st.markdown(
                f'<div class="result-card"><b>💬 Feedback</b><br>{result["feedback"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Expected sounds (CMUdict)**")
            st.write("  ".join(f"/{p}/" for p in result["expected"]) or "—")
            st.markdown("**Sounds detected (Wav2Vec2)**")
            st.write("  ".join(f"/{p}/" for p in result["actual"]) or "Unclear")

            with st.expander("View detailed sound comparison"):
                for op in result["operations"]:
                    if op["type"] == "match":
                        st.write(f"✅ /{op['expected']}/ → /{op['actual']}/ — Match")
                    elif op["type"] == "substitution":
                        st.write(f"🔁 /{op['expected']}/ → /{op['actual']}/ — Possible substitution")
                    elif op["type"] == "omission":
                        st.write(f"➖ /{op['expected']}/ — Possible omission")
                    elif op["type"] == "addition":
                        st.write(f"➕ /{op['actual']}/ — Possible addition")
                if result["ignored_fillers"]:
                    st.caption("Hesitation sounds filtered: " + ", ".join(result["ignored_fillers"]))
        else:
            st.markdown("### 🏅 Your pronunciation result")
            st.info("Your result will appear here after you record the word.")

    nav1, nav2, nav3 = st.columns([1, 1, 2])
    with nav1:
        if st.button("↻ Try this word again", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()
    with nav2:
        if st.button("Next word ➜", type="primary", use_container_width=True,
                     disabled=not any(r["word"] == target for r in st.session_state.results)):
            st.session_state.word_index += 1
            st.session_state.last_result = None
            st.rerun()
    with nav3:
        if st.button("Finish session", use_container_width=True):
            st.session_state.word_index = total
            st.rerun()

else:
    st.markdown("## 🎉 Level completed!")
    results = st.session_state.results
    assessed = [r for r in results if r["score"] is not None]
    avg = float(np.mean([r["score"] for r in assessed])) if assessed else None
    a, b, c = st.columns(3)
    a.metric("Words assessed", len(results))
    b.metric("Average practice score", f"{avg:.1f}%" if avg is not None else "N/A")
    c.metric("Unable to assess", sum(r["score"] is None for r in results))

    if results:
        st.markdown("### 📋 Word-by-word summary")
        st.dataframe(
            [{
                "Word": r["word"].capitalize(),
                "Practice score": f'{r["score"]:.1f}%' if r["score"] is not None else "N/A",
                "Status": r["status"],
            } for r in results],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No words were assessed in this session.")

    if st.button("🔄 Practise this level again", type="primary"):
        st.session_state.level_words = [
            item["target_word"] for item in all_words if item["level"] == chosen
        ]
        random.shuffle(st.session_state.level_words)
        st.session_state.word_index = 0
        st.session_state.results = []
        st.session_state.last_result = None
        st.rerun()

st.markdown("---")
st.caption(
    "Presentation prototype • General-purpose phoneme-recognition model • "
    "Scores are experimental practice indicators, not clinical assessments. "
    "TTS is not included in this version."
)
