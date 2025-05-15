# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # --- FFmpeg configuration for pydub ---
# from pydub import AudioSegment
# ffmpeg_path = r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
# ffprobe_path = r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
# AudioSegment.converter = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path
# os.environ["PATH"] += os.pathsep + r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-sh-shared\bin"
# # --------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import tempfile
# import speech_recognition as sr
# import json
# from streamlit_TTS import text_to_speech  # For text-to-speech

# DATA_FOLDER = r"C:\Users\yuvra\OneDrive\Desktop\philips.data"
# MERGED_JSON = os.path.join(DATA_FOLDER, "merged_faq.json")

# @st.cache_data
# def load_all_faq_data():
#     with open(MERGED_JSON, 'r', encoding='utf-8') as f:
#         faq_list = json.load(f)
#     qa_pairs = []
#     for item in faq_list:
#         if 'question' in item and 'answer' in item:
#             # Pattern 1: direct Q&A
#             qa_pairs.append({'Question': item['question'], 'Answer': item['answer']})
#         elif 'intents' in item:
#             # Pattern 2: intents list
#             for intent in item['intents']:
#                 # Each pattern in 'patterns' is a question, each response is an answer
#                 patterns = intent.get('patterns', [])
#                 responses = intent.get('responses', [])
#                 for pattern in patterns:
#                     for response in responses:
#                         qa_pairs.append({'Question': pattern, 'Answer': response})
#     faq_df = pd.DataFrame(qa_pairs)
#     return faq_df


# faq_df = load_all_faq_data()

# @st.cache_resource
# def load_model_and_index(faq_df):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     questions = faq_df['Question'].astype(str).tolist()
#     emb_file = os.path.join(DATA_FOLDER, 'faq_embeddings.npy')
#     idx_file = os.path.join(DATA_FOLDER, 'faiss_index.bin')

#     if not os.path.exists(emb_file) or len(np.load(emb_file)) != len(questions):
#         embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
#         np.save(emb_file, embeddings)
#     else:
#         embeddings = np.load(emb_file)

#     dim = embeddings.shape[1]
#     if not os.path.exists(idx_file) or faiss.read_index(idx_file).ntotal != len(questions):
#         index = faiss.IndexFlatIP(dim)
#         index.add(embeddings.astype('float32'))
#         faiss.write_index(index, idx_file)
#     else:
#         index = faiss.read_index(idx_file)

#     return model, index, embeddings

# model, index, embeddings = load_model_and_index(faq_df)

# if "question_history" not in st.session_state:
#     st.session_state["question_history"] = []
# if "user_input" not in st.session_state:
#     st.session_state["user_input"] = ""
# if "answer" not in st.session_state:
#     st.session_state["answer"] = ""
# if "matched_question" not in st.session_state:
#     st.session_state["matched_question"] = ""

# def get_voice_input():
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()
#     with mic as source:
#         st.info("Listening... Please speak into your microphone.")
#         audio = recognizer.listen(source, phrase_time_limit=7)
#     try:
#         text = recognizer.recognize_google(audio)
#         st.success(f"You said: {text}")
#         st.session_state["user_input"] = text
#     except sr.UnknownValueError:
#         st.error("Sorry, could not understand your voice.")
#     except sr.RequestError:
#         st.error("Speech recognition service unavailable.")

# st.title("")
# st.write("Ask a medical question by typing or using your voice.")

# with st.sidebar:
#     st.header("Questions Asked")
#     if st.session_state["question_history"]:
#         for q in st.session_state["question_history"]:
#             st.write(q)
#     else:
#         st.write("No questions yet.")

# col1, col2 = st.columns([2, 1])

# with col1:
#     with st.form("ask_form", clear_on_submit=True):
#         user_input = st.text_input("Type your question", value=st.session_state["user_input"], key="user_input_form")
#         btn_col1, btn_col2 = st.columns([1, 1])
#         with btn_col1:
#             submitted = st.form_submit_button("Ask")
#         with btn_col2:
#             speak_clicked = st.form_submit_button("\U0001F3A4 Speak")
#             if speak_clicked:
#                 get_voice_input()

# with col2:
#     st.write("")  # For vertical alignment

# if 'submitted' in locals() and submitted and user_input.strip():
#     query = user_input.strip()
#     st.session_state["question_history"].append(query)
#     query_emb = model.encode([query], normalize_embeddings=True)
#     D, I = index.search(np.array(query_emb).astype('float32'), k=3)
#     best_idx = int(I[0][0])
#     best_sim = float(D[0][0])
#     THRESHOLD = 0.5
#     matched_question = faq_df.iloc[best_idx]['Question']
#     if best_sim > THRESHOLD:
#         answer = faq_df.iloc[best_idx]['Answer']
#     else:
#         answer = "Sorry, I don't have information about that topic."
#     st.session_state["answer"] = answer
#     st.session_state["matched_question"] = matched_question
#     st.session_state["user_input"] = ""

# if st.session_state.get("answer"):
#     st.markdown(f"**Matched Question:** {st.session_state['matched_question']}")
#     st.markdown(f"**Answer:** {st.session_state['answer']}")
#     if st.button("\U0001F50A Read Answer"):
#         try:
#             text_to_speech(
#                 text=st.session_state["answer"],
#                 language='en',
#                 wait=True,
#                 lag=0.01
#             )
#         except Exception as e:
#             st.error(f"Text-to-speech failed: {e}")












import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
st.set_page_config(page_title="MediBot - Ask Your Medical Questions", page_icon="🩺", layout="centered")

# --- FFmpeg configuration for pydub ---
from pydub import AudioSegment
ffmpeg_path = r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
ffprobe_path = r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path
os.environ["PATH"] += os.pathsep + r"C:\Users\yuvra\OneDrive\Desktop\ffmpeg-master-latest-win64-gpl-sh-shared\bin"
# --------------------------------------

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
import speech_recognition as sr
import json
from gtts import gTTS
import base64

DATA_FOLDER = r"C:\Users\yuvra\OneDrive\Desktop\philips.data"
MERGED_JSON = os.path.join(DATA_FOLDER, "merged_faq.json")

@st.cache_data
def load_all_faq_data():
    with open(MERGED_JSON, 'r', encoding='utf-8') as f:
        faq_list = json.load(f)
    qa_pairs = []
    for item in faq_list:
        if 'question' in item and 'answer' in item:
            qa_pairs.append({'Question': item['question'], 'Answer': item['answer']})
        elif 'intents' in item:
            for intent in item['intents']:
                patterns = intent.get('patterns', [])
                responses = intent.get('responses', [])
                for pattern in patterns:
                    for response in responses:
                        qa_pairs.append({'Question': pattern, 'Answer': response})
    faq_df = pd.DataFrame(qa_pairs)
    return faq_df

faq_df = load_all_faq_data()

@st.cache_resource
def load_model_and_index(faq_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = faq_df['Question'].astype(str).tolist()
    emb_file = os.path.join(DATA_FOLDER, 'faq_embeddings.npy')
    idx_file = os.path.join(DATA_FOLDER, 'faiss_index.bin')

    if not os.path.exists(emb_file) or len(np.load(emb_file)) != len(questions):
        embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
        np.save(emb_file, embeddings)
    else:
        embeddings = np.load(emb_file)

    dim = embeddings.shape[1]
    if not os.path.exists(idx_file) or faiss.read_index(idx_file).ntotal != len(questions):
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, idx_file)
    else:
        index = faiss.read_index(idx_file)

    return model, index, embeddings

model, index, embeddings = load_model_and_index(faq_df)

if "question_history" not in st.session_state:
    st.session_state["question_history"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "answer" not in st.session_state:
    st.session_state["answer"] = ""
if "matched_question" not in st.session_state:
    st.session_state["matched_question"] = ""

def get_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎙️ Listening... Please speak into your microphone.")
        audio = recognizer.listen(source, phrase_time_limit=7)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        st.session_state["user_input"] = text
    except sr.UnknownValueError:
        st.error(" Sorry, could not understand your voice.")
    except sr.RequestError:
        st.error(" Speech recognition service unavailable.")

def text_to_speech_fast(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_file = open(fp.name, 'rb')
        audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"<audio autoplay controls><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
        st.markdown(audio_html, unsafe_allow_html=True)

st.title("🩺 MediBot")
st.markdown("### Ask a medical question by typing or using your voice.")

with st.sidebar:
    st.header("📜 Question History")
    if st.button(" Clear History"):
        st.session_state["question_history"] = []
    if st.session_state["question_history"]:
        for q in reversed(st.session_state["question_history"]):
            st.markdown(f"- {q}")
    else:
        st.info("No questions asked yet.")

st.markdown("---")

col1, col2 = st.columns([4, 1])

user_input = ""

with col1:
    with st.form("ask_form", clear_on_submit=True):
        user_input = st.text_input(" Type your question:", value=st.session_state["user_input"], key="user_input_form")
        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn1:
            submitted = st.form_submit_button(" Ask")
        with btn2:
            speak_clicked = st.form_submit_button(" Speak")
        with btn3:
            read_clicked = st.form_submit_button(" Read Answer")
        if speak_clicked:
            get_voice_input()
            user_input = st.session_state["user_input"]

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)

if user_input.strip():
    st.session_state["question_history"].append(user_input.strip())
    query_emb = model.encode([user_input.strip()], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb).astype('float32'), k=3)
    best_idx = int(I[0][0])
    best_sim = float(D[0][0])
    THRESHOLD = 0.5
    matched_question = faq_df.iloc[best_idx]['Question']
    if best_sim > THRESHOLD:
        answer = faq_df.iloc[best_idx]['Answer']
    else:
        answer = "❓ Sorry, I don't have information about that topic."
    st.session_state["answer"] = answer
    st.session_state["matched_question"] = matched_question
    st.session_state["user_input"] = ""

if st.session_state.get("answer"):
    st.markdown(f"####  Did you mean?:")
    st.markdown(f"`{st.session_state['matched_question']}`")
    st.markdown(f"####  Answer:")
    st.success(st.session_state['answer'])
    if 'read_clicked' in locals() and read_clicked:
        try:
            text_to_speech_fast(st.session_state["answer"])
        except Exception as e:
            st.error(f"Text-to-speech failed: {e}")










           

























