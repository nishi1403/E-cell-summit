# ==========================================================
# Deepfake Audio Analyzer (Streamlit, CPU-friendly)
# - Upload an audio file
# - Real/Fake + Gender from your .pth model
# - Duration
# - Transcription (Whisper via Transformers, CPU by default)
# - Word Cloud
# - Waveform + audio playback
# ==========================================================

import os
import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Deepfake Audio Analyzer",
    page_icon="🎧",
    layout="wide"
)

# ---------------------------
# Utilities
# ---------------------------
def bytes_to_tempfile(uploaded_file) -> str:
    """Save uploaded file to a NamedTemporaryFile and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def load_audio_mono_16k(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    duration = librosa.get_duration(y=y, sr=target_sr)
    return y, target_sr, duration

def extract_mfcc_mean(y, sr, n_mfcc=40, max_duration=2.0):
    """
    Pad/truncate to max_duration, then compute MFCC and take mean across time.
    Returns a 1D torch.FloatTensor of shape [n_mfcc].
    """
    max_len = int(sr * max_duration)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape (n_mfcc,)
    return torch.tensor(mfcc_mean, dtype=torch.float32)

def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    """Return softmax probabilities as numpy array."""
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    times = np.arange(len(y)) / sr
    ax.plot(times, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amp")
    ax.set_title(title)
    ax.margins(x=0)
    st.pyplot(fig, clear_figure=True)

def show_wordcloud(text: str):
    if not text.strip():
        st.info("No transcript text to render word cloud.")
        return
    wc = WordCloud(width=1000, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Transcribed Audio")
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Your trained model (same shape as shared code)
# ---------------------------
class MyModel(nn.Module):
    def __init__(self, num_classes=2, num_genders=2):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(40, 64)   # MFCC (40 features)
        self.fc_real_fake = nn.Linear(64, num_classes)  # Real/Fake
        self.fc_gender = nn.Linear(64, num_genders)     # Male/Female

    def forward(self, x):
        # x: [batch, 40]
        x = F.relu(self.fc1(x))
        real_fake_out = self.fc_real_fake(x)
        gender_out = self.fc_gender(x)
        return real_fake_out, gender_out

# ---------------------------
# Caching: load model & ASR only once
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_deepfake_model(model_path: str, device: str = "cpu"):
    model = MyModel().to(device)
    # torch.load on CPU
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Be lenient if keys differ slightly
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=True)
def load_asr_pipeline(model_name: str = "openai/whisper-tiny", device: int = -1):
    # device = -1 -> CPU; 0 -> first GPU if available
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=device
    )

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("⚙️ Settings")

# Device selection (default CPU)
cuda_available = torch.cuda.is_available()
device_choice = st.sidebar.selectbox(
    "Inference device",
    options=["cpu"] + (["cuda"] if cuda_available else []),
    index=0
)
device = "cuda" if (device_choice == "cuda" and cuda_available) else "cpu"

# Path to your .pth
default_model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(default_model_dir, exist_ok=True)
st.sidebar.markdown("**Deepfake Model (.pth)**")
pth_source = st.sidebar.radio(
    "Choose source",
    options=["Browse file", "Use path string"],
    index=0
)
if pth_source == "Browse file":
    uploaded_pth = st.sidebar.file_uploader("Upload your .pth", type=["pth", "pt"])
    model_path = None
    if uploaded_pth:
        # Save to temp file to use torch.load
        tmp_pth = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp_pth.write(uploaded_pth.getvalue())
        tmp_pth.close()
        model_path = tmp_pth.name
else:
    model_path = st.sidebar.text_input(
        "Absolute or relative path to .pth",
        os.path.join(default_model_dir, "ecapa_bilstm_transformer_final.pth")
    )
    if not os.path.isfile(model_path):
        st.sidebar.info("Enter a valid path to your saved model (.pth).")

# Whisper model size (tiny/base/small are best for CPU)
asr_size = st.sidebar.selectbox(
    "ASR model",
    options=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
    index=0
)
st.sidebar.caption("Tip: *tiny* is fastest on CPU, *small* is better quality but slower.")

max_duration = st.sidebar.slider("Deepfake model window (seconds)", 1.0, 5.0, 2.0, 0.5)
sr_target = 16000

st.title("🎧 Eco Shield- Audio Deepfake Detection")
st.caption("Upload an audio file to detect deepfake, estimate speaker gender, get duration, transcribe speech, and visualize a word cloud.")

# ---------------------------
# Upload audio
# ---------------------------
uploaded_audio = st.file_uploader(
    "Upload audio (wav/mp3/m4a/ogg/flac)",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

# ---------------------------
# Main action
# ---------------------------
if uploaded_audio is not None:
    # Save uploaded audio to a temp file
    audio_path = bytes_to_tempfile(uploaded_audio)

    # Load audio mono 16k
    try:
        y, sr, duration = load_audio_mono_16k(audio_path, target_sr=sr_target)
    except Exception as e:
        st.error(f"Failed to read audio: {e}")
        st.stop()

    # Preview UI
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("▶️ Player")
        st.audio(uploaded_audio, format=uploaded_audio.type)
        st.caption(f"Filename: {uploaded_audio.name}")

    with col_b:
        st.subheader("Waveform")
        plot_waveform(y, sr, title="Waveform (mono, 16 kHz)")

    # Prepare features for deepfake/gender model
    feats = extract_mfcc_mean(y, sr, n_mfcc=40, max_duration=max_duration).unsqueeze(0)  # [1, 40]

    # Load deepfake model once
    deepfake_model = None
    if model_path and os.path.isfile(model_path):
        try:
            with st.spinner("Loading deepfake model..."):
                deepfake_model = load_deepfake_model(model_path, device=device)
        except Exception as e:
            st.error(f"Could not load model: {e}")
    else:
        st.warning("Please provide a valid .pth to run deepfake detection.")

    # Run deepfake + gender inference
    rf_label = None
    gender_label = None
    rf_probs = None
    gender_probs = None

    if deepfake_model is not None:
        with torch.no_grad():
            x = feats.to(device)
            real_fake_logits, gender_logits = deepfake_model(x)
            rf_probs = softmax_probs(real_fake_logits)[0]      # [2]
            gender_probs = softmax_probs(gender_logits)[0]     # [2]

            rf_idx = int(np.argmax(rf_probs))
            gender_idx = int(np.argmax(gender_probs))

            rf_label = "Real" if rf_idx == 1 else "Fake"
            gender_label = "Male" if gender_idx == 1 else "Female"

    # Show metrics (left) and probs (right)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration (s)", f"{duration:.2f}")
    with col2:
        st.metric("Real / Fake", rf_label if rf_label else "—")
        if rf_probs is not None:
            st.progress(float(rf_probs[1]))  # Real prob bar
            st.caption(f"P(Real)={rf_probs[1]:.2f} | P(Fake)={rf_probs[0]:.2f}")
    with col3:
        st.metric("Gender", gender_label if gender_label else "—")
        if gender_probs is not None:
            st.progress(float(gender_probs[1]))  # Male prob bar
            st.caption(f"P(Male)={gender_probs[1]:.2f} | P(Female)={gender_probs[0]:.2f}")

    st.divider()

    # ---------------------------
    # Transcription
    # ---------------------------
    st.subheader("📝 Transcription")

    # Whisper pipeline (CPU by default)
    use_gpu = (device == "cuda")
    asr_device = 0 if use_gpu else -1
    try:
        with st.spinner(f"Loading ASR model ({asr_size})..."):
            asr = load_asr_pipeline(model_name=asr_size, device=asr_device)
        # Whisper expects either raw array or path; we pass the raw float32 array (16k)
        with st.spinner("Transcribing..."):
            result = asr(y)  # <-- Remove sampling_rate argument
            transcript = result.get("text", "").strip()
    except Exception as e:
        st.error(f"ASR failed: {e}")
        transcript = ""

    st.text_area("Transcript", transcript, height=150)

    # ---------------------------
    # Word Cloud
    # ---------------------------
    st.subheader("☁️ Word Cloud")
    show_wordcloud(transcript)

    # Cleanup temp audio when session ends (optional)
    # os.remove(audio_path)

else:
    st.info("Upload an audio file to get started.")

# ---------------------------
# Footer / Tips
# ---------------------------
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
- The deepfake model uses 40-MFCC mean features over a **configurable window** (default 2s).  
- For **CPU-only machines**, prefer **`openai/whisper-tiny`** or **`openai/whisper-base`** for faster transcription.  
- If your trained model expects exactly 2 seconds, keep the **window at 2.0s** (slider in the sidebar). Longer uploads are truncated/padded to this window for the classifier.  
- For best accuracy on long files, you can extend the app to segment the audio and aggregate predictions (majority vote or mean probs).
        """
    )
