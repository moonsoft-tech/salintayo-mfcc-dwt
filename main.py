"""
SalinTayo Pronunciation Scoring Server
---------------------------------------
Scores a learner's pronunciation against a reference using:
  - MFCC  (Mel-Frequency Cepstral Coefficients) via librosa
  - DTW   (Dynamic Time Warping) via fastdtw

POST /score/pronunciation
  Body: { audio_base64: str, reference_base64: str, word: str }
  Returns: { score: float, feedback: str, dtw_distance: float }

POST /reference/generate
  Body: { word: str, language_code: str }
  Returns: { reference_base64: str, word: str }
  Uses gTTS to generate a clean reference pronunciation on the fly.
"""

import base64
import io
import logging
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastdtw import fastdtw
from gtts import gTTS
from pydantic import BaseModel
from scipy.spatial.distance import euclidean

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("salintayo-scorer")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SalinTayo Pronunciation Scorer",
    description="MFCC + DTW pronunciation scoring API for SalinTayo language learning app.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050          # librosa default
N_MFCC      = 13             # number of MFCC coefficients
HOP_LENGTH  = 512
MAX_AUDIO_BYTES = 5_000_000  # 5 MB decoded max

# ── Language code → gTTS lang mapping ─────────────────────────────────────────
GTTS_LANG_MAP = {
    "fil": "tl",   # Filipino / Tagalog
    "ceb": "tl",   # Cebuano — gTTS has no Cebuano; Tagalog is closest
    "ilo": "tl",   # Ilocano — same fallback
    "hil": "tl",   # Hiligaynon
    "war": "tl",   # Waray
    "bik": "tl",   # Bikol
    "pam": "tl",   # Kapampangan
    "tsg": "tl",   # Tausug
    "pag": "tl",   # Pangasinense
    "en":  "en",
}

# ── Pydantic models ────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    audio_base64: str        # learner's recorded audio (base64, any common format)
    reference_base64: str    # reference pronunciation (base64)
    word: str                # the target word (for logging / feedback)

class ScoreResponse(BaseModel):
    score: float             # 0–100
    feedback: str            # human-readable feedback string
    dtw_distance: float      # raw DTW distance (lower = better)

class ReferenceRequest(BaseModel):
    word: str
    language_code: str = "fil"

class ReferenceResponse(BaseModel):
    reference_base64: str
    word: str

# ── Helpers ────────────────────────────────────────────────────────────────────
def decode_audio(b64: str) -> np.ndarray:
    """Decode a base64 audio string to a mono float32 numpy array at SAMPLE_RATE."""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data.")

    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (max 5 MB decoded).")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")
    finally:
        os.unlink(tmp_path)

    if len(y) == 0:
        raise HTTPException(status_code=422, detail="Audio file is empty or silent.")

    return y


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """Return MFCC matrix shape (n_mfcc, T) — each column is one frame."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    )
    # Delta and delta-delta for richer representation
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])  # shape (39, T)


def dtw_distance(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """Compute DTW distance between two MFCC matrices (columns are frames)."""
    # fastdtw expects sequences of vectors — transpose so shape is (T, features)
    seq_a = mfcc_a.T
    seq_b = mfcc_b.T
    distance, _ = fastdtw(seq_a, seq_b, dist=euclidean)
    # Normalize by the length of the longer sequence so short vs long is fair
    norm = max(len(seq_a), len(seq_b))
    return float(distance / norm) if norm > 0 else float(distance)


def distance_to_score(distance: float) -> float:
    """
    Convert a normalized DTW distance to a 0–100 score.
    Empirically tuned thresholds:
      distance ~  0  → score 100  (perfect)
      distance ~ 50  → score  50  (acceptable)
      distance ~ 150 → score   0  (very different)
    Uses an exponential decay so small improvements matter most.
    """
    score = 100.0 * np.exp(-distance / 60.0)
    return float(np.clip(score, 0.0, 100.0))


def score_to_feedback(score: float, word: str) -> str:
    if score >= 85:
        return f"Excellent! Your pronunciation of '{word}' is very accurate."
    elif score >= 70:
        return f"Good job! Your pronunciation of '{word}' is close — keep practicing."
    elif score >= 50:
        return f"Not bad! Try to match the rhythm and stress of '{word}' more closely."
    else:
        return f"Keep practicing '{word}'. Listen to the reference and try to mirror it carefully."


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SalinTayo Pronunciation Scorer",
        "status": "ok",
        "endpoints": ["/score/pronunciation", "/reference/generate"],
    }


@app.post("/score/pronunciation", response_model=ScoreResponse)
def score_pronunciation(body: ScoreRequest):
    """
    Compare learner audio against a reference pronunciation using MFCC + DTW.
    Both audio fields must be base64-encoded audio (WAV, WebM, MP3, M4A, etc.).
    """
    logger.info("Scoring pronunciation for word: '%s'", body.word)

    # 1. Decode audio
    y_learner   = decode_audio(body.audio_base64)
    y_reference = decode_audio(body.reference_base64)

    # 2. Extract MFCCs (with delta + delta-delta)
    mfcc_learner   = extract_mfcc(y_learner)
    mfcc_reference = extract_mfcc(y_reference)

    # 3. DTW alignment
    dist = dtw_distance(mfcc_learner, mfcc_reference)
    logger.info("DTW distance (normalized): %.4f", dist)

    # 4. Convert to score
    score    = distance_to_score(dist)
    feedback = score_to_feedback(score, body.word)
    logger.info("Score: %.1f — %s", score, feedback)

    return ScoreResponse(score=score, feedback=feedback, dtw_distance=dist)


@app.post("/reference/generate", response_model=ReferenceResponse)
def generate_reference(body: ReferenceRequest):
    """
    Generate a TTS reference pronunciation for a given word using gTTS.
    Returns the audio as base64 so the app can cache it and reuse it
    as the `reference_base64` in /score/pronunciation calls.
    """
    lang = GTTS_LANG_MAP.get(body.language_code, "tl")
    logger.info("Generating reference for '%s' (lang=%s → gtts=%s)", body.word, body.language_code, lang)

    try:
        tts = gTTS(text=body.word, lang=lang, slow=True)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {e}")

    return ReferenceResponse(reference_base64=b64, word=body.word)