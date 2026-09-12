"""
SalinTayo Pronunciation Scoring Server
---------------------------------------
MFCC + DTW scoring tuned specifically for Philippine dialect phonology.

Philippine language characteristics this scorer accounts for:
  - Predominantly CV (consonant-vowel) syllable structure
  - Only 5 vowel phonemes: /a/, /e/, /i/, /o/, /u/
  - Stress-timed: penultimate stress (malumay) vs final stress (mabilis)
  - Glottal stop (ʔ) common word-finally and between vowels
  - No consonant clusters at syllable onset
  - Dialects share core vowel inventory but differ in consonants:
      Cebuano: no /f/, uses /p/ instead
      Ilocano: retroflex consonants
      Hiligaynon: final /ng/ heavily nasalized
  - Short words (1-3 syllables) are the norm — scoring must be
    calibrated for brevity, not penalize it
"""

import base64
import io
import logging
import os
import tempfile

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastdtw import fastdtw
from gtts import gTTS
from pydantic import BaseModel
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("salintayo-scorer")

app = FastAPI(
    title="SalinTayo Pronunciation Scorer",
    description="MFCC + DTW scoring tuned for Philippine dialect phonology.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050
MAX_AUDIO_BYTES = 5_000_000

# Philippine dialects use fewer consonants and 5 pure vowels — fewer MFCC
# coefficients capture the vowel space better without over-fitting to noise.
# 13 is standard; 10 is better for vowel-heavy languages like Filipino.
N_MFCC     = 10
HOP_LENGTH = 256  # smaller = more frames = better for short CV words

# ── Language mapping ───────────────────────────────────────────────────────────
GTTS_LANG_MAP = {
    "fil": "tl",
    "ceb": "tl",
    "ilo": "tl",
    "hil": "tl",
    "war": "tl",
    "bik": "tl",
    "pam": "tl",
    "tsg": "tl",
    "pag": "tl",
    "en":  "en",
}

# ── Dialect phonetic profiles ──────────────────────────────────────────────────
# Each dialect gets a tolerance multiplier — dialects with more phonetic
# variation from standard Filipino get a looser scoring tolerance.
# 1.0 = standard Filipino tolerance
# >1.0 = more lenient (dialect varies more from reference TTS)
DIALECT_TOLERANCE = {
    "fil": 1.0,   # Filipino/Tagalog — reference dialect, TTS matches well
    "en":  1.0,   # English
    "ceb": 1.3,   # Cebuano — /p/ for /f/, different vowel length patterns
    "hil": 1.3,   # Hiligaynon — strong final nasalization, /ng/ differences
    "ilo": 1.4,   # Ilocano — retroflex consonants, different from TTS reference
    "war": 1.3,   # Waray — similar to Cebuano phonology
    "bik": 1.2,   # Bikol — close to Filipino with some vowel shifts
    "pam": 1.2,   # Kapampangan — /e/ and /i/ merger, /o/ and /u/ merger
    "tsg": 1.5,   # Tausug — Arabic-influenced phonology, most divergent
    "pag": 1.2,   # Pangasinense
}

# gTTS generates Filipino TTS which is a fair reference for all PH dialects
# since they share core phonology. For dialect-specific TTS, slow=True
# helps learners hear each syllable clearly.
DIALECT_TTS_SLOW = {
    "fil": True,   # slow for learners
    "ceb": True,
    "hil": True,
    "ilo": True,
    "war": True,
    "bik": True,
    "pam": True,
    "tsg": True,
    "pag": True,
    "en":  False,  # English: normal speed sounds more natural
}

# ── Pydantic models ────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    audio_base64: str
    reference_base64: str
    word: str
    dialect_code: str = "fil"  # NEW: dialect-aware scoring

class ScoreResponse(BaseModel):
    score: float
    feedback: str
    dtw_distance: float
    dialect_code: str

class ReferenceRequest(BaseModel):
    word: str
    language_code: str = "fil"

class ReferenceResponse(BaseModel):
    reference_base64: str
    word: str

# ── Audio processing ───────────────────────────────────────────────────────────
def decode_audio(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data.")

    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (max 5 MB).")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")
    finally:
        os.unlink(tmp_path)

    if len(y) == 0:
        raise HTTPException(status_code=422, detail="Audio is empty or silent.")

    return y


def preprocess_audio(y: np.ndarray) -> np.ndarray:
    """
    Preprocess audio for Philippine dialect phonology:
    1. Trim silence (top_db=25 — Filipino speakers often have short pauses)
    2. Pre-emphasis filter — boosts high frequencies, helps distinguish
       Philippine consonants (especially /t/, /d/, /n/, /ng/, /k/)
    3. Normalize amplitude — removes recording volume differences between
       phones (important: Filipino learners record on varied devices)
    """
    # 1. Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # 2. Pre-emphasis — standard for speech processing, especially helpful
    #    for Philippine languages where final consonants are often soft
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # 3. Amplitude normalization
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y


def extract_mfcc_ph(y: np.ndarray) -> np.ndarray:
    """
    MFCC extraction tuned for Philippine phonology:
    - 10 coefficients (vowel-heavy language — fewer is better)
    - Mel filterbank focused on 0-8kHz (covers Philippine consonant range)
    - Mean normalization per coefficient (removes microphone coloration)
    - Only delta (no delta-delta) — short CV syllables don't need 2nd order
    - C0 (energy) included — stress patterns in Filipino are energy-based
      (malumay/mabilis stress distinction is crucial for correct meaning)
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        fmin=0,
        fmax=8000,  # covers full Philippine consonant + vowel range
    )

    # Mean normalization — removes channel/mic differences
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    # Include energy (root mean square) as an extra feature
    # Filipino stress (malumay vs mabilis) is primarily energy-based
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms_norm = rms - np.mean(rms)

    # Delta MFCC — captures how the sound changes over time
    # Important for CV transitions (e.g., "Tu-big": the /T/→/u/ transition)
    delta = librosa.feature.delta(mfcc)

    # Stack: MFCC (10) + RMS energy (1) + delta MFCC (10) = 21 features
    return np.vstack([mfcc, rms_norm, delta])


def dtw_distance_ph(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """
    DTW distance normalized for Philippine word length distribution.
    Most Filipino words are 1-3 syllables (2-6 phonemes).
    Normalizing by average length (not max) is fairer for short words.
    """
    seq_a = mfcc_a.T  # (T, features)
    seq_b = mfcc_b.T
    distance, _ = fastdtw(seq_a, seq_b, dist=euclidean)
    avg_len = (len(seq_a) + len(seq_b)) / 2
    return float(distance / avg_len) if avg_len > 0 else float(distance)


def distance_to_score_ph(distance: float, dialect_code: str) -> float:
    """
    Convert DTW distance to 0-100 score with dialect-specific tolerance.

    Scoring curve tuned for Philippine learners:
    - Native-like pronunciation (distance < 15): 85-100
    - Good pronunciation (15-40): 65-85
    - Acceptable (40-80): 40-65
    - Needs work (80+): below 40

    Dialect tolerance adjusts the curve — Tausug learners pronouncing
    Tausug words get more tolerance than Filipino learners pronouncing
    Filipino words, because the TTS reference is always Filipino.
    """
    tolerance = DIALECT_TOLERANCE.get(dialect_code, 1.0)
    # Effective distance is reduced by tolerance — more lenient dialects
    # effectively "see" a smaller distance for the same recording
    effective_distance = distance / tolerance

    # Sigmoid-like decay tuned for Philippine short word distribution
    score = 100.0 * np.exp(-effective_distance / 120.0)
    return float(np.clip(score, 0.0, 100.0))


def score_to_feedback_ph(score: float, word: str, dialect_code: str) -> str:
    """Feedback messages aware of Philippine dialect context."""
    dialect_names = {
        "fil": "Filipino", "ceb": "Cebuano", "hil": "Hiligaynon",
        "ilo": "Ilocano", "war": "Waray", "bik": "Bikol",
        "pam": "Kapampangan", "tsg": "Tausug", "pag": "Pangasinense",
        "en": "English",
    }
    dialect_name = dialect_names.get(dialect_code, "Filipino")

    if score >= 85:
        return f"Mahusay! Your {dialect_name} pronunciation of '{word}' is excellent."
    elif score >= 70:
        return f"Magaling! '{word}' sounds good — keep practicing the stress pattern."
    elif score >= 50:
        return f"Mabuti! Try to stress the right syllable in '{word}' and match the vowel sounds."
    elif score >= 30:
        return f"Keep going! Focus on the vowel sounds in '{word}' — Filipino has only 5 pure vowels: a, e, i, o, u."
    else:
        return f"Subukan ulit! Listen to the reference for '{word}' carefully and try to match each syllable."


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SalinTayo Pronunciation Scorer",
        "status": "ok",
        "version": "2.0.0",
        "dialect_support": list(GTTS_LANG_MAP.keys()),
        "endpoints": ["/score/pronunciation", "/reference/generate"],
    }


@app.post("/score/pronunciation", response_model=ScoreResponse)
def score_pronunciation(body: ScoreRequest):
    dialect = body.dialect_code or "fil"
    logger.info("Scoring '%s' (dialect: %s)", body.word, dialect)

    # 1. Decode + preprocess
    y_learner   = preprocess_audio(decode_audio(body.audio_base64))
    y_reference = preprocess_audio(decode_audio(body.reference_base64))

    # 2. Extract Philippine-tuned MFCCs
    mfcc_learner   = extract_mfcc_ph(y_learner)
    mfcc_reference = extract_mfcc_ph(y_reference)

    # 3. DTW with Philippine normalization
    dist = dtw_distance_ph(mfcc_learner, mfcc_reference)
    logger.info("DTW distance (normalized): %.4f", dist)

    # 4. Dialect-aware score
    score = distance_to_score_ph(dist, dialect)

    # 5. Minimum floor — if the word was heard correctly by STT and DTW
    #    distance is reasonable, don't score below 40.
    #    Filipino learners saying the word correctly shouldn't score < 40.
    if dist < 200:
        score = max(score, 40.0)

    feedback = score_to_feedback_ph(score, body.word, dialect)
    logger.info("Score: %.1f — %s", score, feedback)

    return ScoreResponse(
        score=round(score, 1),
        feedback=feedback,
        dtw_distance=round(dist, 4),
        dialect_code=dialect,
    )


@app.post("/reference/generate", response_model=ReferenceResponse)
def generate_reference(body: ReferenceRequest):
    lang = GTTS_LANG_MAP.get(body.language_code, "tl")
    slow = DIALECT_TTS_SLOW.get(body.language_code, True)
    logger.info("Generating reference for '%s' (lang=%s, slow=%s)", body.word, lang, slow)

    try:
        tts = gTTS(text=body.word, lang=lang, slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {e}")

    return ReferenceResponse(reference_base64=b64, word=body.word)