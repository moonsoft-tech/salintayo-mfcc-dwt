# Build: 2026-09-13 02:43 UTC
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
    version="3.3.0",
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
    word: str                   # target word (what they should say)
    heard_word: str = ""        # what STT actually heard (from app)
    dialect_code: str = "fil"   # dialect-aware scoring

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

    # Safety check — must have at least 0.1s of audio
    min_samples = int(SAMPLE_RATE * 0.1)
    if len(y) < min_samples:
        logger.warning("Audio very short: %d samples (%.2fs)", len(y), len(y)/SAMPLE_RATE)
        # Don't reject — pad with zeros so MFCC can still run
        y = np.pad(y, (0, min_samples - len(y)))

    return y


def preprocess_audio(y: np.ndarray) -> np.ndarray:
    """
    Preprocess audio for Philippine dialect phonology:
    1. Trim silence — lenient threshold so short words aren't wiped out
    2. Pre-emphasis filter — boosts consonants (/t/, /d/, /n/, /ng/, /k/)
    3. Normalize amplitude — removes volume differences between devices
    """
    # 1. Trim silence — top_db=40 is lenient enough to keep short Filipino
    #    words intact. top_db=25 was too aggressive and wiped out gTTS audio.
    trimmed, _ = librosa.effects.trim(y, top_db=40)
    # Safety: if trim removed everything, use original audio
    if len(trimmed) < 100:
        trimmed = y

    # 2. Pre-emphasis
    y_out = np.append(trimmed[0], trimmed[1:] - 0.97 * trimmed[:-1])

    # 3. Amplitude normalization
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val

    return y_out


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


def duration_penalty(y_learner: np.ndarray, y_reference: np.ndarray) -> float:
    """
    Returns a penalty multiplier (0.0–1.0) based on how well the learner's
    audio duration matches the reference. A held vowel ("ahhhhh") against a
    short word ("Ama") should score low even if the vowel formant matches.

    Ratio = learner_duration / reference_duration
      - Close to 1.0 → no penalty (multiplier = 1.0)
      - Very short or very long → penalty applied
      - Ratio > 3.0 (held much longer) or < 0.3 (way too short) → strong penalty
    """
    dur_learner   = len(y_learner)  / SAMPLE_RATE
    dur_reference = len(y_reference) / SAMPLE_RATE

    if dur_reference < 0.05:
        return 1.0  # can't judge — reference too short

    ratio = dur_learner / dur_reference

    if 0.5 <= ratio <= 2.0:
        # Within reasonable range — no penalty
        return 1.0
    elif ratio > 2.0:
        # Learner held too long (e.g. "ahhhh" vs "Ama")
        # ratio 2.0→1.0, ratio 3.0→0.6, ratio 4.0+→0.4
        penalty = max(0.4, 1.0 - (ratio - 2.0) * 0.3)
        return penalty
    else:
        # Learner too short (ratio < 0.5)
        penalty = max(0.5, ratio / 0.5)
        return penalty
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


def distance_to_score_ph(distance: float, dialect_code: str, word: str = '') -> float:
    """
    Convert DTW distance to 0-100 score matching the app's five score tiers.
    Calibrated against real observed distances (2026-09-14 session):
      - Single words: effective range ~58 (best) to ~189 (worst)
      - Multi-word phrases need stricter thresholds — more syllables means
        more chance for a wrong phrase to coincidentally align with parts of
        the target, inflating the score unfairly.

    Thresholds tighten proportionally with word count so a completely wrong
    3-word phrase can't score 88-90% by accidentally matching stray syllables.

    Tiers (after dialect tolerance + word-count adjustment):
      100        — near-perfect acoustic match
      70 – 99    — almost there, very close
      40 – 69    — still far, needs more practice
      15 – 39    — quite bad
      0  – 14    — very bad
    """
    tolerance = DIALECT_TOLERANCE.get(dialect_code, 1.0)
    d = distance / tolerance  # effective distance after dialect leniency

    # Word-count tightening: each additional word beyond the first reduces
    # thresholds by 12%, so a 3-word phrase is ~24% stricter than a single word.
    # This prevents wrong multi-word phrases from scoring high via coincidental
    # syllable alignment. Single words (count=1) get no adjustment (factor=1.0).
    word_count = max(1, len((word or '').strip().split()))
    tightening = 1.0 - (word_count - 1) * 0.12  # 1 word→1.0, 2→0.88, 3→0.76, 4→0.64
    tightening = max(0.55, tightening)           # floor at 0.55 for very long phrases

    perfect   = 100  * tightening   # single: 100,  2-word: 88,  3-word: 76
    near_end  = 140  * tightening   # single: 140,  2-word: 123, 3-word: 106
    far_end   = 175  * tightening   # single: 175,  2-word: 154, 3-word: 133
    wrong_end = 250  * tightening   # single: 250,  2-word: 220, 3-word: 190

    if d < perfect:
        return 100.0
    elif d < near_end:
        # Near band: perfect → 99, near_end → 70
        return float(np.clip(99.0 - (d - perfect) / (near_end - perfect) * 29.0, 70.0, 99.0))
    elif d < far_end:
        # Far band: near_end → 69, far_end → 40
        return float(np.clip(69.0 - (d - near_end) / (far_end - near_end) * 29.0, 40.0, 69.0))
    elif d < wrong_end:
        # Bad band: far_end → 39, wrong_end → 15
        return float(np.clip(39.0 - (d - far_end) / (wrong_end - far_end) * 24.0, 15.0, 39.0))
    else:
        # Very bad band: wrong_end → 14, tails to 0
        return float(np.clip(14.0 - (d - wrong_end) / 100.0 * 14.0, 0.0, 14.0))


def score_to_feedback_ph(score: float, word: str, dialect_code: str) -> str:
    """Feedback messages matching the app's five score tiers."""
    if score == 100:
        return f"Perfect! Your pronunciation of '{word}' is spot on. Great job! 🎉"
    elif score >= 70:
        return f"Almost there! You're very close on '{word}' — keep practicing and you'll get it. 💪"
    elif score >= 40:
        return f"Not quite yet. Focus on each syllable of '{word}' and try listening to it again before your next attempt."
    elif score >= 15:
        return f"Keep going — pronunciation takes practice. Listen carefully to '{word}' and give it another shot."
    else:
        return f"Don't give up! Try listening to '{word}' a few more times and speak slowly, one syllable at a time."


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SalinTayo Pronunciation Scorer",
        "status": "ok",
        "version": "3.3.0",
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

    # 4. Dialect-aware score with word-count tightening
    score = distance_to_score_ph(dist, dialect, body.word)

    # 5. Duration penalty — catches held vowels / silent gaps scored against
    #    short words (e.g. "ahhh" vs "Ama" should not score 100%).
    d_penalty = duration_penalty(y_learner, y_reference)
    if d_penalty < 1.0:
        logger.info("Duration penalty applied: ratio=%.2f penalty=%.2f",
                    (len(y_learner) / SAMPLE_RATE) / max(0.05, len(y_reference) / SAMPLE_RATE),
                    d_penalty)
    score = score * d_penalty

    # 5. Smart floor — only boost score when STT confirmed the right word
    #    was heard AND the acoustic distance is tight.
    #    Wrong word said = no floor = honest low acoustic score.
    #    Threshold tightened: word_correct requires very close string match
    #    (not just substring) AND tight acoustic distance (< 200, not 350)
    #    to prevent near-misses from getting an artificial score boost.
    # Score comes purely from acoustic DTW distance — no word_correct
    # floor or cap. The DTW bands already encode the full tier system:
    # a wrong word acoustically far from the reference scores low naturally,
    # and a close pronunciation scores high regardless of what STT heard.
    # Adding a text-based cap on top of acoustic scoring punishes users
    # when Whisper mishears (e.g. "Magkanu" → heard as "Magkano" or something
    # else) even though their audio was genuinely close.
    feedback = score_to_feedback_ph(score, body.word, dialect)
    logger.info("Score: %.1f — %s", score, feedback)

    logger.info("RAW dist=%.4f effective=%.4f words=%d score=%.1f",
                dist, dist / DIALECT_TOLERANCE.get(dialect, 1.0), len((body.word or '').strip().split()), score)

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