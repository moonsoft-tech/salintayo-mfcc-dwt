# SalinTayo Pronunciation Scorer

MFCC + DTW pronunciation scoring server for the SalinTayo language learning app.

## Endpoints

### `POST /score/pronunciation`
Scores a learner's pronunciation against a reference audio.

**Request:**
```json
{
  "audio_base64": "<base64 encoded audio>",
  "reference_base64": "<base64 encoded reference audio>",
  "word": "salamat"
}
```

**Response:**
```json
{
  "score": 87.4,
  "feedback": "Excellent! Your pronunciation of 'salamat' is very accurate.",
  "dtw_distance": 12.3
}
```

### `POST /reference/generate`
Generates a TTS reference pronunciation for a word.

**Request:**
```json
{
  "word": "salamat",
  "language_code": "fil"
}
```

**Response:**
```json
{
  "reference_base64": "<base64 encoded audio>",
  "word": "salamat"
}
```

## Supported Language Codes
| Code | Language |
|------|----------|
| `fil` | Filipino |
| `ceb` | Cebuano |
| `ilo` | Ilocano |
| `hil` | Hiligaynon |
| `war` | Waray |
| `bik` | Bikol |
| `pam` | Kapampangan |
| `tsg` | Tausug |
| `en`  | English |

## Deploy to Railway
1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo — Railway auto-detects the Dockerfile
4. Copy the generated URL and set it as `VITE_SCORER_URL` in your app

## Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
API runs at `http://localhost:8000`