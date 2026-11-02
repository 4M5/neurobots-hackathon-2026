# Exam Behavior Anomaly Detection

An IIT Palakkad Hackathon prototype for running an online exam and collecting a limited set of browser behavior signals. An Isolation Forest can flag an aggregate behavior summary for human review. A model flag or a rule warning is **not proof of cheating** and must not be used for automatic punishment.

## Implemented architecture

```text
Browser (HTML/CSS/JavaScript)
  ├─ camera permission + MediaPipe Face Mesh preview
  ├─ browser focus, visibility, paste, typing, and timer events
  ├─ question navigation, review, and optional post-exam speech recognition
  └─ same-origin HTTP requests
       └─ Flask (server.py)
            ├─ Gemini question generation with validated offline fallback
            ├─ /api/predict → ml_service.py → saved scaler + Isolation Forest
            └─ exam session APIs → exam_storage.py → local SQLite
```

The frontend asks the Flask app for five questions. A Gemini key enables generation; otherwise the fixed fallback questions are returned. Exam start creates and activates a server-side session. Submission saves answers and proctoring counts using a per-session bearer token. Oral verification responses are saved separately after verification. The database contains student identifiers and answer data; see [`exam_hckathon - Replica/SECURITY.md`](exam_hckathon%20-%20Replica/SECURITY.md) before using real student records.

The browser records camera frames for face tracking and browser events such as tab changes, focus loss, paste counts, keystroke intervals, and idle intervals. Audio activity is only a volume indicator. Speech recognition is used for optional oral verification after the written exam. The five-warning threshold flags manual review and does not terminate the exam.

## Isolation Forest pipeline

The current generator creates **synthetic** normal-profile and anomaly-profile samples. These are hand-authored patterns, not observations or ground-truth labels from students.

`train_model.py` uses a seeded, stratified 70/15/15 train/validation/test split. It fits `StandardScaler` and Isolation Forest on normal-labeled rows from the training partition only. It selects a decision-score threshold by maximizing F1 on the validation partition, with deterministic tie-breaking. The test partition remains separate and is evaluated once with that fixed threshold. The saved metadata records the seed, source row indices, feature schema, scaler/model settings, software versions, threshold, and validation/test metrics. `test_model.py` verifies the saved test rows are disjoint from train and validation before recomputing test metrics.

Metrics describe only the supplied dataset and its split. Synthetic results do not estimate performance on real students. The validation-selected threshold may not transfer to another class, exam, or population. Isolation Forest detects deviation from a learned distribution; it does not determine intent or explain why a behavior occurred.

## Implemented and planned

### Implemented

- Flask app serving the frontend and JSON APIs.
- Gemini question generation with schema and length validation plus fixed fallback questions.
- Browser exam flow with setup checks, Face Mesh preview, rule-based event logging, question review, timer, written submission, and optional oral verification.
- Five-warning manual-review flag; the warning threshold does not automatically terminate the exam.
- SQLite-backed exam sessions with server-side student-ID validation, bearer-token authentication, state validation, and idempotent written submission.
- Isolation Forest training and inference using the same ordered 12-feature schema and the saved preprocessing scaler.
- Reproducible synthetic-data splits, validation threshold calibration, held-out test metrics, and model metadata.

### Not implemented / not established

- Verified cheating detection, real-world accuracy, or validation on real student data.
- Automatic disciplinary decisions or automatic termination based on model predictions.
- A teacher/admin dashboard or API for reviewing stored submissions.
- Encryption at rest for SQLite data, a retention/deletion workflow, or production identity management.
- A production-hardened deployment. Use the security notes and review privacy requirements before any school pilot.

## Repository structure

```text
.
├── README.md
└── exam_hckathon - Replica/
    ├── server.py                    # Flask routes, app config, security headers
    ├── exam_storage.py              # authenticated SQLite session persistence
    ├── ml_service.py                # model loading, feature schema, inference
    ├── generate_dataset.py          # reproducible synthetic data generation
    ├── train_model.py               # split, fit, threshold calibration, evaluation
    ├── test_model.py                # held-out evaluation and optional interactive demo
    ├── index.html                   # exam UI
    ├── app.js                       # browser exam lifecycle and API calls
    ├── exam-lifecycle.js            # timer, navigation, and feature helpers
    ├── styles.css
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── .env.example
    ├── .gitignore
    └── tests/
```

Generated files such as `behavior_dataset.csv`, `behavior_model.pkl`, split CSVs, evaluation JSON, and the SQLite database are ignored by Git. Recreate them locally with the commands below.

## Setup

Python 3.10 or newer is recommended. Run commands from the `exam_hckathon - Replica` directory.

### Windows PowerShell

```powershell
cd "exam_hckathon - Replica"
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, allow it for this terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
cd "exam_hckathon - Replica"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
```

### Configure and create the model

Edit `.env` only if needed. Set `GEMINI_API_KEY` to enable Gemini question generation; leaving it unset uses the fallback question set. The default model and database paths are local to this directory. Keep `.env` private.

```bash
python generate_dataset.py --seed 42 --output behavior_dataset.csv
python train_model.py --dataset behavior_dataset.csv --output-dir . --seed 42 --dataset-kind synthetic
python test_model.py
```

`test_model.py` evaluates `behavior_test_split.csv`, which is emitted by training. It reports precision, recall, F1, and the confusion matrix; it does not report a full-dataset or training-set score as test performance. `--interactive` optionally scores one manually entered example and is not an evaluation metric.

The saved `behavior_model.pkl` is required for `/api/predict`. Without it the Flask app remains available, but predictions return a model-unavailable response. `joblib` artifacts use pickle serialization: load only artifacts produced by a trusted training run.

### Start the app

```bash
python server.py
```

Open <http://127.0.0.1:8000/>. The default development server binds to loopback and debug mode is off. The exam APIs require the page to be served by Flask or a same-origin reverse proxy; a static-only server cannot create or persist exam sessions. Production deployment instructions and data caveats are in [`SECURITY.md`](exam_hckathon%20-%20Replica/SECURITY.md).

## API

All endpoints are served from the Flask origin. JSON API responses are not cached. Cross-origin clients must configure an exact comma-separated `CORS_ORIGINS` allowlist in `.env`; same-origin use needs no CORS entry.

### `GET /api/questions`

Returns five `{id, text, topic}` questions. Invalid or unavailable Gemini responses fall back to the fixed question list.

### `POST /api/predict`

Scores a numeric behavior summary. It does not accept answers, student identity, or transcripts.

```json
{
  "features": {
    "avg_key_interval": 0.2,
    "std_key_interval": 0.05,
    "typing_speed": 250,
    "backspace_rate": 0.08,
    "paste_count": 0,
    "tab_switch": 1,
    "focus_loss": 0,
    "avg_idle": 3.0,
    "max_idle": 8.0,
    "edit_count": 2,
    "time_to_first_key": 5.0,
    "answer_duration": 120.0
  }
}
```

Returns `prediction` (`1` inlier, `-1` anomaly), `is_anomaly`, the raw `decision_score`, and `alert_category` (`none` or `review`). A review result is not proof of cheating. Invalid feature schema or values return HTTP 400; missing model returns 503.

### `POST /api/exam-sessions`

Creates a `ready` session and returns its ID and one-time bearer token:

```json
{"student_name":"Example Student","student_id":"STU-123","question_count":5}
```

### `POST /api/exam-sessions/{session_id}/start`

Activates a ready session. Requires `Authorization: Bearer <session_token>` and body `{}`.

### `POST /api/exam-sessions/{session_id}/submit`

Submits the written exam once. Requires the bearer token and JSON containing `answers` (one string per question), `proctoring` (the six event counters plus boolean `manual_review_recommended`), and exactly three `verification_responses` strings. Repeated authorized submissions are idempotent; changed data is not written after the first successful submission.

### `PUT /api/exam-sessions/{session_id}/verification`

Saves the three oral verification transcript strings after the written exam was submitted. Requires the same bearer token. The token is held in page memory and is not stored in browser local storage.

## Tests

```bash
python -m pytest -q tests
node --test tests/exam-lifecycle.test.js
```

Python tests cover feature validation, reproducible independent splits, validation-only threshold selection, confusion-matrix/metric calculation, model metadata, held-out evaluation, model loading, and Flask APIs. JavaScript tests cover timer lifecycle, question navigation, typing calculations, and behavior feature aggregation.

## Dependencies and project context

Runtime packages are listed in [`requirements.txt`](exam_hckathon%20-%20Replica/requirements.txt); test tooling is in [`requirements-dev.txt`](exam_hckathon%20-%20Replica/requirements-dev.txt). The project was developed for the IIT Palakkad Hackathon.
