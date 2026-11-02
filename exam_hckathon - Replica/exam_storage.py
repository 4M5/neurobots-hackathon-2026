"""Authenticated, one-time exam session and submission storage."""

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path


STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$")
PROCTORING_KEYS = frozenset({
    "warnings", "tab_switches", "focus_losses", "paste_events", "gaze_away", "typing_bursts",
    "manual_review_recommended",
})
MAX_ANSWER_LENGTH = 50_000
MAX_TRANSCRIPT_LENGTH = 5_000


class ExamStoreError(Exception):
    status_code = 400
    error_code = "invalid_request"


class ExamValidationError(ExamStoreError):
    pass


class ExamSessionNotFound(ExamStoreError):
    status_code = 404
    error_code = "session_not_found"


class ExamSessionUnauthorized(ExamStoreError):
    status_code = 401
    error_code = "invalid_session_token"


class ExamSessionStateError(ExamStoreError):
    status_code = 409
    error_code = "invalid_session_state"


class ExamSessionExpired(ExamStoreError):
    status_code = 410
    error_code = "session_expired"


def validate_student_details(student_name, student_id):
    if not isinstance(student_name, str):
        raise ExamValidationError("student_name must be a string")
    student_name = student_name.strip()
    if not student_name or len(student_name) > 100 or any(ord(char) < 32 for char in student_name):
        raise ExamValidationError("student_name must contain 1 to 100 printable characters")
    if not isinstance(student_id, str):
        raise ExamValidationError("student_id must be a string")
    student_id = student_id.strip()
    if not STUDENT_ID_PATTERN.fullmatch(student_id):
        raise ExamValidationError("student_id must be 1 to 64 letters, digits, dots, underscores, slashes, or hyphens")
    return student_name, student_id


def validate_question_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
        raise ExamValidationError("question_count must be an integer from 1 to 50")
    return value


def validate_submission(payload, expected_question_count):
    if not isinstance(payload, dict) or set(payload) != {"answers", "proctoring", "verification_responses"}:
        raise ExamValidationError("submission must contain answers, proctoring, and verification_responses only")

    answers = payload["answers"]
    if not isinstance(answers, list) or len(answers) != expected_question_count:
        raise ExamValidationError("answers must match the session question count")
    if any(not isinstance(answer, str) or len(answer) > MAX_ANSWER_LENGTH for answer in answers):
        raise ExamValidationError("each answer must be text no longer than 50000 characters")

    proctoring = payload["proctoring"]
    if not isinstance(proctoring, dict) or set(proctoring) != PROCTORING_KEYS:
        raise ExamValidationError("proctoring must contain the expected numeric counters")
    for key, value in proctoring.items():
        if key == "manual_review_recommended":
            if not isinstance(value, bool):
                raise ExamValidationError("proctoring.manual_review_recommended must be boolean")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100_000:
            raise ExamValidationError(f"proctoring.{key} must be an integer from 0 to 100000")

    responses = payload["verification_responses"]
    if not isinstance(responses, list) or len(responses) != 3:
        raise ExamValidationError("verification_responses must contain exactly three entries")
    if any(not isinstance(response, str) or len(response) > MAX_TRANSCRIPT_LENGTH for response in responses):
        raise ExamValidationError("each verification response must be text no longer than 5000 characters")
    return payload


class ExamStore:
    def __init__(self, database_path, setup_ttl_seconds=1800, active_ttl_seconds=4 * 60 * 60):
        self.database_path = str(database_path)
        self.setup_ttl_seconds = setup_ttl_seconds
        self.active_ttl_seconds = active_ttl_seconds
        if self.database_path == ":memory:":
            raise ValueError("Exam records require a file-backed SQLite database")
        path = Path(self.database_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS exam_sessions (
                    session_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    question_count INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ready', 'active', 'submitted')),
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    submitted_at INTEGER,
                    answers_json TEXT,
                    proctoring_json TEXT,
                    verification_json TEXT
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS exam_sessions_student_idx ON exam_sessions(student_id, created_at)")
            now = int(time.time())
            connection.execute(
                "DELETE FROM exam_sessions WHERE status = 'ready' AND created_at < ?",
                (now - self.setup_ttl_seconds,),
            )
            connection.execute(
                "DELETE FROM exam_sessions WHERE status = 'active' AND started_at < ?",
                (now - self.active_ttl_seconds,),
            )
        try:
            Path(self.database_path).chmod(0o600)
        except OSError:
            # Windows permissions are governed by the containing directory ACL.
            pass

    @staticmethod
    def _token_hash(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _require_uuid(session_id):
        try:
            if str(uuid.UUID(session_id)) != session_id.lower():
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ExamSessionNotFound("Unknown exam session")

    @staticmethod
    def _authorize(row, token):
        if row is None:
            raise ExamSessionNotFound("Unknown exam session")
        if not isinstance(token, str) or not hmac.compare_digest(row["token_hash"], ExamStore._token_hash(token)):
            raise ExamSessionUnauthorized("Invalid exam session token")

    def create_session(self, student_name, student_id, question_count):
        student_name, student_id = validate_student_details(student_name, student_id)
        question_count = validate_question_count(question_count)
        session_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM exam_sessions WHERE status = 'ready' AND created_at < ?",
                (now - self.setup_ttl_seconds,),
            )
            connection.execute(
                "DELETE FROM exam_sessions WHERE status = 'active' AND started_at < ?",
                (now - self.active_ttl_seconds,),
            )
            connection.execute(
                "INSERT INTO exam_sessions (session_id, token_hash, student_name, student_id, question_count, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'ready', ?)",
                (session_id, self._token_hash(token), student_name, student_id, question_count, now),
            )
        return {"session_id": session_id, "session_token": token}

    def start_session(self, session_id, token):
        self._require_uuid(session_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM exam_sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._authorize(row, token)
            if row["status"] == "active":
                connection.commit()
                return {"status": "active", "already_started": True}
            if row["status"] != "ready":
                raise ExamSessionStateError("Exam session cannot be started in its current state")
            now = int(time.time())
            if now - row["created_at"] > self.setup_ttl_seconds:
                raise ExamSessionExpired("Exam setup session expired")
            connection.execute(
                "UPDATE exam_sessions SET status = 'active', started_at = ? WHERE session_id = ? AND status = 'ready'",
                (now, session_id),
            )
            connection.commit()
            return {"status": "active", "already_started": False}

    def submit(self, session_id, token, payload):
        self._require_uuid(session_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM exam_sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._authorize(row, token)
            clean_payload = validate_submission(payload, row["question_count"])
            if row["status"] == "submitted":
                connection.commit()
                return {"session_id": session_id, "status": "submitted", "already_submitted": True}
            if row["status"] != "active":
                raise ExamSessionStateError("Exam session must be active before submission")
            now = int(time.time())
            if row["started_at"] is None or now - row["started_at"] > self.active_ttl_seconds:
                raise ExamSessionExpired("Exam session expired")
            dumps = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
            connection.execute(
                "UPDATE exam_sessions SET status = 'submitted', submitted_at = ?, answers_json = ?, "
                "proctoring_json = ?, verification_json = ? WHERE session_id = ? AND status = 'active'",
                (now, dumps(clean_payload["answers"]), dumps(clean_payload["proctoring"]),
                 dumps(clean_payload["verification_responses"]), session_id),
            )
            connection.commit()
            return {"session_id": session_id, "status": "submitted", "already_submitted": False}

    def save_verification(self, session_id, token, responses):
        self._require_uuid(session_id)
        if not isinstance(responses, list) or len(responses) != 3:
            raise ExamValidationError("verification_responses must contain exactly three entries")
        if any(not isinstance(response, str) or len(response) > MAX_TRANSCRIPT_LENGTH for response in responses):
            raise ExamValidationError("each verification response must be text no longer than 5000 characters")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM exam_sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._authorize(row, token)
            if row["status"] != "submitted":
                raise ExamSessionStateError("Exam must be submitted before verification can be saved")
            now = int(time.time())
            if row["started_at"] is None or now - row["started_at"] > self.active_ttl_seconds:
                raise ExamSessionExpired("Exam session expired")
            connection.execute(
                "UPDATE exam_sessions SET verification_json = ? WHERE session_id = ? AND status = 'submitted'",
                (json.dumps(responses, ensure_ascii=False, allow_nan=False), session_id),
            )
            connection.commit()
            return {"session_id": session_id, "status": "submitted", "verification_saved": True}
