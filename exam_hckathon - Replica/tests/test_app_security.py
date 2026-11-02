import json
import sqlite3

import pytest

import server
from exam_storage import ExamStore, ExamValidationError, validate_submission
from server import FALLBACK_QUESTIONS, create_app, get_server_settings, validate_generated_questions


class StubBehaviorService:
    def predict(self, features):
        return {"prediction": 1, "is_anomaly": False, "decision_score": 0.2, "alert_category": "none"}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "genai", None)
    flask_app = create_app(
        model_service=StubBehaviorService(),
        database_path=tmp_path / "exam-records.sqlite3",
        trusted_origins=["http://localhost:8080"],
    )
    flask_app.config["TESTING"] = True
    return flask_app


def create_and_start(client, question_count=5):
    created = client.post("/api/exam-sessions", json={
        "student_name": "Ada Example",
        "student_id": "IPK-001",
        "question_count": question_count,
    })
    assert created.status_code == 201
    session = created.get_json()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    started = client.post(f"/api/exam-sessions/{session['session_id']}/start", json={}, headers=headers)
    assert started.status_code == 200
    return session, headers


def submission(question_count=5):
    return {
        "answers": [f"Answer {i + 1}" for i in range(question_count)],
        "proctoring": {
            "warnings": 0,
            "tab_switches": 0,
            "focus_losses": 1,
            "paste_events": 0,
            "gaze_away": 0,
            "typing_bursts": 0,
            "manual_review_recommended": False,
        },
        "verification_responses": ["", "", ""],
    }


def test_server_debug_is_off_by_default_and_dev_server_defaults_to_loopback(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    flask_app = create_app(model_service=StubBehaviorService(), database_path=tmp_path / "records.sqlite3")
    assert flask_app.config["DEBUG"] is False
    assert get_server_settings({}) == {"environment": "development", "debug": False, "host": "127.0.0.1", "port": 8000}
    assert get_server_settings({"APP_ENV": "production", "FLASK_DEBUG": "1"})["debug"] is False
    assert get_server_settings({"APP_ENV": "development", "FLASK_DEBUG": "1"})["debug"] is True


def test_cors_allows_only_configured_origins_and_headers(app):
    client = app.test_client()
    trusted = client.options("/api/questions", headers={
        "Origin": "http://localhost:8080",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "content-type",
    })
    assert trusted.status_code == 204
    assert trusted.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"

    denied = client.options("/api/questions", headers={"Origin": "https://untrusted.example"})
    assert denied.status_code == 403
    denied_header = client.options("/api/questions", headers={
        "Origin": "http://localhost:8080",
        "Access-Control-Request-Headers": "x-unapproved-header",
    })
    assert denied_header.status_code == 403


def test_security_headers_and_api_cache_controls_are_set(app):
    response = app.test_client().get("/api/questions")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store"


def test_student_identity_is_validated_and_not_echoed_in_errors(app):
    response = app.test_client().post("/api/exam-sessions", json={
        "student_name": "Ada",
        "student_id": "<script>alert(1)</script>",
        "question_count": 5,
    })
    assert response.status_code == 400
    assert "<script>" not in response.get_data(as_text=True)


def test_exam_state_authentication_persistence_and_idempotent_submit(app, caplog):
    client = app.test_client()
    created = client.post("/api/exam-sessions", json={
        "student_name": "Sensitive Student Name",
        "student_id": "PRIVATE-123",
        "question_count": 5,
    })
    session = created.get_json()
    route = f"/api/exam-sessions/{session['session_id']}"
    payload = submission()
    payload["proctoring"]["manual_review_recommended"] = True

    no_auth = client.post(f"{route}/start", json={})
    assert no_auth.status_code == 401
    too_early = client.post(f"{route}/submit", json=payload, headers={
        "Authorization": f"Bearer {session['session_token']}"
    })
    assert too_early.status_code == 409
    wrong_token = client.post(f"{route}/start", json={}, headers={"Authorization": "Bearer wrong"})
    assert wrong_token.status_code == 401

    headers = {"Authorization": f"Bearer {session['session_token']}"}
    assert client.post(f"{route}/start", json={}, headers=headers).status_code == 200
    invalid = submission()
    invalid["answers"].pop()
    assert client.post(f"{route}/submit", json=invalid, headers=headers).status_code == 400

    saved = client.post(f"{route}/submit", json=payload, headers=headers)
    assert saved.status_code == 200
    assert saved.get_json()["status"] == "submitted"
    changed_payload = submission()
    changed_payload["answers"][0] = "changed after submit"
    duplicate = client.post(f"{route}/submit", json=changed_payload, headers=headers)
    assert duplicate.status_code == 200
    assert duplicate.get_json()["already_submitted"] is True

    database = app.extensions["exam_store"].database_path
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT status, student_id, answers_json, proctoring_json, token_hash FROM exam_sessions WHERE session_id = ?",
            (session["session_id"],),
        ).fetchone()
    assert row[0] == "submitted"
    assert row[1] == "PRIVATE-123"
    assert json.loads(row[2])[0] == "Answer 1"
    assert json.loads(row[3])["manual_review_recommended"] is True
    assert row[4] != session["session_token"]
    assert "PRIVATE-123" not in caplog.text
    assert "Sensitive Student Name" not in caplog.text


def test_verification_response_is_saved_only_for_authenticated_submitted_session(app):
    client = app.test_client()
    session, headers = create_and_start(client)
    route = f"/api/exam-sessions/{session['session_id']}"
    assert client.put(f"{route}/verification", json={"verification_responses": ["a", "b", "c"]}, headers=headers).status_code == 409
    assert client.post(f"{route}/submit", json=submission(), headers=headers).status_code == 200
    update = client.put(f"{route}/verification", json={"verification_responses": ["response A", "", "response C"]}, headers=headers)
    assert update.status_code == 200
    assert update.get_json()["verification_saved"] is True
    unauthorized = client.put(f"{route}/verification", json={"verification_responses": ["", "", ""]})
    assert unauthorized.status_code == 401


def test_gemini_question_json_is_validated_and_normalized():
    response_text = "```json\n" + json.dumps([
        {"id": i + 1, "text": f"Question {i + 1}", "topic": f"Topic {i + 1}"}
        for i in range(5)
    ]) + "\n```"
    questions = validate_generated_questions(response_text)
    assert len(questions) == 5
    assert questions[-1]["id"] == 5
    for invalid in ("not json", "[]", json.dumps([{"id": 1, "text": "x", "topic": "t", "script": "bad"}] * 5)):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            validate_generated_questions(invalid)


def test_invalid_gemini_response_falls_back_to_safe_questions(app, monkeypatch):
    class FakeModel:
        def generate_content(self, _prompt):
            class Result:
                text = '<script>alert("xss")</script>'
            return Result()

    class FakeGemini:
        @staticmethod
        def GenerativeModel(_name):
            return FakeModel()

    monkeypatch.setattr(server, "API_KEY", "configured")
    monkeypatch.setattr(server, "genai", FakeGemini)
    response = app.test_client().get("/api/questions")
    assert response.status_code == 200
    assert response.get_json() == FALLBACK_QUESTIONS


def test_submission_payload_rejects_extra_fields_and_oversized_answers():
    data = submission()
    data["student_id"] = "should not be accepted here"
    with pytest.raises(ExamValidationError):
        validate_submission(data, 5)
    data = submission()
    data["answers"][0] = "x" * 50_001
    with pytest.raises(ExamValidationError):
        validate_submission(data, 5)
    data = submission()
    data["proctoring"]["manual_review_recommended"] = "yes"
    with pytest.raises(ExamValidationError):
        validate_submission(data, 5)
