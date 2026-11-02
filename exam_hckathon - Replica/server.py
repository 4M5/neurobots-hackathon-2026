"""Flask application for the exam UI, question generation, and exam APIs."""

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from exam_storage import ExamSessionExpired, ExamSessionNotFound, ExamSessionStateError
from exam_storage import ExamSessionUnauthorized, ExamStore, ExamStoreError, ExamValidationError
from ml_service import BehaviorModelService, InvalidBehaviorFeatures, ModelUnavailableError

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY and genai is not None:
    genai.configure(api_key=API_KEY)

FALLBACK_QUESTIONS = [
    {"id": 1, "text": "Explain the concept of Object-Oriented Programming (OOP) and list its four main principles with brief descriptions.", "topic": "OOP Principles"},
    {"id": 2, "text": "What is the difference between a stack and a queue data structure? Provide examples of real-world applications for each.", "topic": "Data Structures"},
    {"id": 3, "text": "Describe what an API (Application Programming Interface) is and explain the difference between REST and GraphQL APIs.", "topic": "APIs"},
    {"id": 4, "text": "What is machine learning? Explain the difference between supervised and unsupervised learning with examples.", "topic": "Machine Learning"},
    {"id": 5, "text": "Explain the concept of database normalization. What are the benefits and when might you choose to denormalize?", "topic": "Databases"},
]
QUESTION_TEXT_MAX = 2_000
QUESTION_TOPIC_MAX = 100
ALLOWED_CORS_HEADERS = frozenset({"authorization", "content-type"})


def parse_trusted_origins(value=None):
    raw_origins = os.getenv("CORS_ORIGINS", "") if value is None else value
    if isinstance(raw_origins, str):
        origins = [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]
    else:
        origins = [str(origin).strip().rstrip("/") for origin in raw_origins if str(origin).strip()]
    for origin in origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("CORS_ORIGINS entries must be HTTP(S) origins without paths")
    return frozenset(origins)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_server_settings(environ=None):
    environment = os.environ if environ is None else environ
    app_env = environment.get("APP_ENV", "development").strip().lower()
    if app_env not in {"development", "production", "test"}:
        raise ValueError("APP_ENV must be development, production, or test")
    debug = app_env == "development" and _as_bool(environment.get("FLASK_DEBUG"), default=False)
    try:
        port = int(environment.get("FLASK_PORT", "8000"))
    except (TypeError, ValueError):
        raise ValueError("FLASK_PORT must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("FLASK_PORT must be between 1 and 65535")
    host = environment.get("FLASK_HOST", "127.0.0.1").strip() or "127.0.0.1"
    return {"environment": app_env, "debug": debug, "host": host, "port": port}


def validate_generated_questions(raw_text):
    if not isinstance(raw_text, str) or len(raw_text) > 30_000:
        raise ValueError("Generated question response is malformed")
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    questions = json.loads(text)
    if not isinstance(questions, list) or len(questions) != len(FALLBACK_QUESTIONS):
        raise ValueError("Expected exactly five generated questions")

    normalized = []
    for question in questions:
        if not isinstance(question, dict) or set(question) != {"id", "text", "topic"}:
            raise ValueError("Generated question has an invalid shape")
        if isinstance(question["id"], bool) or not isinstance(question["id"], int) or question["id"] < 1:
            raise ValueError("Generated question id must be a positive integer")
        question_text = question["text"]
        topic = question["topic"]
        if not isinstance(question_text, str) or not isinstance(topic, str):
            raise ValueError("Generated question fields must be text")
        question_text = question_text.strip()
        topic = topic.strip()
        if not question_text or len(question_text) > QUESTION_TEXT_MAX or not topic or len(topic) > QUESTION_TOPIC_MAX:
            raise ValueError("Generated question fields are outside allowed lengths")
        if any(ord(char) < 32 and char not in "\n\t" for char in question_text + topic):
            raise ValueError("Generated question contains control characters")
        normalized.append({"id": len(normalized) + 1, "text": question_text, "topic": topic})
    return normalized


def _bearer_token():
    value = request.headers.get("Authorization", "")
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _store_error_response(error):
    return jsonify(error=error.error_code, message=str(error)), error.status_code


def create_app(model_path=None, model_service=None, database_path=None, trusted_origins=None):
    settings = get_server_settings()
    flask_app = Flask(__name__, static_folder=".")
    flask_app.config.update(
        DEBUG=settings["debug"],
        MAX_CONTENT_LENGTH=1_000_000,
        TRUSTED_ORIGINS=parse_trusted_origins(trusted_origins),
        APP_ENV=settings["environment"],
    )
    configured_database = database_path or os.getenv(
        "EXAM_DATABASE_PATH", str(Path(__file__).resolve().parent / "exam_records.sqlite3")
    )
    flask_app.extensions["exam_store"] = ExamStore(configured_database)
    flask_app.extensions["behavior_model_service"] = (
        model_service if model_service is not None else BehaviorModelService(model_path)
    )

    @flask_app.before_request
    def validate_preflight():
        if request.method != "OPTIONS":
            return None
        origin = request.headers.get("Origin")
        if origin and origin not in flask_app.config["TRUSTED_ORIGINS"]:
            return jsonify(error="cors_origin_denied", message="Origin is not trusted"), 403
        requested = {
            header.strip().lower()
            for header in request.headers.get("Access-Control-Request-Headers", "").split(",")
            if header.strip()
        }
        if requested - ALLOWED_CORS_HEADERS:
            return jsonify(error="cors_header_denied", message="Requested header is not allowed"), 403
        return Response(status=204)

    @flask_app.after_request
    def apply_security_headers(response):
        origin = request.headers.get("Origin")
        if origin and origin in flask_app.config["TRUSTED_ORIGINS"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.vary.add("Origin")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), fullscreen=(self)"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
            "script-src 'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; worker-src 'self' blob:; "
            "connect-src 'self' https://cdn.jsdelivr.net; form-action 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @flask_app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(error="request_too_large", message="Request body exceeds the 1 MB limit"), 413

    @flask_app.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return error
        flask_app.logger.exception("Unhandled application error")
        if request.path.startswith("/api/"):
            return jsonify(error="internal_error", message="The request could not be completed"), 500
        return "The request could not be completed", 500

    @flask_app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @flask_app.route("/<path:path>")
    def serve_file(path):
        return send_from_directory(".", path)

    @flask_app.route("/api/predict", methods=["POST"])
    def predict_behavior():
        if not request.is_json:
            return jsonify(error="content_type", message="Request must use application/json"), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"features"}:
            return jsonify(error="invalid_payload", message="Expected a JSON object containing only features"), 400
        try:
            result = flask_app.extensions["behavior_model_service"].predict(payload["features"])
            return jsonify(result)
        except InvalidBehaviorFeatures as exc:
            return jsonify(error="invalid_features", message=str(exc)), 400
        except ModelUnavailableError:
            return jsonify(error="model_unavailable", message="Behavior model is unavailable"), 503
        except Exception:
            flask_app.logger.exception("Behavior model prediction failed")
            return jsonify(error="prediction_failed", message="Behavior prediction failed"), 500

    @flask_app.route("/api/exam-sessions", methods=["POST"])
    def create_exam_session():
        if not request.is_json:
            return jsonify(error="content_type", message="Request must use application/json"), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"student_name", "student_id", "question_count"}:
            return jsonify(error="invalid_payload", message="Invalid exam session request"), 400
        try:
            session = flask_app.extensions["exam_store"].create_session(
                payload["student_name"], payload["student_id"], payload["question_count"]
            )
            return jsonify(session), 201
        except ExamStoreError as exc:
            return _store_error_response(exc)

    @flask_app.route("/api/exam-sessions/<session_id>/start", methods=["POST"])
    def start_exam_session(session_id):
        if not request.is_json or request.get_json(silent=True) != {}:
            return jsonify(error="invalid_payload", message="Expected an empty JSON object"), 400
        try:
            result = flask_app.extensions["exam_store"].start_session(session_id, _bearer_token())
            return jsonify(result)
        except ExamStoreError as exc:
            return _store_error_response(exc)

    @flask_app.route("/api/exam-sessions/<session_id>/submit", methods=["POST"])
    def submit_exam_session(session_id):
        if not request.is_json:
            return jsonify(error="content_type", message="Request must use application/json"), 400
        payload = request.get_json(silent=True)
        try:
            result = flask_app.extensions["exam_store"].submit(session_id, _bearer_token(), payload)
            return jsonify(result)
        except ExamStoreError as exc:
            return _store_error_response(exc)

    @flask_app.route("/api/exam-sessions/<session_id>/verification", methods=["PUT"])
    def save_exam_verification(session_id):
        if not request.is_json:
            return jsonify(error="content_type", message="Request must use application/json"), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"verification_responses"}:
            return jsonify(error="invalid_payload", message="Invalid verification submission"), 400
        try:
            result = flask_app.extensions["exam_store"].save_verification(
                session_id, _bearer_token(), payload["verification_responses"]
            )
            return jsonify(result)
        except ExamStoreError as exc:
            return _store_error_response(exc)

    @flask_app.route("/api/questions", methods=["GET"])
    def get_questions():
        if not API_KEY or genai is None:
            return jsonify([dict(question) for question in FALLBACK_QUESTIONS])
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = """
            Generate exactly five computer-science exam questions covering OOP, data structures,
            web development, machine learning, and databases. Return only a JSON array. Each
            item must contain exactly: id (number), text (string), and topic (string).
            Keep each question text under 2000 characters and each topic under 100 characters.
            """
            response = model.generate_content(prompt)
            generated_text = getattr(response, "text", None)
            return jsonify(validate_generated_questions(generated_text))
        except Exception:
            flask_app.logger.warning("Question generation failed validation or was unavailable; serving fallback questions")
            return jsonify([dict(question) for question in FALLBACK_QUESTIONS])

    return flask_app


app = create_app()


if __name__ == "__main__":
    settings = get_server_settings()
    if settings["environment"] != "development":
        raise SystemExit("The Flask development server is disabled outside development. Run: gunicorn --bind 127.0.0.1:8000 server:app")
    app.run(host=settings["host"], port=settings["port"], debug=settings["debug"], use_reloader=settings["debug"])
