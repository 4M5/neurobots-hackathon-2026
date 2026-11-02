# Security and runtime configuration

Copy `.env.example` to `.env` for local configuration. Keep `.env` private; it is ignored by Git.

- `APP_ENV=development` runs the Flask development server only on `127.0.0.1` by default. Debug/reload are off unless `FLASK_DEBUG=1` is explicitly set in development.
- In production, run the WSGI app with a production server, for example `APP_ENV=production gunicorn --bind 127.0.0.1:8000 server:app`, behind an HTTPS reverse proxy. `python server.py` refuses to start the development server when `APP_ENV` is not `development`.
- Same-origin requests need no CORS entry. Set `CORS_ORIGINS` to a comma-separated list of exact trusted frontend origins only when the frontend is hosted separately.
- `EXAM_DATABASE_PATH` selects the SQLite file. It stores student identifiers, answers, proctoring counters, and oral transcripts. The session bearer token is only returned once and only its hash is stored. The database file is ignored by Git and receives restrictive POSIX permissions where supported.
- SQLite data is not encrypted by this app. For real student records, use a protected/encrypted volume or an encrypted database service, enforce appropriate Windows ACLs, and define a retention policy before deployment.
- `behavior_model.pkl` is a pickle-based joblib artifact. Only load a model produced by a trusted training run. The artifact, generated CSV, database, and local environment files are ignored by Git.
- CORS is not authentication. Exam APIs require an unguessable per-session bearer token; use HTTPS outside localhost so that token and exam content are encrypted in transit.
