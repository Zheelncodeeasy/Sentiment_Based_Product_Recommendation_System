Heroku deployment notes

1. Ensure the `models/` directory is present in the repo or set `MODEL_ROOT` to a reachable path in Heroku (or use S3 and download at startup).

2. Files added for Heroku:
   - `Procfile` (uses gunicorn to serve `app:app`)
   - `runtime.txt` (Python runtime)
   - `requirements.txt` updated to include `Flask` and `gunicorn`

3. Quick deploy commands:
   - heroku login
   - heroku create <app-name>
   - git add . && git commit -m "Prepare for Heroku"
   - git push heroku main
   - (If using env var MODEL_ROOT) heroku config:set MODEL_ROOT=/app/models
   - heroku logs --tail

4. If model artifacts are large, consider hosting them on S3 and adding startup code to download to `/app/models/` on first run.
