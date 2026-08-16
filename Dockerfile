FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole app (backend + frontend)
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Run from backend/ (app.py expects frontend/ as a sibling)
WORKDIR /app/backend

EXPOSE 5000

# Uses $PORT when set (Render/Railway/Heroku), falls back to 5000 locally
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]

