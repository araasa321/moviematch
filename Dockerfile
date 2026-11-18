# runtime image (PLAN §11.1). HARD RULE: no ML libraries here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY addon/ addon/

# non-root
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn addon.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
