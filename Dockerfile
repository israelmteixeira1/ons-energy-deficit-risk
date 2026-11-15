FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# OS-level deps that some wheels link against on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY app/ ./

# Run as an unprivileged user (nothing in the app needs write access to /app).
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8501

# Healthcheck for streamlit: the /_stcore/health endpoint returns 200 when
# the server is ready. Useful for compose/k8s and docker run --healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else sys.exit(1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
