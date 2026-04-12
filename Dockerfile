# Fincent — Hugging Face Spaces (Docker SDK) + Streamlit
# Build context should be this directory (`fincent/`) as the Space repository root.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# faiss-cpu wheels are binary; keep image minimal otherwise
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Hugging Face Spaces sets PORT (often 7860)
ENV PORT=7860
EXPOSE 7860

# Spaces inject PORT; default keeps local `docker run` predictable.
CMD streamlit run streamlit_app.py \
    --server.port=${PORT:-7860} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
