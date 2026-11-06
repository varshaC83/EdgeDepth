# EdgeDepth — CPU image for Hugging Face Spaces (Docker SDK), Render, Railway, or local use.
#   docker build -t edgedepth .
#   docker run -p 7860:7860 edgedepth      ->  http://localhost:7860
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Hugging Face Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
