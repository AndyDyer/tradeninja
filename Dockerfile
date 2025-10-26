FROM python:3.10-slim

WORKDIR /app

# Install deps (add numpy manually since code imports it but toml misses it)
RUN pip install --no-cache-dir streamlit==1.38.0 pandas==2.2.3 numpy==2.1.2

# Copy app code (assumes csv data files are in trade2/ or handled via upload/volume)
COPY . /app

# Expose streamlit default port
EXPOSE 8501

# Run streamlit (override port/host via env if needed, e.g. PORT=80)
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
